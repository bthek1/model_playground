// The site stack under Pulumi's mocks: no AWS, no CLI, no state. Every
// resource the program declares is recorded with the inputs it was given, and
// the assertions are on those inputs — the properties that, got wrong, leave a
// working-looking site that is public, or cached wrong, or masking errors.

import * as pulumi from "@pulumi/pulumi";
import { beforeAll, describe, expect, it } from "vitest";

import { CERT_REGION, PROJECT_TAG } from "../shared";

type Recorded = { type: string; name: string; inputs: Record<string, any> };
const resources: Recorded[] = [];
const calls: { token: string; inputs: Record<string, any> }[] = [];

const ZONE_ID = "Z08957092MQHNE94LIHJK";
const DOMAIN = "playground.benedictthekkel.com";
const BUCKET = "model-playground-site-762233760445";

let stack: typeof import("./index");

beforeAll(async () => {
  pulumi.runtime.setAllConfig({
    "site:domain": DOMAIN,
    "site:zoneName": "benedictthekkel.com",
    "site:bucketName": BUCKET,
  });
  await pulumi.runtime.setMocks(
    {
      newResource(args) {
        resources.push({ type: args.type, name: args.name, inputs: args.inputs });
        const state: Record<string, any> = {
          ...args.inputs,
          arn: `arn:aws:mock:::${args.type}/${args.name}`,
        };
        if (args.type === "aws:s3/bucket:Bucket") {
          state.bucketRegionalDomainName = `${args.inputs.bucket}.s3.ap-southeast-2.amazonaws.com`;
        }
        if (args.type === "aws:acm/certificate:Certificate") {
          state.domainValidationOptions = [
            {
              domainName: DOMAIN,
              resourceRecordName: `_abc.${DOMAIN}.`,
              resourceRecordType: "CNAME",
              resourceRecordValue: "_xyz.acm-validations.aws.",
            },
          ];
        }
        if (args.type === "aws:cloudfront/distribution:Distribution") {
          state.domainName = "d111111abcdef8.cloudfront.net";
          state.hostedZoneId = "Z2FDTNDATAQYW2";
        }
        if (args.type === "aws:route53/record:Record") {
          state.fqdn = args.inputs.name;
        }
        return { id: `${args.name}-id`, state };
      },
      call(args) {
        calls.push({ token: args.token, inputs: args.inputs });
        if (args.token === "aws:route53/getZone:getZone") {
          return { ...args.inputs, zoneId: ZONE_ID, id: ZONE_ID };
        }
        return args.inputs;
      },
    },
    "site",
    "production",
  );
  stack = await import("./index");
  // The bucket policy and alias records wait on the distribution's outputs;
  // let those applies run so every resource is registered before asserting.
  await value(stack.distributionId);
  await new Promise((resolve) => setTimeout(resolve, 50));
});

function value<T>(output: pulumi.Output<T> | T): Promise<T> {
  return pulumi.Output.isInstance(output)
    ? new Promise((resolve) => (output as pulumi.Output<T>).apply(resolve))
    : Promise.resolve(output as T);
}

const ofType = (type: string) => resources.filter((r) => r.type === type);
const one = (type: string) => {
  const found = ofType(type);
  expect(found, type).toHaveLength(1);
  return found[0].inputs;
};

describe("the bucket", () => {
  it("blocks every form of public access", () => {
    expect(one("aws:s3/bucketPublicAccessBlock:BucketPublicAccessBlock")).toMatchObject({
      blockPublicAcls: true,
      blockPublicPolicy: true,
      ignorePublicAcls: true,
      restrictPublicBuckets: true,
    });
  });

  it("is named from config, so the bootstrap stack's IAM can name it", () => {
    expect(one("aws:s3/bucket:Bucket").bucket).toBe(BUCKET);
  });

  it("lets exactly one principal read it: this distribution", async () => {
    const policy = JSON.parse(one("aws:s3/bucketPolicy:BucketPolicy").policy);
    const distributionArn =
      "arn:aws:mock:::aws:cloudfront/distribution:Distribution/site";
    expect(policy.Statement).toHaveLength(1);
    const [statement] = policy.Statement;
    expect(statement.Effect).toBe("Allow");
    expect(statement.Principal).toEqual({ Service: "cloudfront.amazonaws.com" });
    expect(statement.Action).toBe("s3:GetObject");
    expect(statement.Condition).toEqual({
      StringEquals: { "AWS:SourceArn": distributionArn },
    });
  });

  it("expires only assets the upload script tagged stale", () => {
    const { rules } = one(
      "aws:s3/bucketLifecycleConfiguration:BucketLifecycleConfiguration",
    );
    const expiring = rules.filter((r: any) => r.expiration);
    expect(expiring).toHaveLength(1);
    expect(expiring[0].filter).toEqual({ tag: { key: "stale", value: "true" } });
    expect(expiring[0].expiration.days).toBeGreaterThanOrEqual(7);
  });
});

describe("the distribution", () => {
  const dist = () => one("aws:cloudfront/distribution:Distribution");

  it("redirects plain HTTP to HTTPS (WebGPU needs a secure context)", () => {
    expect(dist().defaultCacheBehavior.viewerProtocolPolicy).toBe(
      "redirect-to-https",
    );
  });

  it("has no custom error responses — the SPA fallback is the function", () => {
    expect(dist().customErrorResponses ?? []).toEqual([]);
    expect(dist().defaultCacheBehavior.functionAssociations).toEqual([
      {
        eventType: "viewer-request",
        functionArn:
          "arn:aws:mock:::aws:cloudfront/function:Function/spa-rewrite",
      },
    ]);
  });

  it("serves the domain over SNI with the validated certificate", () => {
    expect(dist().aliases).toEqual([DOMAIN]);
    expect(dist().viewerCertificate).toMatchObject({
      acmCertificateArn: "arn:aws:mock:::aws:acm/certificate:Certificate/site",
      sslSupportMethod: "sni-only",
    });
  });

  it("uses HTTP/2 + HTTP/3 and the cheapest price class", () => {
    expect(dist()).toMatchObject({
      httpVersion: "http2and3",
      priceClass: "PriceClass_100",
      isIpv6Enabled: true,
    });
  });

  it("reaches the bucket only through Origin Access Control", () => {
    const [origin] = dist().origins;
    expect(origin.originAccessControlId).toBe("site-id");
    expect(origin.s3OriginConfig).toBeUndefined();
    expect(one("aws:cloudfront/originAccessControl:OriginAccessControl")).toMatchObject({
      originAccessControlOriginType: "s3",
      signingBehavior: "always",
    });
  });

  it("ships no Content-Security-Policy", () => {
    const policy = one(
      "aws:cloudfront/responseHeadersPolicy:ResponseHeadersPolicy",
    );
    expect(policy.securityHeadersConfig.contentSecurityPolicy).toBeUndefined();
    expect(policy.securityHeadersConfig.strictTransportSecurity).toBeDefined();
  });

  it("carries the project tag the deploy role's conditions key on", () => {
    expect(dist().tags).toEqual({ [PROJECT_TAG.key]: PROJECT_TAG.value });
  });
});

describe("same-origin analytics: /ingest/* (#60)", () => {
  const dist = () => one("aws:cloudfront/distribution:Distribution");
  const ingest = () =>
    (dist().orderedCacheBehaviors ?? []).find(
      (b: { pathPattern: string }) => b.pathPattern === "/ingest/*",
    );

  it("is an ordered behaviour, so it is matched before the default", () => {
    // Without it the default behaviour's spa-rewrite answers /ingest/e/ with
    // index.html and a 200, and every event vanishes silently.
    expect(dist().orderedCacheBehaviors.map((b: { pathPattern: string }) => b.pathPattern)).toEqual([
      "/ingest/*",
    ]);
  });

  it("goes to PostHog's ingestion host over HTTPS", () => {
    const origin = dist().origins.find(
      (o: { originId: string }) => o.originId === ingest().targetOriginId,
    );
    expect(origin.domainName).toBe("us.i.posthog.com");
    expect(origin.customOriginConfig.originProtocolPolicy).toBe("https-only");
    expect(origin.originAccessControlId).toBeUndefined();
  });

  it("caches nothing and forwards everything but Host", () => {
    expect(ingest()).toMatchObject({
      cachePolicyId: "4135ea2d-6df8-44a3-9df3-4b5a84be39ad",
      originRequestPolicyId: "b689b0a8-53d0-40ab-baf2-68738e2966ac",
      viewerProtocolPolicy: "https-only",
    });
    expect(ingest().allowedMethods).toEqual(
      expect.arrayContaining(["POST", "OPTIONS", "GET"]),
    );
  });

  it("carries only the prefix-strip function, never the SPA rewrite", () => {
    expect(ingest().functionAssociations).toEqual([
      {
        eventType: "viewer-request",
        functionArn:
          "arn:aws:mock:::aws:cloudfront/function:Function/ingest-strip",
      },
    ]);
    expect(
      resources.find((r) => r.name === "ingest-strip")?.inputs.name,
    ).toBe("model-playground-ingest-strip"); // inside the deploy role's function scope
  });

  it("has no /static/* twin — the shipped SDK build loads no scripts", () => {
    expect(dist().origins).toHaveLength(2);
  });
});

describe("certificate and DNS", () => {
  it("requests the certificate in us-east-1, tagged", () => {
    expect(one("aws:acm/certificate:Certificate")).toMatchObject({
      region: CERT_REGION,
      domainName: DOMAIN,
      validationMethod: "DNS",
      tags: { [PROJECT_TAG.key]: PROJECT_TAG.value },
    });
    expect(one("aws:acm/certificateValidation:CertificateValidation").region).toBe(
      CERT_REGION,
    );
  });

  it("looks the zone up and never creates one", () => {
    expect(ofType("aws:route53/zone:Zone")).toEqual([]);
    const lookup = calls.find((c) => c.token === "aws:route53/getZone:getZone");
    expect(lookup?.inputs).toMatchObject({
      name: "benedictthekkel.com",
      privateZone: false,
    });
  });

  it("writes only playground records into the shared zone", () => {
    const records = ofType("aws:route53/record:Record").map((r) => r.inputs);
    expect(records.every((r) => r.zoneId === ZONE_ID)).toBe(true);
    for (const r of records) {
      expect(r.name.replace(/\.$/, "")).toMatch(
        /^(playground\.benedictthekkel\.com|_[^.]+\.playground\.benedictthekkel\.com)$/,
      );
    }
    const aliases = records.filter((r) => r.aliases);
    expect(aliases.map((r) => r.type).sort()).toEqual(["A", "AAAA"]);
  });
});

describe("outputs", () => {
  it("exposes what the upload script and smoke test need", async () => {
    expect(await value(stack.url)).toBe(`https://${DOMAIN}`);
    expect(await value(stack.bucketName)).toBe(BUCKET);
    expect(await value(stack.distributionId)).toBe("site-id");
    expect(await value(stack.cloudfrontDomain)).toBe(
      "d111111abcdef8.cloudfront.net",
    );
  });
});
