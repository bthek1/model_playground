import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import {
  deployPolicy,
  deploySubject,
  previewPolicy,
  previewSubject,
  trustPolicy,
  type Names,
  type PolicyDocument,
} from "./policies";

const names: Names = {
  accountId: "762233760445",
  repo: "bthek1/model_playground",
  environment: "production",
  previewRef: "refs/heads/develop",
  stateBucket: "model-playground-pulumi-state-762233760445",
  siteBucket: "model-playground-site-762233760445",
  domain: "playground.benedictthekkel.com",
  zoneId: "Z08957092MQHNE94LIHJK",
};

const list = (x: string | string[]) => (Array.isArray(x) ? x : [x]);
const statements = (p: PolicyDocument) => p.Statement;
const actions = (p: PolicyDocument) => p.Statement.flatMap((s) => list(s.Action));
const isRead = (action: string) =>
  /^[a-z0-9]+:(Get|List|Describe)/.test(action);

describe("trust", () => {
  const arn = "arn:aws:iam::762233760445:oidc-provider/token.actions.githubusercontent.com";

  it("the deploy role trusts the production environment, not a branch", () => {
    expect(deploySubject(names)).toBe(
      "repo:bthek1/model_playground:environment:production",
    );
  });

  it("the preview role trusts develop only", () => {
    expect(previewSubject(names)).toBe(
      "repo:bthek1/model_playground:ref:refs/heads/develop",
    );
  });

  it("pins sub and aud with StringEquals — no wildcard can widen it", () => {
    const trust = trustPolicy(arn, deploySubject(names));
    const [s] = trust.Statement;
    expect(trust.Statement).toHaveLength(1);
    expect(s.Principal).toEqual({ Federated: arn });
    expect(s.Action).toBe("sts:AssumeRoleWithWebIdentity");
    expect(Object.keys(s.Condition)).toEqual(["StringEquals"]);
    expect(s.Condition.StringEquals).toEqual({
      "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
      "token.actions.githubusercontent.com:sub":
        "repo:bthek1/model_playground:environment:production",
    });
    expect(JSON.stringify(trust)).not.toContain("*");
  });
});

describe.each([
  ["deploy", deployPolicy(names)],
  ["preview", previewPolicy(names)],
])("the %s policy", (_, policy) => {
  it("grants no *:* and no service-wide action on every resource", () => {
    for (const s of statements(policy)) {
      for (const action of list(s.Action)) {
        expect(action).not.toBe("*");
        if (list(s.Resource).includes("*") && !s.Condition) {
          // Unscoped resources only for reads (List* actions have no ARN).
          expect(isRead(action), `${s.Sid}: ${action}`).toBe(true);
        }
      }
    }
  });

  it("names every statement, so a denial in CloudTrail is attributable", () => {
    const sids = statements(policy).map((s) => s.Sid);
    expect(new Set(sids).size).toBe(sids.length);
  });

  it("touches Route 53 only in this zone (or read-only listing)", () => {
    for (const s of statements(policy)) {
      if (!list(s.Action).some((a) => a.startsWith("route53:"))) continue;
      for (const r of list(s.Resource)) {
        expect([
          "*",
          "arn:aws:route53:::hostedzone/Z08957092MQHNE94LIHJK",
          "arn:aws:route53:::change/*",
        ]).toContain(r);
      }
    }
  });
});

describe("the deploy policy", () => {
  const policy = deployPolicy(names);
  const sid = (id: string) => {
    const found = policy.Statement.find((s) => s.Sid === id);
    expect(found, id).toBeDefined();
    return found!;
  };

  it("writes only the playground names — never the apex, mail or other subdomains", () => {
    const s = sid("Route53WriteSiteRecords");
    expect(s.Action).toBe("route53:ChangeResourceRecordSets");
    expect(s.Resource).toBe("arn:aws:route53:::hostedzone/Z08957092MQHNE94LIHJK");
    expect(s.Condition).toEqual({
      "ForAllValues:StringLike": {
        "route53:ChangeResourceRecordSetsNormalizedRecordNames": [
          "playground.benedictthekkel.com",
          "_*.playground.benedictthekkel.com",
        ],
      },
      "ForAllValues:StringEquals": {
        "route53:ChangeResourceRecordSetsRecordTypes": ["A", "AAAA", "CNAME"],
      },
    });
  });

  it("can delete only certificates and distributions tagged as this project's", () => {
    for (const s of policy.Statement) {
      const destructive = list(s.Action).filter((a) =>
        /^(acm:DeleteCertificate|cloudfront:(Delete|Update)Distribution)$/.test(a),
      );
      if (destructive.length === 0) continue;
      expect(s.Condition, s.Sid).toEqual({
        StringEquals: { "aws:ResourceTag/project": "model-playground" },
      });
    }
  });

  it("requests certificates for this domain in us-east-1 only", () => {
    expect(sid("AcmRequest").Condition).toEqual({
      StringEquals: { "aws:RequestedRegion": "us-east-1" },
      "ForAllValues:StringEquals": {
        "acm:DomainNames": ["playground.benedictthekkel.com"],
      },
    });
  });

  it("can invalidate (the upload script needs it)", () => {
    expect(actions(policy)).toContain("cloudfront:CreateInvalidation");
  });

  it("scopes the site bucket's s3:* to that bucket", () => {
    expect(sid("SiteBucket").Resource).toEqual([
      "arn:aws:s3:::model-playground-site-762233760445",
      "arn:aws:s3:::model-playground-site-762233760445/*",
    ]);
  });
});

describe("the preview policy", () => {
  it("writes nothing but Pulumi's lock files", () => {
    const writes = previewPolicy(names).Statement.filter((s) =>
      list(s.Action).some((a) => !isRead(a)),
    );
    expect(writes.map((s) => s.Sid)).toEqual(["StateLocks"]);
    expect(writes[0].Resource).toBe(
      "arn:aws:s3:::model-playground-pulumi-state-762233760445/.pulumi/locks/*",
    );
  });
});

// The bootstrap stack writes IAM for names the site stack has not created yet.
// Nothing links the two configs at runtime, so check they agree here.
describe("stack configs agree", () => {
  const read = (path: string) =>
    Object.fromEntries(
      [...readFileSync(join(__dirname, path), "utf8").matchAll(/^\s+([\w:]+):\s*(\S+)\s*$/gm)].map(
        (m) => [m[1], m[2]],
      ),
    );
  const bootstrap = read("Pulumi.production.yaml");
  const site = read("../site/Pulumi.production.yaml");

  it("on the bucket, the domain and the region", () => {
    expect(bootstrap["bootstrap:siteBucket"]).toBe(site["site:bucketName"]);
    expect(bootstrap["bootstrap:domain"]).toBe(site["site:domain"]);
    expect(bootstrap["aws:region"]).toBe(site["aws:region"]);
  });

  it("and the test's names are the committed ones", () => {
    expect(bootstrap["bootstrap:repo"]).toBe(names.repo);
    expect(bootstrap["bootstrap:stateBucket"]).toBe(names.stateBucket);
    expect(bootstrap["bootstrap:zoneId"]).toBe(names.zoneId);
    expect(site["site:zoneName"]).toBe("benedictthekkel.com");
  });

  it("no hostname has an underscore (ACM refuses them; no HTTPS, no WebGPU)", () => {
    expect(site["site:domain"]).not.toContain("_");
  });
});
