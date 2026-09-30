// The static frontend (#57): `frontend/dist` in a private S3 bucket, served by
// CloudFront over HTTPS at `site:domain`.
//
// HTTPS is functional, not hardening — `navigator.gpu` exists only in a secure
// context, so a plain-HTTP deploy makes every model page report `unsupported`.
//
// Files are NOT Pulumi resources. `scripts/deploy-frontend.sh` uploads them
// after `pulumi up`, ordered so an open tab never loses a chunk it references;
// one `BucketObject` per file would delete the previous build's hashed chunks
// the moment the new build landed. See docs/guides/deployment.md §10.6.

import * as aws from "@pulumi/aws";
import * as pulumi from "@pulumi/pulumi";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  CACHING_OPTIMIZED_POLICY_ID,
  CERT_REGION,
  FUNCTION_PREFIX,
  PROJECT_TAG,
} from "../shared";

const config = new pulumi.Config("site");
/** e.g. playground.benedictthekkel.com — no underscores: ACM refuses them. */
const domain = config.require("domain");
/** The existing public zone. Looked up, never created, imported or deleted. */
const zoneName = config.require("zoneName");
/** Named rather than generated, so the bootstrap stack can scope IAM to it. */
const siteBucket = config.require("bucketName");
/** Days a superseded hashed asset stays servable to tabs opened before a deploy. */
const staleAssetDays = config.getNumber("staleAssetDays") ?? 30;

const tags = { [PROJECT_TAG.key]: PROJECT_TAG.value };

// ── Bucket ──────────────────────────────────────────────────────────────────

const bucket = new aws.s3.Bucket("site", { bucket: siteBucket, tags });

new aws.s3.BucketPublicAccessBlock("site", {
  bucket: bucket.id,
  blockPublicAcls: true,
  blockPublicPolicy: true,
  ignorePublicAcls: true,
  restrictPublicBuckets: true,
});

new aws.s3.BucketOwnershipControls("site", {
  bucket: bucket.id,
  rule: { objectOwnership: "BucketOwnerEnforced" },
});

new aws.s3.BucketServerSideEncryptionConfiguration("site", {
  bucket: bucket.id,
  rules: [{ applyServerSideEncryptionByDefault: { sseAlgorithm: "AES256" } }],
});

// The upload never deletes. When a build stops referencing a hashed asset the
// script re-writes it in place tagged `stale=true` — which resets its creation
// date — so this rule removes it `staleAssetDays` after it went stale, not
// after it was first uploaded. An asset a later build brings back is
// re-uploaded untagged, which takes it out of the rule.
new aws.s3.BucketLifecycleConfiguration("site", {
  bucket: bucket.id,
  rules: [
    {
      id: "expire-stale-assets",
      status: "Enabled",
      filter: { tag: { key: "stale", value: "true" } },
      expiration: { days: staleAssetDays },
    },
    {
      id: "abort-incomplete-uploads",
      status: "Enabled",
      filter: {},
      abortIncompleteMultipartUpload: { daysAfterInitiation: 1 },
    },
  ],
});

// ── CloudFront pieces ───────────────────────────────────────────────────────

const oac = new aws.cloudfront.OriginAccessControl("site", {
  name: siteBucket,
  description: `CloudFront → ${siteBucket}`,
  originAccessControlOriginType: "s3",
  signingBehavior: "always",
  signingProtocol: "sigv4",
});

const spaRewrite = new aws.cloudfront.Function("spa-rewrite", {
  name: `${FUNCTION_PREFIX}spa-rewrite`,
  runtime: "cloudfront-js-2.0",
  comment: "SPA deep links: extensionless paths → /index.html",
  publish: true,
  code: readFileSync(join(__dirname, "spa-rewrite.js"), "utf8"),
});

// No Content-Security-Policy, on purpose: a wrong one silently breaks every
// model load (weights come from the Hugging Face CDN). deployment.md §1.
const headers = new aws.cloudfront.ResponseHeadersPolicy("site", {
  name: `${FUNCTION_PREFIX}headers`,
  comment: "Security headers for the static frontend; no CSP",
  securityHeadersConfig: {
    strictTransportSecurity: {
      accessControlMaxAgeSec: 31536000,
      includeSubdomains: false,
      override: true,
    },
    contentTypeOptions: { override: true },
    frameOptions: { frameOption: "DENY", override: true },
    referrerPolicy: {
      referrerPolicy: "strict-origin-when-cross-origin",
      override: true,
    },
  },
});

// ── Certificate (us-east-1) and DNS ─────────────────────────────────────────

const zone = aws.route53.getZoneOutput({ name: zoneName, privateZone: false });

// CloudFront reads viewer certificates only from us-east-1, whatever region
// the rest of the stack is in.
const cert = new aws.acm.Certificate("site", {
  region: CERT_REGION,
  domainName: domain,
  validationMethod: "DNS",
  tags,
});

// One name, so exactly one validation record. Declared directly rather than
// inside an `apply`, so `pulumi preview` can show it before it exists.
const dvo = cert.domainValidationOptions[0];
const validationRecord = new aws.route53.Record("cert-validation", {
  zoneId: zone.zoneId,
  name: dvo.resourceRecordName,
  type: dvo.resourceRecordType,
  records: [dvo.resourceRecordValue],
  ttl: 300,
  allowOverwrite: true,
});

const validation = new aws.acm.CertificateValidation("site", {
  region: CERT_REGION,
  certificateArn: cert.arn,
  validationRecordFqdns: [validationRecord.fqdn],
});

// ── Distribution ────────────────────────────────────────────────────────────

const originId = "s3-site";

const distribution = new aws.cloudfront.Distribution("site", {
  enabled: true,
  comment: domain,
  aliases: [domain],
  defaultRootObject: "index.html",
  httpVersion: "http2and3",
  isIpv6Enabled: true,
  priceClass: "PriceClass_100",
  origins: [
    {
      originId,
      domainName: bucket.bucketRegionalDomainName,
      originAccessControlId: oac.id,
    },
  ],
  defaultCacheBehavior: {
    targetOriginId: originId,
    viewerProtocolPolicy: "redirect-to-https",
    allowedMethods: ["GET", "HEAD", "OPTIONS"],
    cachedMethods: ["GET", "HEAD"],
    compress: true,
    // Honours the Cache-Control the upload script sets per file: immutable
    // for /assets, no-cache for index.html.
    cachePolicyId: CACHING_OPTIMIZED_POLICY_ID,
    responseHeadersPolicyId: headers.id,
    functionAssociations: [
      { eventType: "viewer-request", functionArn: spaRewrite.arn },
    ],
  },
  // Deliberately no `customErrorResponses`: the SPA fallback is the function
  // above, so a real 403/404 — including a future /api origin's — stays one.
  restrictions: { geoRestriction: { restrictionType: "none" } },
  viewerCertificate: {
    acmCertificateArn: validation.certificateArn,
    sslSupportMethod: "sni-only",
    minimumProtocolVersion: "TLSv1.2_2021",
  },
  tags,
});

// Only this distribution may read the bucket; nothing else, and nobody public.
new aws.s3.BucketPolicy("site", {
  bucket: bucket.id,
  policy: pulumi
    .all([bucket.arn, distribution.arn])
    .apply(([bucketArn, distributionArn]) =>
      JSON.stringify({
        Version: "2012-10-17",
        Statement: [
          {
            Sid: "CloudFrontReadOnly",
            Effect: "Allow",
            Principal: { Service: "cloudfront.amazonaws.com" },
            Action: "s3:GetObject",
            Resource: `${bucketArn}/*`,
            Condition: { StringEquals: { "AWS:SourceArn": distributionArn } },
          },
        ],
      }),
    ),
});

for (const type of ["A", "AAAA"] as const) {
  new aws.route53.Record(`alias-${type.toLowerCase()}`, {
    zoneId: zone.zoneId,
    name: domain,
    type,
    aliases: [
      {
        name: distribution.domainName,
        zoneId: distribution.hostedZoneId,
        evaluateTargetHealth: false,
      },
    ],
  });
}

export const bucketName = bucket.bucket;
export const distributionId = distribution.id;
export const url = `https://${domain}`;
export const cloudfrontDomain = distribution.domainName;
