// The IAM documents for the two CI roles, as pure functions of names so they
// can be asserted without AWS (`policies.test.ts`).
//
// Threat model: the repo is public, and the account holds other projects
// (other certificates in us-east-1, another CloudFront site, a shared Route 53
// zone with the apex and mail records). A bad `pulumi up` from CI must not be
// able to touch any of them. So:
//
//   - trust is on the OIDC `sub` claim, pinned to a GitHub *environment* for
//     the deploy role (a fork's PR cannot claim it; the environment can require
//     approval) and to the `develop` ref for the read-only preview role;
//   - every Resource is this project's, and where AWS cannot name a resource
//     before it exists (ACM certificates, CloudFront distributions) the
//     destructive actions are conditioned on the project tag instead.

import { CERT_REGION, FUNCTION_PREFIX, PROJECT_TAG } from "../shared";

export interface Names {
  accountId: string;
  /** `owner/repo` */
  repo: string;
  /** The GitHub environment the deploy job runs in. */
  environment: string;
  /** The ref the preview job runs on. */
  previewRef: string;
  stateBucket: string;
  siteBucket: string;
  domain: string;
  zoneId: string;
}

type Statement = {
  Sid: string;
  Effect: "Allow";
  Action: string | string[];
  Resource: string | string[];
  Condition?: Record<string, Record<string, string | string[]>>;
};
export type PolicyDocument = { Version: "2012-10-17"; Statement: Statement[] };

const doc = (Statement: Statement[]): PolicyDocument => ({
  Version: "2012-10-17",
  Statement,
});

const OIDC_HOST = "token.actions.githubusercontent.com";

export const deploySubject = (n: Names) =>
  `repo:${n.repo}:environment:${n.environment}`;
export const previewSubject = (n: Names) => `repo:${n.repo}:ref:${n.previewRef}`;

/** Trust exactly one `sub`; no wildcards — a `repo:owner/name:*` trust would
 *  let any branch or PR workflow assume the role. */
export function trustPolicy(providerArn: string, subject: string) {
  return {
    Version: "2012-10-17",
    Statement: [
      {
        Effect: "Allow",
        Principal: { Federated: providerArn },
        Action: "sts:AssumeRoleWithWebIdentity",
        Condition: {
          StringEquals: {
            [`${OIDC_HOST}:aud`]: "sts.amazonaws.com",
            [`${OIDC_HOST}:sub`]: subject,
          },
        },
      },
    ],
  };
}

const s3Bucket = (b: string) => `arn:aws:s3:::${b}`;
const s3Objects = (b: string) => `arn:aws:s3:::${b}/*`;
const cf = (n: Names, resource: string) =>
  `arn:aws:cloudfront::${n.accountId}:${resource}`;
const certArns = (n: Names) =>
  `arn:aws:acm:${CERT_REGION}:${n.accountId}:certificate/*`;
const zoneArn = (n: Names) => `arn:aws:route53:::hostedzone/${n.zoneId}`;

const tagged = {
  StringEquals: { [`aws:ResourceTag/${PROJECT_TAG.key}`]: PROJECT_TAG.value },
};
const tagging = {
  StringEquals: { [`aws:RequestTag/${PROJECT_TAG.key}`]: PROJECT_TAG.value },
};

/** Pulumi's self-managed backend: checkpoints and locks in the state bucket. */
function stateStatements(n: Names, readOnly: boolean): Statement[] {
  return [
    {
      Sid: "StateBucketList",
      Effect: "Allow",
      Action: ["s3:ListBucket", "s3:GetBucketLocation"],
      Resource: s3Bucket(n.stateBucket),
    },
    {
      Sid: "StateRead",
      Effect: "Allow",
      Action: "s3:GetObject",
      Resource: s3Objects(n.stateBucket),
    },
    readOnly
      ? {
          // `pulumi preview` takes a lock even though it writes no checkpoint.
          Sid: "StateLocks",
          Effect: "Allow",
          Action: ["s3:PutObject", "s3:DeleteObject"],
          Resource: `arn:aws:s3:::${n.stateBucket}/.pulumi/locks/*`,
        }
      : {
          Sid: "StateWrite",
          Effect: "Allow",
          Action: ["s3:PutObject", "s3:DeleteObject"],
          Resource: s3Objects(n.stateBucket),
        },
  ];
}

/** Route 53 reads: the zone lookup (`getZone` by name lists zones — that
 *  action has no resource scope) and change polling. */
function dnsReadStatements(n: Names): Statement[] {
  return [
    {
      Sid: "Route53ListZones",
      Effect: "Allow",
      Action: ["route53:ListHostedZones", "route53:ListHostedZonesByName"],
      Resource: "*",
    },
    {
      Sid: "Route53ReadZone",
      Effect: "Allow",
      Action: [
        "route53:GetHostedZone",
        "route53:ListResourceRecordSets",
        "route53:ListTagsForResource",
      ],
      Resource: zoneArn(n),
    },
    {
      Sid: "Route53GetChange",
      Effect: "Allow",
      Action: "route53:GetChange",
      Resource: "arn:aws:route53:::change/*",
    },
  ];
}

export function deployPolicy(n: Names): PolicyDocument {
  return doc([
    ...stateStatements(n, false),

    // Pulumi reads a dozen bucket sub-resources on every refresh; listing them
    // is brittle, and the resource scope is the boundary that matters.
    {
      Sid: "SiteBucket",
      Effect: "Allow",
      Action: "s3:*",
      Resource: [s3Bucket(n.siteBucket), s3Objects(n.siteBucket)],
    },

    // ── CloudFront ─────────────────────────────────────────────────────────
    {
      Sid: "CloudFrontRead",
      Effect: "Allow",
      Action: ["cloudfront:Get*", "cloudfront:List*", "cloudfront:Describe*"],
      Resource: "*",
    },
    {
      // A distribution's ID is not known until it exists, so it is scoped by
      // tag: another project's distribution in this account is untagged.
      Sid: "CloudFrontDistributionTagged",
      Effect: "Allow",
      Action: [
        "cloudfront:UpdateDistribution",
        "cloudfront:DeleteDistribution",
        "cloudfront:CreateInvalidation",
        "cloudfront:UntagResource",
      ],
      Resource: cf(n, "distribution/*"),
      Condition: tagged,
    },
    {
      Sid: "CloudFrontDistributionCreate",
      Effect: "Allow",
      Action: ["cloudfront:CreateDistribution", "cloudfront:TagResource"],
      Resource: cf(n, "distribution/*"),
      Condition: tagging,
    },
    {
      Sid: "CloudFrontFunctions",
      Effect: "Allow",
      Action: "cloudfront:*",
      Resource: cf(n, `function/${FUNCTION_PREFIX}*`),
    },
    {
      // Both are ID-addressed and hold no data; neither can serve content
      // without a distribution, which is scoped above.
      Sid: "CloudFrontPolicies",
      Effect: "Allow",
      Action: [
        "cloudfront:CreateOriginAccessControl",
        "cloudfront:UpdateOriginAccessControl",
        "cloudfront:DeleteOriginAccessControl",
        "cloudfront:CreateResponseHeadersPolicy",
        "cloudfront:UpdateResponseHeadersPolicy",
        "cloudfront:DeleteResponseHeadersPolicy",
      ],
      Resource: [
        cf(n, "origin-access-control/*"),
        cf(n, "response-headers-policy/*"),
      ],
    },

    // ── ACM (us-east-1 only; other projects' certificates live there too) ──
    {
      Sid: "AcmRead",
      Effect: "Allow",
      Action: [
        "acm:DescribeCertificate",
        "acm:ListCertificates",
        "acm:ListTagsForCertificate",
        "acm:GetCertificate",
      ],
      Resource: "*",
    },
    {
      Sid: "AcmRequest",
      Effect: "Allow",
      Action: "acm:RequestCertificate",
      Resource: "*",
      Condition: {
        StringEquals: { "aws:RequestedRegion": CERT_REGION },
        "ForAllValues:StringEquals": { "acm:DomainNames": [n.domain] },
      },
    },
    {
      Sid: "AcmTagNew",
      Effect: "Allow",
      Action: "acm:AddTagsToCertificate",
      Resource: certArns(n),
      Condition: tagging,
    },
    {
      Sid: "AcmModifyTagged",
      Effect: "Allow",
      Action: ["acm:DeleteCertificate", "acm:RemoveTagsFromCertificate"],
      Resource: certArns(n),
      Condition: tagged,
    },

    // ── Route 53: one zone, and within it only this site's names ──────────
    ...dnsReadStatements(n),
    {
      Sid: "Route53WriteSiteRecords",
      Effect: "Allow",
      Action: "route53:ChangeResourceRecordSets",
      Resource: zoneArn(n),
      Condition: {
        "ForAllValues:StringLike": {
          "route53:ChangeResourceRecordSetsNormalizedRecordNames": [
            n.domain,
            `_*.${n.domain}`,
          ],
        },
        "ForAllValues:StringEquals": {
          "route53:ChangeResourceRecordSetsRecordTypes": ["A", "AAAA", "CNAME"],
        },
      },
    },
  ]);
}

/** Read-only: enough for `pulumi preview --refresh` to show drift. */
export function previewPolicy(n: Names): PolicyDocument {
  return doc([
    ...stateStatements(n, true),
    {
      Sid: "SiteBucketRead",
      Effect: "Allow",
      Action: ["s3:Get*", "s3:List*"],
      Resource: [s3Bucket(n.siteBucket), s3Objects(n.siteBucket)],
    },
    {
      Sid: "CloudFrontRead",
      Effect: "Allow",
      Action: ["cloudfront:Get*", "cloudfront:List*", "cloudfront:Describe*"],
      Resource: "*",
    },
    {
      Sid: "AcmRead",
      Effect: "Allow",
      Action: [
        "acm:DescribeCertificate",
        "acm:ListCertificates",
        "acm:ListTagsForCertificate",
      ],
      Resource: "*",
    },
    ...dnsReadStatements(n),
  ]);
}
