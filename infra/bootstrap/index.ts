// Run once, locally, by an admin with ~/.aws credentials (#57 Phase 2). It
// creates what CI needs in order to authenticate at all, so CI cannot create
// it — the chicken-and-egg is resolved by a human, on purpose.
//
//   - a deploy role, assumable only from the `production` GitHub environment
//   - a read-only preview role, assumable only from `refs/heads/develop`
//
// The GitHub OIDC provider is an account-wide singleton (one per issuer URL)
// that other projects' roles already trust, so it is looked up, never created
// or owned here — the same rule as the Route 53 zone. `docs/guides/deployment.md`
// §8 has the one CLI command for an account that lacks it.

import * as aws from "@pulumi/aws";
import * as pulumi from "@pulumi/pulumi";

import { PROJECT_TAG } from "../shared";
import {
  deployPolicy,
  deploySubject,
  previewPolicy,
  previewSubject,
  trustPolicy,
  type Names,
} from "./policies";

const config = new pulumi.Config("bootstrap");
const account = aws.getCallerIdentityOutput();

const provider = aws.iam.getOpenIdConnectProviderOutput({
  url: "https://token.actions.githubusercontent.com",
});

const names = account.accountId.apply(
  (accountId): Names => ({
    accountId,
    repo: config.require("repo"),
    environment: config.get("environment") ?? "production",
    previewRef: config.get("previewRef") ?? "refs/heads/develop",
    stateBucket: config.require("stateBucket"),
    siteBucket: config.require("siteBucket"),
    domain: config.require("domain"),
    zoneId: config.require("zoneId"),
  }),
);

const tags = { [PROJECT_TAG.key]: PROJECT_TAG.value };

function ciRole(
  name: string,
  description: string,
  subject: pulumi.Output<string>,
  policy: pulumi.Output<object>,
) {
  const role = new aws.iam.Role(name, {
    name: `model-playground-${name}`,
    description,
    maxSessionDuration: 3600,
    assumeRolePolicy: pulumi
      .all([provider.arn, subject])
      .apply(([arn, sub]) => JSON.stringify(trustPolicy(arn, sub))),
    tags,
  });
  new aws.iam.RolePolicy(name, {
    role: role.id,
    name: `model-playground-${name}`,
    policy: policy.apply((p) => JSON.stringify(p)),
  });
  return role;
}

const deploy = ciRole(
  "deploy",
  "GitHub Actions (production environment): pulumi up on infra/site + upload",
  names.apply(deploySubject),
  names.apply(deployPolicy),
);

const preview = ciRole(
  "preview",
  "GitHub Actions (develop): read-only pulumi preview on infra/site",
  names.apply(previewSubject),
  names.apply(previewPolicy),
);

/** Set as the `AWS_DEPLOY_ROLE_ARN` variable on the `production` environment. */
export const deployRoleArn = deploy.arn;
/** Set as the repository variable `AWS_PREVIEW_ROLE_ARN`. */
export const previewRoleArn = preview.arn;
export const oidcProviderArn = provider.arn;
