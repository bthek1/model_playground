// The bootstrap stack under Pulumi's mocks. `policies.test.ts` pins the
// documents; this pins the wiring — that each role gets the right document and
// trust, and that the shared OIDC provider is looked up rather than created
// (creating it would fail in this account, or worse, take ownership of it).

import * as pulumi from "@pulumi/pulumi";
import { beforeAll, describe, expect, it } from "vitest";

import { deployPolicy, previewPolicy, type Names } from "./policies";

type Recorded = { type: string; name: string; inputs: Record<string, any> };
const resources: Recorded[] = [];
const calls: { token: string; inputs: Record<string, any> }[] = [];

const PROVIDER_ARN =
  "arn:aws:iam::762233760445:oidc-provider/token.actions.githubusercontent.com";

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

let stack: typeof import("./index");

beforeAll(async () => {
  pulumi.runtime.setAllConfig({
    "bootstrap:repo": names.repo,
    "bootstrap:stateBucket": names.stateBucket,
    "bootstrap:siteBucket": names.siteBucket,
    "bootstrap:domain": names.domain,
    "bootstrap:zoneId": names.zoneId,
  });
  await pulumi.runtime.setMocks(
    {
      newResource(args) {
        resources.push({ type: args.type, name: args.name, inputs: args.inputs });
        return {
          id: `${args.name}-id`,
          state: { ...args.inputs, arn: `arn:aws:iam::762233760445:role/${args.inputs.name}` },
        };
      },
      call(args) {
        calls.push({ token: args.token, inputs: args.inputs });
        if (args.token === "aws:index/getCallerIdentity:getCallerIdentity") {
          return { accountId: names.accountId, arn: "arn:aws:iam::762233760445:user/admin", userId: "x" };
        }
        if (args.token === "aws:iam/getOpenIdConnectProvider:getOpenIdConnectProvider") {
          return { ...args.inputs, arn: PROVIDER_ARN };
        }
        return args.inputs;
      },
    },
    "bootstrap",
    "production",
  );
  stack = await import("./index");
  await new Promise<void>((resolve) => stack.deployRoleArn.apply(() => resolve()));
  await new Promise((resolve) => setTimeout(resolve, 50));
});

const role = (name: string) =>
  resources.find((r) => r.type === "aws:iam/role:Role" && r.name === name)!.inputs;
const inline = (name: string) =>
  resources.find((r) => r.type === "aws:iam/rolePolicy:RolePolicy" && r.name === name)!
    .inputs;

describe("bootstrap stack", () => {
  it("looks the GitHub OIDC provider up and never creates one", () => {
    expect(
      resources.filter((r) => r.type.startsWith("aws:iam/openIdConnectProvider")),
    ).toEqual([]);
    const lookup = calls.find(
      (c) => c.token === "aws:iam/getOpenIdConnectProvider:getOpenIdConnectProvider",
    );
    expect(lookup?.inputs.url).toBe("https://token.actions.githubusercontent.com");
  });

  it("creates exactly two roles, each with one inline policy", () => {
    expect(resources.map((r) => `${r.type}:${r.name}`).sort()).toEqual([
      "aws:iam/role:Role:deploy",
      "aws:iam/role:Role:preview",
      "aws:iam/rolePolicy:RolePolicy:deploy",
      "aws:iam/rolePolicy:RolePolicy:preview",
    ]);
  });

  it.each([
    ["deploy", "repo:bthek1/model_playground:environment:production"],
    ["preview", "repo:bthek1/model_playground:ref:refs/heads/develop"],
  ])("the %s role trusts the looked-up provider for %s only", (name, sub) => {
    const trust = JSON.parse(role(name).assumeRolePolicy);
    expect(trust.Statement).toHaveLength(1);
    expect(trust.Statement[0].Principal).toEqual({ Federated: PROVIDER_ARN });
    expect(
      trust.Statement[0].Condition.StringEquals["token.actions.githubusercontent.com:sub"],
    ).toBe(sub);
  });

  it("gives each role its own document — the preview role is not the deploy role", () => {
    expect(JSON.parse(inline("deploy").policy)).toEqual(deployPolicy(names));
    expect(JSON.parse(inline("preview").policy)).toEqual(previewPolicy(names));
  });

  it("caps sessions at an hour and tags the roles", () => {
    for (const name of ["deploy", "preview"]) {
      expect(role(name).maxSessionDuration).toBe(3600);
      expect(role(name).tags).toEqual({ project: "model-playground" });
    }
  });

  it("exports the ARNs GitHub needs as repository variables", async () => {
    const value = <T>(o: pulumi.Output<T>) =>
      new Promise<T>((resolve) => o.apply(resolve));
    expect(await value(stack.deployRoleArn)).toContain("role/model-playground-deploy");
    expect(await value(stack.previewRoleArn)).toContain("role/model-playground-preview");
    expect(await value(stack.oidcProviderArn)).toBe(PROVIDER_ARN);
  });
});
