// The CD half of .github/workflows/ci.yml, read as data. Nothing runs a
// workflow file before it is pushed, and each property below fails silently
// when edited away: a deploy that rebuilds ships untested bytes, a cancelled
// release half-uploads, a preview job with an `environment:` changes its OIDC
// subject and can no longer assume its role.

import { load } from "js-yaml";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

type Step = { uses?: string; run?: string; with?: Record<string, string>; env?: Record<string, string>; name?: string };
type Job = {
  needs?: string | string[];
  if?: string;
  environment?: string | { name: string; url?: string };
  permissions?: Record<string, string>;
  concurrency?: { group: string; "cancel-in-progress": boolean | string };
  env?: Record<string, string>;
  steps: Step[];
};
type Workflow = {
  concurrency: { group: string; "cancel-in-progress": boolean | string };
  jobs: Record<string, Job>;
};

const wf = load(
  readFileSync(join(__dirname, "../.github/workflows/ci.yml"), "utf8"),
) as Workflow;
const job = (name: string) => {
  expect(wf.jobs[name], name).toBeDefined();
  return wf.jobs[name];
};
const runs = (j: Job) => j.steps.map((s) => s.run ?? "").join("\n");
const uses = (j: Job) => j.steps.map((s) => s.uses ?? "");
const list = (x?: string | string[]) => (x === undefined ? [] : Array.isArray(x) ? x : [x]);

describe("the shipped artifact", () => {
  it("is built with VITE_BACKEND=off by the frontend job", () => {
    const build = job("frontend").steps.find((s) => s.run === "npm run build");
    expect(build?.env?.VITE_BACKEND).toBe("off");
    expect(uses(job("frontend"))).toContain("actions/upload-artifact@v4");
  });

  it("is what e2e-static tests, served as a build (E2E_STATIC), not rebuilt", () => {
    const j = job("e2e-static");
    expect(list(j.needs)).toContain("frontend");
    expect(j.steps.find((s) => s.uses === "actions/download-artifact@v4")?.with?.name).toBe("dist");
    expect(j.steps.find((s) => s.run?.includes("playwright test"))?.env?.E2E_STATIC).toBe("1");
    expect(runs(j)).not.toMatch(/npm run build|vite build/);
  });

  it("is what deploy uploads — deploy never builds", () => {
    const j = job("deploy");
    expect(j.steps.find((s) => s.uses === "actions/download-artifact@v4")?.with?.name).toBe("dist");
    expect(runs(j)).not.toMatch(/npm run build|vite build/);
  });
});

describe("the deploy job", () => {
  const j = () => job("deploy");

  it("waits for every test job, including the one that tested its bytes", () => {
    expect(list(j().needs).sort()).toEqual(["e2e", "e2e-static", "frontend", "infra"]);
  });

  it("runs only on a push to main", () => {
    expect(j().if).toContain("github.event_name == 'push'");
    expect(j().if).toContain("github.ref == 'refs/heads/main'");
  });

  it("runs in the production environment the deploy role's trust names", () => {
    const env = j().environment;
    expect(typeof env === "string" ? env : env?.name).toBe("production");
  });

  it("gets AWS through OIDC — an id-token, a role, and no keys", () => {
    expect(j().permissions).toEqual({ "id-token": "write", contents: "read" });
    const creds = j().steps.find((s) => s.uses?.startsWith("aws-actions/configure-aws-credentials"));
    expect(creds?.with?.["role-to-assume"]).toBe("${{ vars.AWS_DEPLOY_ROLE_ARN }}");
    expect(JSON.stringify(wf)).not.toMatch(/aws-access-key-id|AWS_SECRET_ACCESS_KEY/i);
  });

  it("is never cancelled mid-upload", () => {
    expect(j().concurrency).toEqual({
      group: "deploy-production",
      "cancel-in-progress": false,
    });
    // The workflow-level group would otherwise cancel the whole run on main.
    expect(wf.concurrency["cancel-in-progress"]).toBe(
      "${{ github.ref != 'refs/heads/main' }}",
    );
  });

  it("runs pulumi up, then the upload, then the smoke test — in that order", () => {
    const script = runs(j());
    const up = script.indexOf("pulumi up");
    const upload = script.indexOf("scripts/deploy-frontend.sh");
    const smoke = script.indexOf("scripts/smoke-frontend.sh");
    expect(up).toBeGreaterThanOrEqual(0);
    expect(upload).toBeGreaterThan(up);
    expect(smoke).toBeGreaterThan(upload);
  });
});

describe("the preview job", () => {
  const j = () => job("preview");

  it("declares no environment — that would change the OIDC subject", () => {
    expect(j().environment).toBeUndefined();
  });

  it("runs only on a push to develop, with the read-only role", () => {
    expect(j().if).toContain("github.ref == 'refs/heads/develop'");
    const creds = j().steps.find((s) => s.uses?.startsWith("aws-actions/configure-aws-credentials"));
    expect(creds?.with?.["role-to-assume"]).toBe("${{ vars.AWS_PREVIEW_ROLE_ARN }}");
  });

  it("previews and never applies", () => {
    expect(runs(j())).toContain("pulumi preview");
    expect(runs(j())).not.toMatch(/pulumi (up|destroy)|deploy-frontend/);
  });
});

describe("no other job can touch AWS", () => {
  it("only deploy and preview request an OIDC token", () => {
    const withToken = Object.entries(wf.jobs)
      .filter(([, j]) => j.permissions?.["id-token"] === "write")
      .map(([name]) => name)
      .sort();
    expect(withToken).toEqual(["deploy", "preview"]);
  });
});
