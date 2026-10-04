// scripts/posthog-setup.mjs against a fake PostHog API. No network, no real key.
// The contract: find-or-create, patch only what differs (so a second run is a
// no-op), never print the personal key or a response body, and refuse a
// project key handed to it by mistake.

import { describe, expect, it } from "vitest";

// @ts-expect-error — a plain .mjs script, no declarations.
import * as setup from "../scripts/posthog-setup.mjs";

const PERSONAL = "phx_personal_never_print_me_0123456789";
const PROJECT_KEY = "phc_public_project_key";

interface Call {
  url: string;
  method: string;
  body?: unknown;
  auth?: string;
}

/** A fake PostHog: `region` is the cloud that accepts the key. */
function fakePostHog({
  region = "https://us.posthog.com",
  existing = true,
  settings = {} as Record<string, unknown>,
} = {}) {
  const calls: Call[] = [];
  let project = existing ? { id: 7, name: "model-playground" } : null;
  let state: Record<string, unknown> = { api_token: PROJECT_KEY, ...settings };
  const fetch = async (url: string, init: RequestInit = {}) => {
    const method = init.method ?? "GET";
    const body = init.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ url, method, body, auth: (init.headers as Record<string, string>).Authorization });
    const json = (status: number, data: unknown) =>
      ({ ok: status < 400, status, json: async () => data }) as Response;
    if (!url.startsWith(region)) return json(401, { detail: `bad key ${PERSONAL}` });
    const path = url.slice(region.length);
    if (path === "/api/users/@me/") return json(200, { email: "me@example.com" });
    if (path === "/api/organizations/@current/projects/" && method === "GET")
      return json(200, { results: project ? [project, { id: 1, name: "other" }] : [{ id: 1, name: "other" }] });
    if (path === "/api/organizations/@current/projects/" && method === "POST") {
      project = { id: 9, name: body.name };
      return json(201, project);
    }
    if (path === `/api/projects/${project?.id}/` && method === "GET") return json(200, state);
    if (path === `/api/projects/${project?.id}/` && method === "PATCH") {
      state = { ...state, ...body };
      return json(200, state);
    }
    return json(404, {});
  };
  return { fetch, calls, state: () => state };
}

async function runWith(fake: ReturnType<typeof fakePostHog>, key = PERSONAL) {
  const lines: string[] = [];
  const result = await setup.run({
    env: { POSTHOG_ALL_ACCESS: key },
    fetch: fake.fetch,
    log: (m: string) => lines.push(m),
  });
  return { result, out: lines.join("\n") };
}

describe("posthog-setup (#60)", () => {
  it("creates the project when missing and applies every privacy setting", async () => {
    const fake = fakePostHog({ existing: false });
    const { result, out } = await runWith(fake);
    expect(fake.calls.some((c) => c.method === "POST" && (c.body as { name: string }).name === "model-playground")).toBe(true);
    expect(fake.state()).toMatchObject(setup.WANTED);
    expect(result.changed.sort()).toEqual(Object.keys(setup.WANTED).sort());
    expect(out).toContain(`VITE_POSTHOG_KEY=${PROJECT_KEY}`);
  });

  it("is idempotent: a second run changes nothing and says so", async () => {
    const fake = fakePostHog();
    await runWith(fake);
    fake.calls.length = 0;
    const { result, out } = await runWith(fake);
    expect(result.changed).toEqual([]);
    expect(out).toMatch(/no changes/);
    expect(fake.calls.filter((c) => c.method !== "GET")).toEqual([]);
  });

  it("patches only the settings that differ", async () => {
    const fake = fakePostHog({ settings: { ...setup.WANTED, anonymize_ips: false } });
    const { result } = await runWith(fake);
    expect(result.changed).toEqual(["anonymize_ips"]);
    expect(fake.calls.find((c) => c.method === "PATCH")?.body).toEqual({ anonymize_ips: true });
  });

  it("finds the region — an EU key is told to re-point /ingest", async () => {
    const fake = fakePostHog({ region: "https://eu.posthog.com" });
    const { result, out } = await runWith(fake);
    expect(result.region).toBe("EU");
    expect(out).toContain("site:posthogHost eu.i.posthog.com");
  });

  it("never prints the personal key, even when the API echoes it", async () => {
    const fake = fakePostHog({ region: "https://eu.posthog.com" }); // US answers 401 with the key in the body
    const { out } = await runWith(fake);
    expect(out).not.toContain(PERSONAL);
    expect(out).not.toMatch(/phx_/);
    await expect(runWith(fakePostHog({ region: "https://nowhere" }))).rejects.toThrow(
      /Neither PostHog US nor EU/,
    );
  });

  it("sends the personal key only as a bearer token, to PostHog's own API hosts", async () => {
    const fake = fakePostHog();
    await runWith(fake);
    for (const c of fake.calls) {
      expect(c.url).toMatch(/^https:\/\/(us|eu)\.posthog\.com\/api\//);
      expect(c.auth).toBe(`Bearer ${PERSONAL}`);
      expect(c.url).not.toContain(PERSONAL);
    }
  });

  it("refuses a project key handed to it by mistake", async () => {
    await expect(runWith(fakePostHog(), PROJECT_KEY)).rejects.toThrow(/project\* key/);
  });

  it("reads one key from a dotenv file", () => {
    const text = `SECRET_KEY=x\nPOSTHOG_ALL_ACCESS="phx_abc"\nVITE_BACKEND=on\n`;
    expect(setup.readDotenvKey(text, "POSTHOG_ALL_ACCESS")).toBe("phx_abc");
    expect(setup.readDotenvKey(text, "MISSING")).toBeUndefined();
  });
});
