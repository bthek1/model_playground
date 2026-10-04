#!/usr/bin/env node
// One-time PostHog project setup for the playground's analytics (#60).
//
//     just posthog-setup            (or: node scripts/posthog-setup.mjs)
//
// Uses the **personal** API key (`phx_…`, `POSTHOG_ALL_ACCESS` in the root
// .env or the environment) to find or create the `model-playground` project,
// set its privacy options, and print the one thing the app needs: the public,
// write-only **project** key (`phc_…`) and the ingestion host.
//
// The personal key can read and delete every project in the organisation, so:
//   * it is used from this script only, on your machine — never CI, never a
//     VITE_ name (check:bundle fails the build if a `phx_` reaches dist/);
//   * it is never printed, and neither is any response body, which could echo it;
//   * if it ever has to live anywhere else, rotate it and scope a new one to
//     `user:read`, `organization:read`, `project:read` and `project:write`.
//
// Idempotent: settings already as wanted are left alone, and a second run says
// "no changes". Phase 0 of #60 is its first step: it reports which region the
// key belongs to (US or EU) by asking each cloud who the key is.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const PROJECT_NAME = "model-playground";

/**
 * The project settings this app's privacy story depends on. The client already
 * asks for none of these features (src/analytics/client.ts); turning them off
 * server-side as well means a click in the PostHog UI cannot switch them back on.
 */
export const WANTED = {
  anonymize_ips: true,
  session_recording_opt_in: false,
  heatmaps_opt_in: false,
  surveys_opt_in: false,
  autocapture_opt_out: true,
  autocapture_exceptions_opt_in: false,
  capture_console_log_opt_in: false,
  capture_performance_opt_in: false,
};

export const REGIONS = [
  { name: "US", api: "https://us.posthog.com", ingest: "us.i.posthog.com" },
  { name: "EU", api: "https://eu.posthog.com", ingest: "eu.i.posthog.com" },
];

/** Read one key from a dotenv file without exporting the rest. */
export function readDotenvKey(text, key) {
  for (const line of text.split(/\r?\n/)) {
    const m = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/.exec(line);
    if (m && m[1] === key) return m[2].replace(/^(['"])(.*)\1$/, "$2");
  }
  return undefined;
}

/** Fields of `current` that differ from `WANTED`. */
export function diffSettings(current) {
  const patch = {};
  for (const [k, v] of Object.entries(WANTED)) if (current[k] !== v) patch[k] = v;
  return patch;
}

export async function run({ env, fetch, log }) {
  const key = env.POSTHOG_ALL_ACCESS;
  if (!key) throw new Error("POSTHOG_ALL_ACCESS is not set (root .env or environment).");
  if (key.startsWith("phc_")) {
    throw new Error("POSTHOG_ALL_ACCESS is a *project* key (phc_). This script needs a personal API key (phx_).");
  }
  if (!key.startsWith("phx_")) log("Note: the key does not start with phx_; trying it as a personal API key anyway.");

  const call = async (base, path, init = {}) => {
    const res = await fetch(`${base}${path}`, {
      ...init,
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", ...init.headers },
    });
    // Status only — a body could echo the request, and so the key.
    if (!res.ok) throw new Error(`${init.method ?? "GET"} ${path} → HTTP ${res.status}`);
    return res.json();
  };

  // Phase 0: whose key, which region.
  let region;
  for (const r of REGIONS) {
    try {
      await call(r.api, "/api/users/@me/");
      region = r;
      break;
    } catch {
      // Not this cloud; try the next.
    }
  }
  if (!region) throw new Error("Neither PostHog US nor EU accepted the key as a personal API key.");
  log(`Personal API key accepted by PostHog ${region.name} (${region.api}).`);

  const projects = await call(region.api, "/api/organizations/@current/projects/");
  let project = (projects.results ?? projects).find((p) => p.name === PROJECT_NAME);
  if (project) {
    log(`Found project "${PROJECT_NAME}" (id ${project.id}).`);
  } else {
    project = await call(region.api, "/api/organizations/@current/projects/", {
      method: "POST",
      body: JSON.stringify({ name: PROJECT_NAME }),
    });
    log(`Created project "${PROJECT_NAME}" (id ${project.id}).`);
  }

  const current = await call(region.api, `/api/projects/${project.id}/`);
  const patch = diffSettings(current);
  if (Object.keys(patch).length === 0) {
    log("Privacy settings already as wanted — no changes.");
  } else {
    await call(region.api, `/api/projects/${project.id}/`, { method: "PATCH", body: JSON.stringify(patch) });
    log(`Updated: ${Object.keys(patch).join(", ")}.`);
  }

  const token = current.api_token;
  if (typeof token !== "string" || !token.startsWith("phc_")) {
    throw new Error("The project has no phc_ api_token in its settings; check it in the PostHog UI.");
  }
  log("");
  log("Set these on GitHub (Settings → Secrets and variables → Actions → *Variables*, not Secrets):");
  log(`  VITE_POSTHOG_KEY=${token}`);
  if (region.name !== "US") {
    log("");
    log(`EU project: point the CloudFront /ingest behaviour at it before deploying:`);
    log(`  cd infra/site && pulumi config set site:posthogHost ${region.ingest} --stack production`);
  }
  return { region: region.name, projectId: project.id, token, changed: Object.keys(patch) };
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const root = join(dirname(fileURLToPath(import.meta.url)), "..");
  let env = { ...process.env };
  if (!env.POSTHOG_ALL_ACCESS) {
    try {
      const value = readDotenvKey(readFileSync(join(root, ".env"), "utf8"), "POSTHOG_ALL_ACCESS");
      if (value) env = { ...env, POSTHOG_ALL_ACCESS: value };
    } catch {
      // No root .env — the error below says what is missing.
    }
  }
  run({ env, fetch: globalThis.fetch, log: (m) => console.log(m) }).catch((e) => {
    console.error(`posthog-setup: ${e.message}`);
    process.exit(1);
  });
}
