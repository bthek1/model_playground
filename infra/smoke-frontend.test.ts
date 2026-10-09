// scripts/smoke-frontend.sh against a stub `curl` on PATH. The healthy case
// must pass; then each property the plan names is broken one at a time and
// the script must fail *naming it* — a smoke test that passes a broken site,
// or fails without saying why, is worse than none.

import { spawnSync } from "node:child_process";
import { chmodSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";

const SCRIPT = join(__dirname, "../scripts/smoke-frontend.sh");
const HOST = "playground.example.com";

// Answers by URL; each property has an env override so a test can break it.
const STUB = `#!/usr/bin/env bash
url=""; fmt=""; headers=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -w) fmt="$2"; shift 2 ;;
    -D) headers=1; shift 2 ;;
    -o) shift 2 ;;
    http*) url="$1"; shift ;;
    *) shift ;;
  esac
done
case "$url" in
  *amazonaws.com*) printf '%s' "\${STUB_BUCKET_CODE:-403}" ;;
  http://*) printf '%s' "\${STUB_HTTP:-301 https://${HOST}/asr}" ;;
  */asr) printf '%s' "\${STUB_ASR_BODY:-<!doctype html><link rel=\\"canonical\\" href=\\"https://${HOST}/asr\\" /><div id=\\"root\\"></div>}" ;;
  */ASR/) printf '%s' "\${STUB_UPPER:-301 https://${HOST}/asr}" ;;
  */robots.txt)
    if [[ -n $fmt ]]; then printf '%s' "\${STUB_ROBOTS_CODE:-200}"
    else printf '%s\n' "User-agent: *" "\${STUB_ROBOTS_SITEMAP-Sitemap: https://${HOST}/sitemap.xml}"; fi ;;
  */sitemap.xml)
    if [[ -n $fmt ]]; then printf '%s' "\${STUB_SITEMAP_CODE:-200}"
    else printf '%s\n' "<urlset>" "\${STUB_SITEMAP_LOC-<loc>https://${HOST}/asr</loc>}" "</urlset>"; fi ;;
  */) printf '%s' "\${STUB_ROOT:-301 https://${HOST}/home}" ;;
  */home)
    if [[ $headers == 1 ]]; then
      printf 'HTTP/2 200\\r\\nCache-Control: %s\\r\\n\\r\\n' "\${STUB_INDEX_CACHE:-no-cache, must-revalidate}"
    else
      printf '%s' "\${STUB_ROOT_CODE:-200}"
    fi ;;
  *.js) printf 'HTTP/2 200\\r\\nCache-Control: %s\\r\\n\\r\\n' "\${STUB_JS_CACHE:-public, max-age=31536000, immutable}" ;;
  *.wasm) printf 'HTTP/2 200\\r\\nContent-Type: %s\\r\\n\\r\\n' "\${STUB_WASM_TYPE:-application/wasm}" ;;
esac
`;

let bin: string;
let dist: string;

beforeAll(() => {
  const dir = mkdtempSync(join(tmpdir(), "smoke-frontend-"));
  bin = join(dir, "bin");
  dist = join(dir, "dist");
  mkdirSync(bin);
  mkdirSync(join(dist, "assets"), { recursive: true });
  writeFileSync(join(bin, "curl"), STUB);
  chmodSync(join(bin, "curl"), 0o755);
  writeFileSync(join(dist, "assets", "index-abc.js"), "x");
  writeFileSync(join(dist, "assets", "ort-abc.wasm"), "x");
});

function smoke(env: Record<string, string> = {}) {
  const r = spawnSync("bash", [SCRIPT, `https://${HOST}`, dist], {
    env: {
      ...process.env,
      PATH: `${bin}:${process.env.PATH}`,
      BUCKET: "site-bucket",
      SMOKE_ATTEMPTS: "2",
      SMOKE_WAIT: "0",
      ...env,
    },
    encoding: "utf8",
  });
  return { code: r.status, out: r.stdout + r.stderr };
}

describe("smoke-frontend.sh", () => {
  it("passes a healthy site, checking every property the plan names", () => {
    const { code, out } = smoke();
    expect(out).toContain("✓ /home is 200");
    expect(out).toContain("✓ /home is no-cache");
    expect(out).toContain("✓ / redirects to /home");
    expect(out).toContain("✓ /ASR/ redirects to /asr");
    expect(out).toContain("✓ /asr is its own page, with its canonical");
    expect(out).toContain("✓ /robots.txt and /sitemap.xml are served");
    expect(out).toContain("✓ /assets/index-abc.js is immutable");
    expect(out).toContain("✓ /assets/ort-abc.wasm is application/wasm");
    expect(out).toContain("✓ /asr (a deep link) returns the app shell");
    expect(out).toContain("✓ http:// redirects to https://");
    expect(out).toContain("✓ the bucket refuses direct requests (403)");
    expect(code).toBe(0);
  });

  it.each([
    ["/home never answers 200", { STUB_ROOT_CODE: "503" }, "/home is 503, not 200"],
    ["/ serves the shell", { STUB_ROOT: "200 " }, "want a 301 to /home"],
    ["a non-canonical URL is served", { STUB_UPPER: "200 " }, "want a 301 to /asr"],
    ["a route gets the generic shell", { STUB_ASR_BODY: '<div id="root"></div>' }, "the generic shell, not its own page"],
    ["robots.txt is missing", { STUB_ROBOTS_CODE: "403" }, "/robots.txt is 403"],
    ["the sitemap is missing", { STUB_SITEMAP_CODE: "403" }, "/sitemap.xml is 403"],
    ["robots.txt names no sitemap", { STUB_ROBOTS_SITEMAP: "" }, "does not name"],
    ["the sitemap misses a route", { STUB_SITEMAP_LOC: "" }, "does not list"],
    ["index.html is cacheable", { STUB_INDEX_CACHE: "public, max-age=3600" }, "want no-cache"],
    ["assets are not immutable", { STUB_JS_CACHE: "max-age=60" }, "want immutable"],
    ["wasm has a guessed type", { STUB_WASM_TYPE: "application/octet-stream" }, "content-type is 'application/octet-stream'"],
    ["a deep link 404s", { STUB_ASR_BODY: "<Error>NoSuchKey</Error>" }, "/asr did not return the app shell"],
    ["plain HTTP is served", { STUB_HTTP: "200 " }, "want a redirect to https"],
    ["the bucket is public", { STUB_BUCKET_CODE: "200" }, "want 403"],
  ])("fails when %s", (_, env, message) => {
    const { code, out } = smoke(env);
    expect(code).not.toBe(0);
    expect(out).toContain(message);
  });

  it("retries a /home that is still propagating, then gives up", () => {
    const { out } = smoke({ STUB_ROOT_CODE: "404" });
    expect(out).toContain("waiting (1/2)");
    expect(out).toContain("waiting (2/2)");
  });
});
