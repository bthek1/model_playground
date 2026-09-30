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
  */asr) printf '%s' "\${STUB_ASR_BODY:-<!doctype html><div id=\\"root\\"></div>}" ;;
  */)
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
    expect(out).toContain("✓ / is 200");
    expect(out).toContain("✓ / is no-cache");
    expect(out).toContain("✓ /assets/index-abc.js is immutable");
    expect(out).toContain("✓ /assets/ort-abc.wasm is application/wasm");
    expect(out).toContain("✓ /asr (a deep link) returns the app shell");
    expect(out).toContain("✓ http:// redirects to https://");
    expect(out).toContain("✓ the bucket refuses direct requests (403)");
    expect(code).toBe(0);
  });

  it.each([
    ["/ never answers 200", { STUB_ROOT_CODE: "503" }, "/ is 503, not 200"],
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

  it("retries a / that is still propagating, then gives up", () => {
    const { out } = smoke({ STUB_ROOT_CODE: "404" });
    expect(out).toContain("waiting (1/2)");
    expect(out).toContain("waiting (2/2)");
  });
});
