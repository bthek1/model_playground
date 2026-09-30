// scripts/deploy-frontend.sh against a stub `aws` on PATH. Every invocation is
// logged; the bucket's current keys and stale tags are canned. The assertions
// are the script's contract: upload order, explicit wasm type, cache headers,
// no delete, stale-marking once, and a two-path invalidation.

import { execFileSync } from "node:child_process";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";

const SCRIPT = join(__dirname, "../scripts/deploy-frontend.sh");

const STUB = `#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$1 $2" in
  "s3api list-objects-v2") printf '%s\\n' "$STUB_KEYS" ;;
  "s3api get-object-tagging")
    for k in $STUB_STALE; do [[ "$*" == *"--key $k "* ]] && { echo true; exit 0; }; done
    echo "" ;;
  "s3api head-object") printf 'text/javascript\\tpublic, max-age=31536000, immutable\\n' ;;
  "cloudfront create-invalidation") echo I123 ;;
esac
`;

let calls: string[];

beforeAll(() => {
  const dir = mkdtempSync(join(tmpdir(), "deploy-frontend-"));
  const bin = join(dir, "bin");
  const dist = join(dir, "dist");
  mkdirSync(bin);
  mkdirSync(join(dist, "assets"), { recursive: true });
  writeFileSync(join(bin, "aws"), STUB);
  chmodSync(join(bin, "aws"), 0o755);
  for (const f of ["index.html", "site.webmanifest", "favicon.ico"]) {
    writeFileSync(join(dist, f), "x");
  }
  for (const f of ["index-new.js", "ort-new.wasm"]) {
    writeFileSync(join(dist, "assets", f), "x");
  }
  const log = join(dir, "log");
  writeFileSync(log, "");

  execFileSync("bash", [SCRIPT, dist], {
    env: {
      ...process.env,
      PATH: `${bin}:${process.env.PATH}`,
      STUB_LOG: log,
      BUCKET: "site-bucket",
      DISTRIBUTION_ID: "EDIST",
      // In the bucket: this build's two, one superseded chunk not yet marked,
      // and one already marked by an earlier deploy.
      STUB_KEYS: [
        "assets/index-new.js",
        "assets/ort-new.wasm",
        "assets/index-old.js",
        "assets/index-older.js",
      ].join("\t"),
      // index-new.js was stale after an earlier deploy and this build brings it
      // back: the upload re-writes it untagged, and it must not be re-marked.
      STUB_STALE: "assets/index-older.js assets/index-new.js",
    },
    stdio: "pipe",
  });
  calls = readFileSync(log, "utf8").trim().split("\n");
});

const indexOf = (pred: (c: string) => boolean) => {
  const i = calls.findIndex(pred);
  expect(i, "call not found").toBeGreaterThanOrEqual(0);
  return i;
};

describe("deploy-frontend.sh", () => {
  it("uploads hashed assets and wasm before index.html, and index.html last", () => {
    const assets = indexOf((c) => c.includes("assets s3://site-bucket/assets") && c.includes('--exclude *.wasm'));
    const wasm = indexOf((c) => c.includes("--include *.wasm"));
    const manifest = indexOf((c) => c.includes("site.webmanifest s3://"));
    const index = indexOf((c) => c.includes("index.html s3://site-bucket/index.html"));
    expect(assets).toBeLessThan(index);
    expect(wasm).toBeLessThan(index);
    expect(manifest).toBeLessThan(index);
    const uploads = calls.filter((c) => c.startsWith("s3 cp"));
    expect(uploads.at(-1)).toContain("index.html s3://");
  });

  it("states application/wasm rather than trusting a MIME guess", () => {
    const wasm = calls[indexOf((c) => c.includes("--include *.wasm"))];
    expect(wasm).toContain("--content-type application/wasm");
    expect(wasm).toContain("immutable");
  });

  it("caches hashed assets forever and index.html not at all", () => {
    expect(calls[indexOf((c) => c.includes("--exclude *.wasm"))]).toContain(
      "public, max-age=31536000, immutable",
    );
    const index = calls[indexOf((c) => c.includes("index.html s3://site-bucket/index.html"))];
    expect(index).toContain("no-cache, must-revalidate");
    expect(index).toContain("text/html");
  });

  it("never deletes anything", () => {
    for (const c of calls) {
      expect(c).not.toMatch(/--delete|s3 rm|delete-object/);
    }
  });

  it("marks only the newly superseded asset stale, and only once", () => {
    const copies = calls.filter((c) => c.startsWith("s3api copy-object"));
    expect(copies).toHaveLength(1);
    expect(copies[0]).toContain("--key assets/index-old.js");
    expect(copies[0]).toContain("--tagging stale=true");
    // Metadata survives the in-place rewrite.
    expect(copies[0]).toContain("--content-type text/javascript");
    expect(copies[0]).toContain("--cache-control public, max-age=31536000, immutable");
  });

  it("never inspects or rewrites an asset this build uses, even one stale before", () => {
    const touched = calls.filter(
      (c) =>
        /^s3api (get-object-tagging|head-object|copy-object)/.test(c) &&
        /--key assets\/(index-new\.js|ort-new\.wasm) /.test(c),
    );
    expect(touched).toEqual([]);
    // It is back as a fresh, untagged upload: a plain `s3 cp` of assets/.
    expect(calls.some((c) => c.startsWith("s3 cp") && c.includes("s3://site-bucket/assets"))).toBe(true);
  });

  it("marks stale only after the new index.html is live", () => {
    const index = indexOf((c) => c.includes("index.html s3://site-bucket/index.html"));
    const copy = indexOf((c) => c.startsWith("s3api copy-object"));
    expect(copy).toBeGreaterThan(index);
  });

  it("invalidates only the two un-hashed switch-over files, last", () => {
    const inv = calls.filter((c) => c.startsWith("cloudfront create-invalidation"));
    expect(inv).toHaveLength(1);
    expect(inv[0]).toContain("--distribution-id EDIST");
    expect(inv[0]).toContain("--paths /index.html /site.webmanifest");
    expect(calls.at(-1)).toBe(inv[0]);
  });
});
