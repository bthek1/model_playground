import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";

// @ts-expect-error -- a plain .mjs build script, typed by its JSDoc
import * as notices from "../../scripts/third-party-notices.mjs";

const { ALLOWED, declaredLicence, isAllowed, packageRoot, readPackage, renderNotices, thirdPartyNotices } =
  notices;

const tmp = mkdtempSync(join(tmpdir(), "notices-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

/** A fake package on disk, as `node_modules/<name>` would hold it. */
function fakePackage(name: string, pkg: Record<string, unknown>, files: Record<string, string> = {}) {
  const root = join(tmp, "node_modules", name);
  mkdirSync(root, { recursive: true });
  writeFileSync(join(root, "package.json"), JSON.stringify({ name, version: "1.0.0", ...pkg }));
  for (const [f, text] of Object.entries(files)) writeFileSync(join(root, f), text);
  return root;
}

describe("packageRoot", () => {
  it("finds the package a module id belongs to, scoped or not", () => {
    expect(packageRoot("/w/node_modules/react/cjs/react.js")).toBe("/w/node_modules/react");
    expect(packageRoot("/w/node_modules/@base-ui/react/esm/index.js")).toBe(
      "/w/node_modules/@base-ui/react",
    );
  });

  it("takes the innermost copy of a nested dependency", () => {
    // Two versions of one package ship different notices; the outer one is
    // not the code that was bundled.
    expect(packageRoot("/w/node_modules/kokoro-js/node_modules/@huggingface/transformers/dist/x.js")).toBe(
      "/w/node_modules/kokoro-js/node_modules/@huggingface/transformers",
    );
  });

  it("is null for app code and strips Rollup's virtual-module markers", () => {
    expect(packageRoot("/w/src/main.tsx")).toBeNull();
    expect(packageRoot("\0/w/node_modules/react/index.js?commonjs-proxy")).toBe("/w/node_modules/react");
  });
});

describe("licence checks", () => {
  it("reads every shape package.json has used for a licence", () => {
    expect(declaredLicence({ license: "MIT" })).toBe("MIT");
    expect(declaredLicence({ license: { type: "ISC" } })).toBe("ISC");
    expect(declaredLicence({ licenses: [{ type: "MIT" }, { type: "Apache-2.0" }] })).toBe("MIT OR Apache-2.0");
    expect(declaredLicence({})).toBe("UNKNOWN");
  });

  it("accepts an OR when one side is allowed, an AND only when both are", () => {
    expect(isAllowed("(MIT OR GPL-3.0)")).toBe(true);
    expect(isAllowed("(Apache-2.0 AND MIT)")).toBe(true);
    expect(isAllowed("MIT AND GPL-3.0")).toBe(false);
  });

  it("refuses copyleft and unknown licences", () => {
    for (const l of ["GPL-3.0", "AGPL-3.0", "LGPL-2.1", "SSPL-1.0", "UNKNOWN", "UNLICENSED"]) {
      expect(isAllowed(l), l).toBe(false);
    }
    expect(ALLOWED.has("MIT")).toBe(true);
  });
});

describe("renderNotices", () => {
  it("includes each package's licence file, verbatim", () => {
    const root = fakePackage("left-pad", { license: "MIT" }, { LICENSE: "Copyright (c) Someone\n\nPermission is hereby granted" });
    const text = renderNotices([readPackage(root)]);
    expect(text).toContain("left-pad@1.0.0");
    expect(text).toContain("License: MIT");
    expect(text).toContain("Copyright (c) Someone");
  });

  it("includes an Apache NOTICE file as well as the licence", () => {
    // Apache-2.0 §4(d): a NOTICE file must be carried along with the licence.
    const root = fakePackage("apache-thing", { license: "Apache-2.0" }, { LICENSE: "Apache text", NOTICE: "Apache Thing\nCopyright Corp" });
    expect(renderNotices([readPackage(root)])).toContain("Copyright Corp");
  });

  it("says so when a package ships no licence file", () => {
    const root = fakePackage("bare", { license: "ISC" });
    expect(renderNotices([readPackage(root)])).toContain("No licence file is included");
  });
});

describe("the plugin", () => {
  function run(plugin: ReturnType<typeof thirdPartyNotices>, ids: string[]) {
    const emitted: { fileName: string; source: string }[] = [];
    const errors: string[] = [];
    const ctx = {
      emitFile: (f: { fileName: string; source: string }) => emitted.push(f),
      error: (m: string) => {
        errors.push(m);
        throw new Error(m);
      },
    };
    const bundle = {
      "a.js": { type: "chunk", modules: Object.fromEntries(ids.map((id) => [id, { renderedLength: 10 }])) },
    };
    try {
      plugin.generateBundle.call(ctx, {}, bundle);
    } catch {
      /* the error is in `errors` */
    }
    return { emitted, errors };
  }

  it("fails the build, naming the package, on a licence off the allowlist", () => {
    const root = fakePackage("copyleft-lib", { license: "GPL-3.0" });
    const { errors, emitted } = run(thirdPartyNotices(), [join(root, "index.js")]);
    expect(errors[0]).toMatch(/copyleft-lib@1\.0\.0 — GPL-3\.0/);
    expect(emitted).toEqual([]);
  });

  it("counts what a worker build recorded, and only the main build writes", () => {
    // The plugin's `seen` set is module-level by design (that is how worker
    // builds reach the main one), so the GPL package from the test above is
    // still in it — hence GPL on this test's allowlist.
    const fromWorker = fakePackage("worker-only", { license: "MIT" });
    const fromMain = fakePackage("main-only", { license: "MIT" });
    const worker = run(thirdPartyNotices({ emit: false, allowed: new Set(["MIT", "GPL-3.0"]) }), [
      join(fromWorker, "x.js"),
    ]);
    expect(worker.emitted).toEqual([]);
    const main = run(thirdPartyNotices({ allowed: new Set(["MIT", "GPL-3.0"]) }), [join(fromMain, "x.js")]);
    expect(main.emitted[0].fileName).toBe("THIRD-PARTY-NOTICES.txt");
    expect(main.emitted[0].source).toContain("worker-only@1.0.0");
    expect(main.emitted[0].source).toContain("main-only@1.0.0");
  });
});
