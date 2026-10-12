import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { SITE_PAGES } from "./pages";
import { addModulePreloads, type BundleLike, routeChunk, routePreloads } from "./preload";

const R = "/repo/frontend/src/routes";

/** A bundle shaped like the real one: an entry, shared chunks, split routes. */
const BUNDLE: BundleLike = {
  "assets/index-a.js": {
    type: "chunk",
    fileName: "assets/index-a.js",
    isEntry: true,
    facadeModuleId: "/repo/frontend/index.html",
    imports: ["assets/jsx-runtime-b.js"],
  },
  "assets/jsx-runtime-b.js": { type: "chunk", fileName: "assets/jsx-runtime-b.js", imports: [] },
  "assets/ModelPage-c.js": {
    type: "chunk",
    fileName: "assets/ModelPage-c.js",
    imports: ["assets/jsx-runtime-b.js"],
  },
  "assets/asr-d.js": {
    type: "chunk",
    fileName: "assets/asr-d.js",
    facadeModuleId: `${R}/asr.tsx?tsr-split=component`,
    imports: ["assets/ModelPage-c.js", "assets/jsx-runtime-b.js"],
  },
  "assets/privacy.lazy-e.js": {
    type: "chunk",
    fileName: "assets/privacy.lazy-e.js",
    facadeModuleId: `${R}/privacy.lazy.tsx`,
    imports: [],
  },
  // A module whose name merely starts with a route's: must not match /asr.
  "assets/asr-extra-f.js": {
    type: "chunk",
    fileName: "assets/asr-extra-f.js",
    facadeModuleId: `${R}/asr-extra.tsx?tsr-split=component`,
    imports: [],
  },
  "assets/index-g.css": { type: "asset", fileName: "assets/index-g.css" },
};

describe("routeChunk", () => {
  it("finds the plugin's split chunk for a route file", () => {
    expect(routeChunk(BUNDLE, "asr")?.fileName).toBe("assets/asr-d.js");
  });

  it("finds a hand-written .lazy.tsx route", () => {
    expect(routeChunk(BUNDLE, "privacy")?.fileName).toBe("assets/privacy.lazy-e.js");
  });

  it("matches the whole file name, not a prefix of it", () => {
    expect(routeChunk(BUNDLE, "asr-extra")?.fileName).toBe("assets/asr-extra-f.js");
    expect(routeChunk(BUNDLE, "as")).toBeUndefined();
  });
});

describe("routePreloads", () => {
  it("is the route chunk and its imports, dependencies first, minus the entry's", () => {
    expect(routePreloads(BUNDLE, "/asr")).toEqual(["assets/ModelPage-c.js", "assets/asr-d.js"]);
  });

  it("fails the build for a page whose route did not split", () => {
    expect(() => routePreloads(BUNDLE, "/rl")).toThrow(/no split chunk for \/rl/);
  });
});

describe("addModulePreloads", () => {
  const HTML = `<head>\n    <link rel="modulepreload" crossorigin href="/assets/jsx-runtime-b.js">\n  </head>`;

  it("adds one link per file under the base, before </head>", () => {
    const out = addModulePreloads(HTML, ["assets/asr-d.js"], "/");
    expect(out).toContain('<link rel="modulepreload" crossorigin href="/assets/asr-d.js">');
    expect(out.indexOf("asr-d.js")).toBeLessThan(out.indexOf("</head>"));
  });

  it("skips a file the head already preloads", () => {
    const out = addModulePreloads(HTML, ["assets/jsx-runtime-b.js"], "/");
    expect(out.match(/jsx-runtime-b\.js/g)).toHaveLength(1);
  });
});

describe("every indexable page", () => {
  // `routePreloads` maps a page to `src/routes/<path>.tsx`; a page whose file
  // is named differently would fail the build, so catch it here first.
  const files = new Set(readdirSync(join(__dirname, "../routes")));
  it.each(SITE_PAGES.map((p) => p.path))("%s has a route file of the same name", (path) => {
    const stem = path.slice(1);
    expect(files.has(`${stem}.tsx`) || files.has(`${stem}.lazy.tsx`)).toBe(true);
  });

  it("is a route module nothing outside src/routes imports", () => {
    // The sidebar, the taxonomy and the head code import data only: one static
    // import of a route module from code the entry reaches would pull that
    // route's component back into the entry (#66). Tests are exempt — they
    // import the route under test by design.
    const src = join(__dirname, "..");
    const offenders = (readdirSync(src, { recursive: true }) as string[])
      .filter((f) => /\.tsx?$/.test(f) && !/\.test\.tsx?$/.test(f))
      .filter((f) => !f.startsWith("routes/") && f !== "routeTree.gen.ts")
      .filter((f) =>
        /from\s+["'](@\/routes\/|(\.\.?\/)+routes\/)/.test(readFileSync(join(src, f), "utf8")),
      );
    expect(offenders).toEqual([]);
  });
});
