import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

// Evaluate the file's exact text, as CloudFront will: it has no exports to
// import, and adding a `module.exports` shim would ship bytes the test put
// there rather than bytes the function needs.
const source = readFileSync(join(__dirname, "spa-rewrite.js"), "utf8");
const handler = new Function(`${source}\nreturn handler;`)() as (event: {
  request: { uri: string; querystring?: unknown };
}) => { uri: string; querystring?: unknown };

const rewrite = (uri: string) => handler({ request: { uri } }).uri;

describe("spa-rewrite (CloudFront viewer-request)", () => {
  it.each(["/", "/asr", "/home", "/tasks/text-to-image", "/asr/"])(
    "serves the app shell for the route %s",
    (uri) => expect(rewrite(uri)).toBe("/index.html"),
  );

  it.each([
    "/index.html",
    "/assets/index-abc123.js",
    "/assets/ort-wasm-simd-threaded.jsep-B0T3yYHD.wasm",
    "/assets/cora-x1.bin",
    "/favicon.ico",
    "/site.webmanifest",
  ])("fetches the file %s as itself", (uri) => expect(rewrite(uri)).toBe(uri));

  it("leaves the query string alone", () => {
    const querystring = { q: { value: "1" } };
    const out = handler({ request: { uri: "/asr", querystring } });
    expect(out.querystring).toBe(querystring);
  });

  // The rule is "a dot in the last segment means a file". A route with a dot
  // in it would be fetched from the bucket, miss, and fail — so pin the
  // assumption against the router's own generated route list.
  it("no app route contains a dot, so none bypasses the rewrite", () => {
    const tree = readFileSync(
      join(__dirname, "../../frontend/src/routeTree.gen.ts"),
      "utf8",
    );
    const paths = [...tree.matchAll(/\bpath: '([^']*)'/g)].map((m) => m[1]);
    expect(paths.length).toBeGreaterThan(20);
    expect(paths.filter((p) => p.includes("."))).toEqual([]);
    for (const p of paths) expect(rewrite(p)).toBe("/index.html");
  });
});
