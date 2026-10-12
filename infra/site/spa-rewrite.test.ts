import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

// Evaluate the file's exact text, as CloudFront will: it has no exports to
// import, and adding a `module.exports` shim would ship bytes the test put
// there rather than bytes the function needs.
const source = readFileSync(join(__dirname, "spa-rewrite.js"), "utf8");
type Querystring = Record<string, { value: string; multiValue?: { value: string }[] }>;
type Request = {
  uri: string;
  querystring?: Querystring;
  headers?: Record<string, { value: string }>;
};
type Redirect = { statusCode: number; headers: { location: { value: string } } };
const handler = new Function(`${source}\nreturn handler;`)() as (event: {
  request: Request;
}) => Request | Redirect;

const HOST = "playground.example.com";
const run = (uri: string, querystring: Querystring = {}) =>
  handler({ request: { uri, querystring, headers: { host: { value: HOST } } } });
const rewrite = (uri: string) => (run(uri) as Request).uri;
const location = (uri: string, querystring?: Querystring) => {
  const out = run(uri, querystring) as Redirect;
  expect(out.statusCode).toBe(301);
  return out.headers.location.value;
};

describe("spa-rewrite (CloudFront viewer-request)", () => {
  // An indexable page has its own file, written by the build (#64).
  it.each(["/asr", "/home", "/privacy", "/super-resolution"])(
    "serves the route %s its own index.html",
    (uri) => expect(rewrite(uri)).toBe(`${uri}/index.html`),
  );

  // A route the app has but search engines should not.
  it.each(["/login", "/signup"])("serves the app shell for the app-only route %s", (uri) =>
    expect(rewrite(uri)).toBe("/index.html"),
  );

  // Anything else is a not-found page with a real 404 status (#63):
  // /404.html carries `status=404` metadata that object-status.js applies.
  it.each(["/tasks/text-to-image", "/nope", "/home/extra"])(
    "serves /404.html for the path with no page %s",
    (uri) => expect(rewrite(uri)).toBe("/404.html"),
  );

  it("does not mistake an Object.prototype key for a route", () => {
    expect(rewrite("/constructor")).toBe("/404.html");
    expect(rewrite("/tostring")).toBe("/404.html");
  });

  // One URL per page (#63).
  it("redirects / to /home", () => {
    expect(location("/")).toBe(`https://${HOST}/home`);
  });

  // Search Console called the relative `Location: /home` a "Redirect error".
  it("sends an absolute Location, falling back to a relative one with no Host", () => {
    expect(location("/ASR")).toBe(`https://${HOST}/asr`);
    const out = handler({ request: { uri: "/ASR", querystring: {} } }) as Redirect;
    expect(out.headers.location.value).toBe("/asr");
  });

  it.each([
    ["/asr/", "/asr"],
    ["/ASR", "/asr"],
    ["/Text-Classification/", "/text-classification"],
    ["/home//", "/home"],
    ["/Nope", "/nope"],
  ])("redirects %s to its canonical form %s", (from, to) => {
    expect(location(from)).toBe(`https://${HOST}${to}`);
  });

  it("keeps the query string on a redirect", () => {
    expect(location("/ASR", { utm_source: { value: "x" }, flag: { value: "" } })).toBe(
      `https://${HOST}/asr?utm_source=x&flag`,
    );
    expect(
      location("/", { t: { value: "a", multiValue: [{ value: "a" }, { value: "b" }] } }),
    ).toBe(`https://${HOST}/home?t=a&t=b`);
  });

  it.each([
    "/index.html",
    "/assets/index-abc123.js",
    "/assets/ort-wasm-simd-threaded.jsep-B0T3yYHD.wasm",
    "/assets/cora-x1.bin",
    "/favicon.ico",
    "/site.webmanifest",
    "/robots.txt",
    "/sitemap.xml",
    "/asr/index.html",
    "/404.html",
  ])("fetches the file %s as itself", (uri) => expect(rewrite(uri)).toBe(uri));

  it("leaves the query string alone", () => {
    const querystring = { q: { value: "1" } };
    const out = handler({ request: { uri: "/asr", querystring } }) as Request;
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
  });

  // A route the router has but the function does not know would answer 404:
  // each one is either an indexable page (its own file) or an app route.
  it("knows every route in the router, so none answers 404", () => {
    const tree = readFileSync(
      join(__dirname, "../../frontend/src/routeTree.gen.ts"),
      "utf8",
    );
    const paths = [...new Set([...tree.matchAll(/\bpath: '([^']*)'/g)].map((m) => m[1]))];
    const unknown = paths.filter((p) => p !== "/" && rewrite(p) === "/404.html");
    expect(unknown, "add each to ROUTES (indexable) or APP_ROUTES").toEqual([]);
  });

  it("stays well inside CloudFront's 10 KB function limit", () => {
    expect(Buffer.byteLength(source)).toBeLessThan(8 * 1024);
  });
});
