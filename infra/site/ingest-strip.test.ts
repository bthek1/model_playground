import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

// Evaluate the file's exact text, as CloudFront will (see spa-rewrite.test.ts).
const source = readFileSync(join(__dirname, "ingest-strip.js"), "utf8");
const handler = new Function(`${source}\nreturn handler;`)() as (event: {
  request: { uri: string; querystring?: unknown };
}) => { uri: string; querystring?: unknown };

const strip = (uri: string) => handler({ request: { uri } }).uri;

describe("ingest-strip (CloudFront viewer-request on /ingest/*)", () => {
  it.each([
    ["/ingest/e/", "/e/"],
    ["/ingest/i/v0/e/", "/i/v0/e/"],
    ["/ingest/flags/", "/flags/"],
    ["/ingest/", "/"],
    ["/ingest", "/"],
  ])("forwards %s to PostHog as %s", (from, to) => expect(strip(from)).toBe(to));

  it("never rewrites to the app shell, which would swallow events", () => {
    expect(strip("/ingest/e/")).not.toBe("/index.html");
  });

  it("only strips a whole leading segment", () => {
    expect(strip("/ingestion/e/")).toBe("/ingestion/e/");
    expect(strip("/assets/ingest/x.js")).toBe("/assets/ingest/x.js");
  });

  it("leaves the query string alone (the SDK sends compression and version there)", () => {
    const querystring = { compression: { value: "gzip-js" } };
    expect(handler({ request: { uri: "/ingest/e/", querystring } }).querystring).toBe(
      querystring,
    );
  });
});
