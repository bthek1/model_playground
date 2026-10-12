import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

// Evaluate the file's exact text, as CloudFront will (see spa-rewrite.test.ts).
const source = readFileSync(join(__dirname, "object-status.js"), "utf8");
type Response = {
  statusCode: number;
  statusDescription?: string;
  headers: Record<string, { value: string }>;
};
const handler = new Function(`${source}\nreturn handler;`)() as (event: {
  request: { uri: string };
  response: Response;
}) => Response;

const respond = (headers: Record<string, { value: string }>) =>
  handler({
    request: { uri: "/whatever" },
    response: { statusCode: 200, statusDescription: "OK", headers: { ...headers } },
  });

describe("object-status (CloudFront viewer-response)", () => {
  // /404.html is uploaded with `--metadata status=404` (#63).
  it("turns x-amz-meta-status: 404 into a real 404, and drops the header", () => {
    const out = respond({
      "x-amz-meta-status": { value: "404" },
      "content-type": { value: "text/html; charset=utf-8" },
    });
    expect(out.statusCode).toBe(404);
    expect(out.statusDescription).toBe("Not Found");
    expect(out.headers["x-amz-meta-status"]).toBeUndefined();
    expect(out.headers["content-type"]).toEqual({ value: "text/html; charset=utf-8" });
  });

  it("leaves every other response alone", () => {
    const out = respond({ "content-type": { value: "text/html" } });
    expect(out.statusCode).toBe(200);
    expect(out.statusDescription).toBe("OK");
  });

  // Anything but a 4xx could make a page a redirect with no Location, or an
  // error page the browser will not render the app for.
  it.each(["301", "500", "200", "abc", ""])("ignores the non-4xx value %j, but still drops it", (v) => {
    const out = respond({ "x-amz-meta-status": { value: v } });
    expect(out.statusCode).toBe(200);
    expect(out.headers["x-amz-meta-status"]).toBeUndefined();
  });

  it("is keyed on the object, never on the URL", () => {
    expect(source).not.toMatch(/request\.uri/);
  });
});
