import { readFileSync } from "node:fs";
import { join } from "node:path";
import { beforeEach, describe, expect, it } from "vitest";

import { SITE_DESCRIPTION } from "@/lib/site";
import { pageForPath } from "@/seo/pages";
import { renderPageHtml, structuredData } from "@/seo/render";
import { syncHead } from "./useDocumentHead";

const TEMPLATE = readFileSync(join(__dirname, "../../index.html"), "utf8");
const ORIGIN = "https://example.test";

/** A document as the edge would have served it for `path`. */
function landedOn(path: string): Document {
  const page = pageForPath(path);
  const html = page ? renderPageHtml(TEMPLATE, page, ORIGIN) : TEMPLATE;
  return new DOMParser().parseFromString(html, "text/html");
}

const meta = (doc: Document, sel: string) =>
  doc.head.querySelector<HTMLMetaElement>(`meta[${sel}]`)?.content;
const canonical = (doc: Document) =>
  doc.head.querySelector<HTMLLinkElement>('link[rel="canonical"]')?.getAttribute("href");

describe("syncHead", () => {
  let doc: Document;
  beforeEach(() => {
    doc = landedOn("/asr");
  });

  // Client-side navigation never fetches the next route's index.html, so the
  // head has to be brought along by hand (#64).
  it("rewrites the head for the page navigated to", () => {
    syncHead("/depth", doc);
    const depth = pageForPath("/depth")!;
    expect(doc.title).toBe(depth.title);
    expect(meta(doc, 'name="description"')).toBe(depth.description);
    expect(meta(doc, 'property="og:title"')).toBe(depth.title);
    expect(meta(doc, 'property="og:url"')).toBe(`${ORIGIN}/depth`);
    expect(canonical(doc)).toBe(`${ORIGIN}/depth`);
    expect(JSON.parse(doc.getElementById("structured-data")!.textContent!)).toEqual(
      structuredData(depth, ORIGIN),
    );
  });

  it("leaves the head exactly as the edge served it for the landing page", () => {
    const served = doc.head.innerHTML;
    syncHead("/asr", doc);
    expect(doc.head.innerHTML).toBe(served);
  });

  it("writes canonicals against the build's origin, read off og:image", () => {
    syncHead("/privacy", doc);
    expect(canonical(doc)).toBe(`${ORIGIN}/privacy`);
  });

  // A page that is not indexable must not keep claiming to be another one.
  it("drops the canonical and structured data on an unlisted path", () => {
    syncHead("/login", doc);
    expect(doc.title).toBe("Sign in · Model Playground");
    expect(meta(doc, 'name="description"')).toBe(SITE_DESCRIPTION);
    expect(canonical(doc)).toBeUndefined();
    expect(meta(doc, 'property="og:url"')).toBeUndefined();
    expect(doc.getElementById("structured-data")).toBeNull();
  });

  it("adds what the shell lacks when navigating from it to a page", () => {
    const shell = landedOn("/nope");
    syncHead("/asr", shell);
    expect(shell.head.querySelectorAll('link[rel="canonical"]')).toHaveLength(1);
    expect(shell.head.querySelectorAll("#structured-data")).toHaveLength(1);
    // The source index.html's og:image is relative — no origin to read — so
    // the public one is used.
    expect(canonical(shell)).toBe("https://playground.benedictthekkel.com/asr");
  });
});
