import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { pageForPath, SITE_PAGES, type SitePage } from "./pages";
import { crawlableContent, seoManifest } from "./render";

const PUBLIC = join(__dirname, "../../public");
const ORIGIN = "https://example.test";
const sha256 = (t: string) => createHash("sha256").update(t).digest("hex");

// IndexNow (#69): a search engine fetches https://<host>/<key>.txt and checks
// it contains the key before it trusts a submission from us.
describe("the IndexNow key file", () => {
  const keys = readdirSync(PUBLIC).filter((f) => /^[0-9a-f]{32}\.txt$/.test(f));

  it("is exactly one 32-hex file at the site root, containing its own name", () => {
    expect(keys).toHaveLength(1);
    const key = keys[0].replace(/\.txt$/, "");
    expect(readFileSync(join(PUBLIC, keys[0]), "utf8").trim()).toBe(key);
  });
});

describe("seoManifest", () => {
  it("has one digest per indexable page", () => {
    const manifest = seoManifest(SITE_PAGES, ORIGIN, sha256);
    expect(Object.keys(manifest)).toEqual(SITE_PAGES.map((p) => p.path));
    for (const d of Object.values(manifest)) expect(d).toMatch(/^[0-9a-f]{64}$/);
  });

  // What the built HTML changes on almost every deploy — the hashed chunk
  // names — is not in the input at all, so an unchanged page stays unchanged.
  it("is built from what a crawler reads, never from the asset references", () => {
    const content = crawlableContent(pageForPath("/asr")!, ORIGIN);
    expect(content).not.toMatch(/\/assets\//);
    expect(content).toContain("Speech to Text with Whisper, in your browser");
    expect(content).toContain(`${ORIGIN}/asr`);
  });

  it("is deterministic", () => {
    expect(seoManifest(SITE_PAGES, ORIGIN, sha256)).toEqual(seoManifest(SITE_PAGES, ORIGIN, sha256));
  });

  const digestOf = (page: SitePage) => sha256(crawlableContent(page, ORIGIN));
  const asr = pageForPath("/asr")!;

  it.each([
    ["title", { title: "Something else" }],
    ["description", { description: "A different description of the page, long enough to be a real one." }],
    ["heading", { heading: "Speech to Text" }],
  ] as const)("moves when the %s changes", (_, change) => {
    expect(digestOf({ ...asr, ...change })).not.toBe(digestOf(asr));
  });

  // A sibling's link sits in this page's static body, so renaming one task
  // re-submits its neighbours too — correctly, since their HTML changed. Its
  // description does not (#71), so rewording one page re-submits only itself.
  it("covers its siblings' links but not their descriptions", () => {
    const vad = pageForPath("/vad")!;
    const asrBody = crawlableContent(asr, ORIGIN);
    expect(asrBody).toContain(vad.heading);
    expect(asrBody).not.toContain(vad.description);
  });

  // The page's own copy is what a crawler reads most of, so rewording it must
  // re-submit the page — and the manifest only sees it if the build passes it.
  it("moves when the page's own copy changes", () => {
    const copy = { tagline: "One sentence of tagline.", about: ["A paragraph."], aliases: [] };
    const reworded = { ...copy, about: ["A different paragraph."] };
    const one = seoManifest([asr], ORIGIN, sha256, () => copy);
    const two = seoManifest([asr], ORIGIN, sha256, () => reworded);
    expect(one["/asr"]).not.toBe(two["/asr"]);
  });

  it("depends on the origin, since canonicals do", () => {
    expect(sha256(crawlableContent(asr, ORIGIN))).not.toBe(
      sha256(crawlableContent(asr, "https://other.test")),
    );
  });
});
