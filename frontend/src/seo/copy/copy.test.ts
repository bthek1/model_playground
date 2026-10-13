import { describe, expect, it } from "vitest";

import { SITE_PAGES } from "../pages";
import { PAGE_COPY } from "./index";

const tasks = SITE_PAGES.filter((p) => p.kind === "task");
const words = (text: string) => text.split(/\s+/).filter(Boolean).length;

// #71: the words a task page is found by. Each check is a way the copy could
// stop doing that job without anything else failing.
describe("PAGE_COPY", () => {
  it("covers every task page, and nothing else", () => {
    expect(Object.keys(PAGE_COPY).sort()).toEqual(tasks.map((p) => p.path).sort());
  });

  describe.each(tasks.map((p) => [p.path, p] as const))("%s", (path, page) => {
    const copy = PAGE_COPY[path];

    it("opens with a one-sentence tagline", () => {
      expect(copy.tagline.length).toBeGreaterThanOrEqual(40);
      expect(copy.tagline.length).toBeLessThanOrEqual(140);
      expect(copy.tagline).toMatch(/[.!?]$/);
    });

    // Enough to say what the page is in its own words — the thing it lacked —
    // and short enough that it stays a note under a tool, not an article.
    it("has two to four paragraphs, 120–320 words in all", () => {
      expect(copy.about.length).toBeGreaterThanOrEqual(2);
      expect(copy.about.length).toBeLessThanOrEqual(4);
      const total = copy.about.reduce((n, p) => n + words(p), 0);
      expect(total).toBeGreaterThanOrEqual(120);
      expect(total).toBeLessThanOrEqual(320);
    });

    // An alias is a phrase someone searches for. One that appears nowhere on
    // the page is a hidden keyword, which helps nothing and reads as spam.
    it("uses every alias in its own title or text", () => {
      expect(copy.aliases.length).toBeGreaterThanOrEqual(3);
      const text = [page.title, copy.tagline, ...copy.about].join(" ").toLowerCase();
      const unused = copy.aliases.filter((a) => !text.includes(a.toLowerCase()));
      expect(unused).toEqual([]);
    });
  });

  // Two pages sharing a sentence look like near-duplicates — the problem the
  // sibling descriptions caused, reintroduced by copy-and-paste.
  it("shares no tagline and no paragraph between pages", () => {
    const all = Object.values(PAGE_COPY).flatMap((c) => [c.tagline, ...c.about]);
    expect(new Set(all).size).toBe(all.length);
  });
});
