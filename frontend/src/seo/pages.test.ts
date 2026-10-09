import { describe, expect, it } from "vitest";

import { taskCategories } from "@/components/layout/taskTaxonomy";
import { SITE_DESCRIPTION, SITE_URL } from "@/lib/site";
import {
  canonicalUrl,
  categoryAnchor,
  normalisePath,
  pageForPath,
  SITE_PAGES,
  siblingsOf,
} from "./pages";

const tasks = taskCategories.flatMap((c) => c.tasks);

describe("SITE_PAGES", () => {
  it("lists home first, then every taxonomy task, then the legal pages", () => {
    expect(SITE_PAGES[0].path).toBe("/home");
    expect(SITE_PAGES.filter((p) => p.kind === "task").map((p) => p.path)).toEqual(
      tasks.map((t) => t.to),
    );
    expect(SITE_PAGES.filter((p) => p.kind === "static").map((p) => p.path)).toEqual([
      "/privacy",
      "/terms",
      "/licences",
      "/accessibility",
    ]);
  });

  // An account form is not a search result, and `/` is a redirect (#63).
  it("leaves out sign-in, sign-up and the root", () => {
    const paths = SITE_PAGES.map((p) => p.path);
    for (const p of ["/", "/login", "/signup"]) expect(paths).not.toContain(p);
  });

  it("gives every page a unique path, title and description", () => {
    for (const key of ["path", "title", "description"] as const) {
      const values = SITE_PAGES.map((p) => p[key]);
      expect(new Set(values).size, key).toBe(values.length);
    }
  });

  // Google shows roughly 50–60 characters of a title and 155–160 of a
  // description; past that it truncates, and a page that says nothing in the
  // part shown might as well say nothing.
  it("keeps titles and descriptions to what a result shows", () => {
    for (const p of SITE_PAGES) {
      expect(p.title.length, p.title).toBeLessThanOrEqual(70);
      expect(p.description.length, p.path).toBeGreaterThanOrEqual(70);
      expect(p.description.length, p.path).toBeLessThanOrEqual(160);
    }
  });

  it("uses lower-case, slash-free paths — the form the edge redirects to", () => {
    for (const p of SITE_PAGES) expect(normalisePath(p.path)).toBe(p.path);
  });

  it("describes home with the site-wide description", () => {
    expect(pageForPath("/home")?.description).toBe(SITE_DESCRIPTION);
  });
});

describe("pageForPath", () => {
  it("tolerates a trailing slash and upper case", () => {
    expect(pageForPath("/ASR/")?.heading).toBe("Automatic Speech Recognition");
  });

  it("knows nothing of an unlisted path", () => {
    expect(pageForPath("/login")).toBeUndefined();
    expect(pageForPath("/nope")).toBeUndefined();
  });
});

describe("helpers", () => {
  it("builds absolute canonicals against the public origin by default", () => {
    expect(canonicalUrl("/asr")).toBe(`${SITE_URL}/asr`);
    expect(canonicalUrl("/asr", "https://example.test/")).toBe("https://example.test/asr");
  });

  it("anchors a category by its label", () => {
    expect(categoryAnchor("Computer Vision")).toBe("computer-vision");
    expect(categoryAnchor("Natural Language Processing")).toBe("natural-language-processing");
  });

  it("finds a task's siblings in its own category, itself excluded", () => {
    const sibs = siblingsOf(pageForPath("/asr")!);
    expect(sibs.map((p) => p.path)).not.toContain("/asr");
    expect(new Set(sibs.map((p) => p.category))).toEqual(new Set(["Audio"]));
    expect(siblingsOf(pageForPath("/privacy")!)).toEqual([]);
  });
});
