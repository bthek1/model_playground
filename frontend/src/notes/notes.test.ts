import { readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { pageForPath, SITE_PAGES } from "@/seo/pages";
import { NOTE_BODIES } from "./bodies";
import { parseInline, plainText } from "./inline";
import { NOTES, notePath, notesFor } from "./meta";
import type { NoteBlock } from "./types";

const texts = (blocks: readonly NoteBlock[]) =>
  blocks.flatMap((b) => (b.kind === "ul" ? b.items : b.kind === "code" ? [] : [b.text]));

describe("NOTES", () => {
  it("has unique, kebab-case slugs with no dots (a dot would bypass the edge rewrite)", () => {
    const slugs = NOTES.map((n) => n.slug);
    expect(new Set(slugs).size).toBe(slugs.length);
    for (const s of slugs) expect(s).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
  });

  // The same limits as a task's search title and every page's description.
  it("keeps titles to 51 characters and descriptions to 70–150", () => {
    for (const n of NOTES) {
      expect(n.title.length, n.title).toBeLessThanOrEqual(51);
      expect(n.description.length, n.slug).toBeGreaterThanOrEqual(70);
      expect(n.description.length, n.slug).toBeLessThanOrEqual(150);
    }
  });

  it("dates each note, never modified before it was published", () => {
    for (const n of NOTES) {
      expect(n.published).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(n.modified >= n.published, n.slug).toBe(true);
    }
  });

  it("names only real task pages as related, so each one can link back", () => {
    for (const n of NOTES) {
      expect(n.related.length, n.slug).toBeGreaterThan(0);
      for (const path of n.related) expect(pageForPath(path)?.kind, `${n.slug} → ${path}`).toBe("task");
    }
  });

  it("is listed in SITE_PAGES at /notes/<slug>", () => {
    for (const n of NOTES) expect(pageForPath(notePath(n.slug))?.kind).toBe("note");
  });

  it("finds the notes that cite a page", () => {
    expect(notesFor("/translation").map((n) => n.slug)).toContain("ort-web-missing-required-scale");
    expect(notesFor("/home")).toEqual([]);
  });
});

describe("NOTE_BODIES", () => {
  it("has exactly one body per note, and one file per body", () => {
    expect(Object.keys(NOTE_BODIES).sort()).toEqual(NOTES.map((n) => n.slug).sort());
    const files = readdirSync(join(__dirname, "bodies"))
      .filter((f) => f !== "index.ts")
      .map((f) => f.replace(/\.ts$/, ""));
    expect(files.sort()).toEqual(NOTES.map((n) => n.slug).sort());
  });

  // An article, not a stub: thin pages are what "crawled, not indexed" means.
  it("gives every note at least 350 words of prose", () => {
    for (const [slug, blocks] of Object.entries(NOTE_BODIES)) {
      const words = texts(blocks).map(plainText).join(" ").split(/\s+/).filter(Boolean).length;
      expect(words, slug).toBeGreaterThanOrEqual(350);
    }
  });

  it("opens with a paragraph, so the lead is what a snippet quotes", () => {
    for (const [slug, blocks] of Object.entries(NOTE_BODIES)) expect(blocks[0].kind, slug).toBe("p");
  });

  it("links only to pages that exist", () => {
    const paths = new Set(SITE_PAGES.map((p) => p.path));
    for (const [slug, blocks] of Object.entries(NOTE_BODIES)) {
      for (const text of texts(blocks)) {
        for (const t of parseInline(text)) {
          if (t.kind === "link" && t.href.startsWith("/")) expect(paths.has(t.href), `${slug} → ${t.href}`).toBe(true);
        }
      }
    }
  });

  it("leaves no stray markup a reader would see", () => {
    for (const [slug, blocks] of Object.entries(NOTE_BODIES)) {
      for (const text of texts(blocks)) {
        const plain = plainText(text);
        expect(plain, slug).not.toMatch(/`|\]\(/);
      }
    }
  });
});

describe("parseInline", () => {
  it("splits code and links out of text, in order", () => {
    expect(parseInline("Use `q8` on [translation](/translation) now.")).toEqual([
      { kind: "text", text: "Use " },
      { kind: "code", text: "q8" },
      { kind: "text", text: " on " },
      { kind: "link", text: "translation", href: "/translation" },
      { kind: "text", text: " now." },
    ]);
  });

  it("takes absolute https links and nothing else as a link", () => {
    expect(parseInline("[a](https://x.test/y)")).toEqual([{ kind: "link", text: "a", href: "https://x.test/y" }]);
    expect(parseInline("[a](javascript:alert(1))")[0].kind).toBe("text");
  });

  it("returns plain text unchanged", () => {
    expect(parseInline("no markup")).toEqual([{ kind: "text", text: "no markup" }]);
    expect(parseInline("")).toEqual([]);
  });
});
