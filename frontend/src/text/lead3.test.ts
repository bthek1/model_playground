import { describe, expect, it } from "vitest";

import { lead, sentenceCount, splitSentences, wordCount } from "./lead3";

const ARTICLE =
  "The agency confirmed the launch on Tuesday. Four satellites reached orbit. " +
  "Officials said the restart sequence worked. A fourth flight is planned.";

describe("splitSentences", () => {
  it("returns offsets into the original string", () => {
    for (const s of splitSentences(ARTICLE)) {
      expect(ARTICLE.slice(s.start, s.end)).toBe(s.text);
    }
  });

  it("keeps a trailing sentence with no final punctuation", () => {
    const s = splitSentences("First one. Then this one trails off");
    expect(s).toHaveLength(2);
    expect(s[1].text).toBe("Then this one trails off");
  });

  it("does not split a decimal or an abbreviation with no space after it", () => {
    expect(splitSentences("It rose 3.5 percent overnight.")).toHaveLength(1);
  });

  // The documented limitation, asserted rather than left implicit: a page that
  // claims its splitter is bad should be able to show where.
  it("splits an abbreviation wrongly, which is the stated limitation", () => {
    const s = splitSentences("Dr. Smith agreed. She left.");
    expect(s.length).toBeGreaterThan(2);
    expect(s[0].text).toBe("Dr.");
  });

  it("handles a run of terminators", () => {
    const s = splitSentences("Really?! I had no idea.");
    expect(s).toHaveLength(2);
    expect(s[0].text).toBe("Really?!");
  });

  it("returns nothing for empty or whitespace-only input", () => {
    expect(splitSentences("")).toEqual([]);
    expect(splitSentences("   \n ")).toEqual([]);
  });
});

describe("lead", () => {
  // The property the whole module exists for. A baseline assembled by joining
  // split pieces loses whichever whitespace the split consumed, so it quietly
  // differs from the article it claims to be quoting.
  it("returns a slice of the input, never a rebuild", () => {
    const out = lead(ARTICLE, 3);
    expect(ARTICLE).toContain(out);
  });

  it("takes the first three sentences", () => {
    expect(lead(ARTICLE, 3)).toBe(
      "The agency confirmed the launch on Tuesday. Four satellites reached orbit. " +
        "Officials said the restart sequence worked.",
    );
  });

  it("returns what there is when the text has fewer than three sentences", () => {
    const short = "Only one sentence here.";
    expect(lead(short, 3)).toBe(short);
    expect(ARTICLE).toContain(lead(ARTICLE, 99));
  });

  it("returns one sentence for text with no trailing punctuation", () => {
    expect(lead("no punctuation at all", 3)).toBe("no punctuation at all");
  });

  it("returns an empty string for empty input rather than throwing", () => {
    expect(lead("", 3)).toBe("");
    expect(lead("   ", 3)).toBe("");
    expect(lead(ARTICLE, 0)).toBe("");
  });

  it("preserves the article's own whitespace between sentences", () => {
    const spaced = "One sentence.\n\nTwo sentence. Three sentence. Four.";
    expect(spaced).toContain(lead(spaced, 3));
    expect(lead(spaced, 3)).toContain("\n\n");
  });
});

describe("counts", () => {
  it("counts sentences and words", () => {
    expect(sentenceCount(ARTICLE)).toBe(4);
    expect(wordCount("one two  three\nfour")).toBe(4);
    expect(wordCount("  ")).toBe(0);
  });
});
