import { describe, expect, it } from "vitest";

import { buildIndex, DEFAULT_BM25, idf, search, tokenize } from "./bm25";

const CORPUS = [
  "the cat sat on the mat",
  "the dog sat on the log",
  "a completely unrelated sentence about compilers",
];

describe("tokenize", () => {
  it("lowercases and splits on non-letters", () => {
    expect(tokenize("The Cat's mat, again!")).toEqual([
      "the",
      "cat",
      "s",
      "mat",
      "again",
    ]);
  });

  it("returns nothing for empty or punctuation-only input", () => {
    expect(tokenize("")).toEqual([]);
    expect(tokenize("—  ,, ")).toEqual([]);
  });

  it("keeps digits and non-Latin letters", () => {
    expect(tokenize("GPT-2 costs 40€ in Köln")).toEqual([
      "gpt",
      "2",
      "costs",
      "40",
      "in",
      "köln",
    ]);
  });
});

describe("idf", () => {
  // The property the whole scoring rests on — and the reason the `log(1 + …)`
  // form is used rather than the textbook one, which goes *negative* for a term
  // in more than half the corpus and would invert the ranking for exactly the
  // stopword-ish terms people type.
  it("gives a term in every document ~zero, and never a negative", () => {
    const index = buildIndex(["a b", "a c", "a d"]);
    expect(idf(index, "a")).toBeGreaterThan(0);
    expect(idf(index, "a")).toBeLessThan(0.2);
    // And a rare term is worth much more.
    expect(idf(index, "b")).toBeGreaterThan(idf(index, "a") * 5);
  });

  it("never goes negative for a term in most of the corpus", () => {
    const index = buildIndex(["x y", "x z", "x w", "q"]);
    expect(idf(index, "x")).toBeGreaterThan(0);
  });

  it("treats an unseen term as maximally rare rather than throwing", () => {
    const index = buildIndex(CORPUS);
    expect(idf(index, "nonexistent")).toBeGreaterThan(0);
  });
});

describe("search", () => {
  it("ranks the document that shares the query's words first", () => {
    const index = buildIndex(CORPUS);
    const ranked = search(index, "cat mat");
    expect(ranked[0].doc).toBe(0);
    expect(ranked[0].score).toBeGreaterThan(0);
  });

  // A term appearing in every document must not decide the ranking, which is
  // the same property as the IDF test above, observed end to end.
  it("is not swayed by a term the whole corpus shares", () => {
    // Every document contains "the", so it carries almost no information — and
    // the comparison is against a *rare* term rather than an absolute number,
    // which is what the property actually says.
    const index = buildIndex([
      "the cat sat on the mat",
      "the dog sat on the log",
      "the compiler emitted the warning",
    ]);
    const common = search(index, "the")[0].score;
    const rare = search(index, "compiler")[0].score;
    expect(common).toBeGreaterThan(0);
    expect(rare).toBeGreaterThan(common * 4);
  });

  // The other defining property: `b`. Without length normalisation a long
  // document accumulates matches for free.
  it("does not let a longer document win on length alone", () => {
    const short = "webgpu compute shaders";
    const padded = `${short} ${"filler words here ".repeat(40)}`;
    const index = buildIndex([short, padded]);

    const ranked = search(index, "webgpu compute shaders");
    expect(ranked[0].doc, "the short exact match should win").toBe(0);

    // And with `b = 0` — normalisation off — the guarantee is gone, which is
    // what makes it a real parameter rather than a constant.
    const unnormalised = search(index, "webgpu compute shaders", {
      ...DEFAULT_BM25,
      b: 0,
    });
    expect(unnormalised[0].score).toBeCloseTo(unnormalised[1].score, 6);
  });

  it("saturates term frequency, so repetition stops paying", () => {
    const index = buildIndex(["spam", "spam spam spam spam spam spam"]);
    const ranked = search(index, "spam");
    // The repeated document wins, but nowhere near six times over.
    expect(ranked[0].doc).toBe(1);
    expect(ranked[0].score).toBeLessThan(ranked[1].score * 2.5);
  });

  it("scores everything zero for a query term absent from the corpus", () => {
    const index = buildIndex(CORPUS);
    const ranked = search(index, "helicopter");
    expect(ranked.every((r) => r.score === 0)).toBe(true);
    // Order is still stable rather than arbitrary.
    expect(ranked.map((r) => r.doc)).toEqual([0, 1, 2]);
  });

  it("handles an empty corpus, an empty query and empty documents", () => {
    expect(search(buildIndex([]), "anything")).toEqual([]);
    const index = buildIndex(CORPUS);
    expect(search(index, "").every((r) => r.score === 0)).toBe(true);
    // No NaN anywhere — a NaN ranks unpredictably instead of failing.
    const blank = buildIndex(["", "   ", "real words"]);
    for (const r of search(blank, "real")) {
      expect(Number.isFinite(r.score)).toBe(true);
    }
  });

  it("keeps corpus order on ties, so a re-score does not reshuffle", () => {
    const index = buildIndex(["a x", "a y", "a z"]);
    expect(search(index, "a").map((r) => r.doc)).toEqual([0, 1, 2]);
  });
});
