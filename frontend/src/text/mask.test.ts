import { describe, expect, it } from "vitest";

import {
  cleanFill,
  countMasks,
  expandTemplate,
  findMasks,
  insertMask,
  retargetMasks,
  spliceFill,
} from "./mask";

// The three tokenizer families `/fill-mask` ships, as a table. Every test that
// can run against all of them does, because a function that is only ever
// exercised with `[MASK]` is exactly how the literal creeps back in.
const TOKENS = ["[MASK]", "<mask>"] as const;

describe("findMasks / countMasks", () => {
  it.each(TOKENS)("finds the span of a single %s", (mask) => {
    const text = `The capital of France is ${mask}.`;
    expect(findMasks(text, mask)).toEqual([
      { start: 25, end: 25 + mask.length },
    ]);
    expect(text.slice(25, 25 + mask.length)).toBe(mask);
  });

  it.each(TOKENS)("counts every occurrence of %s", (mask) => {
    expect(countMasks(`The ${mask} of France is ${mask}.`, mask)).toBe(2);
    expect(countMasks("no mask here", mask)).toBe(0);
  });

  it("does not treat the token as a pattern", () => {
    // `[MASK]` is a character class if it ever reaches a regex by accident,
    // and would then match the single letters M, A, S and K.
    expect(countMasks("MASK", "[MASK]")).toBe(0);
    expect(countMasks("A K", "[MASK]")).toBe(0);
  });

  it("finds a mask that is the whole input", () => {
    expect(findMasks("[MASK]", "[MASK]")).toEqual([{ start: 0, end: 6 }]);
  });
});

describe("insertMask", () => {
  it("appends to an empty box without a leading space", () => {
    expect(insertMask("", "[MASK]")).toEqual({ text: "[MASK]", caret: 6 });
  });

  it("separates the token from the preceding word", () => {
    // `is[MASK]` tokenizes differently from `is [MASK]` on a WordPiece
    // vocabulary, so this is a correctness rule rather than a tidiness one.
    const { text, caret } = insertMask("The capital is", "[MASK]");
    expect(text).toBe("The capital is [MASK]");
    expect(caret).toBe(text.length);
  });

  it("does not double a space the user already typed", () => {
    expect(insertMask("The capital is ", "<mask>").text).toBe(
      "The capital is <mask>",
    );
  });

  it("inserts in front of punctuation without spacing it off", () => {
    const text = "The capital of France is .";
    // Caret between the space and the full stop.
    expect(insertMask(text, "[MASK]", 25, 25).text).toBe(
      "The capital of France is [MASK].",
    );
  });

  it("adds a trailing space before a word", () => {
    expect(insertMask("The capital", "[MASK]", 0, 0).text).toBe(
      "[MASK] The capital",
    );
  });

  it("replaces a selection rather than inserting beside it", () => {
    const { text, caret } = insertMask("The capital is Paris.", "[MASK]", 15, 20);
    expect(text).toBe("The capital is [MASK].");
    expect(caret).toBe(21);
  });
});

describe("retargetMasks", () => {
  it("rewrites the other family's token to this model's", () => {
    expect(
      retargetMasks("The capital of France is [MASK].", [...TOKENS], "<mask>"),
    ).toBe("The capital of France is <mask>.");
  });

  it("leaves text that already uses this model's token alone", () => {
    const text = "The capital of France is <mask>.";
    expect(retargetMasks(text, [...TOKENS], "<mask>")).toBe(text);
  });

  it("converts each token once, never twice", () => {
    // A naive two-pass rewrite (`<mask>` → `[MASK]`, then `[MASK]` → `<mask>`)
    // turns both tokens into the same one and silently loses the distinction.
    expect(
      retargetMasks("a [MASK] and a <mask>", [...TOKENS], "[MASK]"),
    ).toBe("a [MASK] and a [MASK]");
  });

  it("rewrites every occurrence, not only the first", () => {
    expect(retargetMasks("[MASK] and [MASK]", [...TOKENS], "<mask>")).toBe(
      "<mask> and <mask>",
    );
  });

  it("is a no-op when the catalogue has one family", () => {
    expect(retargetMasks("a [MASK] b", ["[MASK]"], "[MASK]")).toBe("a [MASK] b");
  });

  it("does not corrupt text with no mask in it", () => {
    const text = "Nothing to see here.";
    expect(retargetMasks(text, [...TOKENS], "<mask>")).toBe(text);
  });
});

describe("expandTemplate", () => {
  it.each(TOKENS)("puts %s where the placeholder was", (mask) => {
    expect(expandTemplate("The capital of France is {}.", mask)).toBe(
      `The capital of France is ${mask}.`,
    );
  });

  it("expands every placeholder", () => {
    expect(expandTemplate("{} and {}", "[MASK]")).toBe("[MASK] and [MASK]");
  });
});

describe("spliceFill", () => {
  it("puts the filling where the mask was and returns its span", () => {
    const text = "The capital of France is [MASK].";
    const span = findMasks(text, "[MASK]")[0];
    const out = spliceFill(text, span, "Paris");

    expect(out.text).toBe("The capital of France is Paris.");
    // The span points at the filling itself, in the *new* string.
    expect(out.text.slice(out.span.start, out.span.end)).toBe("Paris");
  });

  it("preserves the user's own casing and spacing", () => {
    // The pipeline also returns a decoded `sequence`, and on an uncased model
    // that is "the capital of france is paris." — rendering it would silently
    // rewrite the user's sentence.
    const text = "The  CAPITAL of France is <mask>!";
    const span = findMasks(text, "<mask>")[0];
    expect(spliceFill(text, span, "Paris").text).toBe(
      "The  CAPITAL of France is Paris!",
    );
  });

  it("handles a mask at the very start", () => {
    const out = spliceFill("[MASK] is the capital.", { start: 0, end: 6 }, "Paris");
    expect(out.text).toBe("Paris is the capital.");
    expect(out.span).toEqual({ start: 0, end: 5 });
  });
});

describe("cleanFill", () => {
  it("strips the leading space a byte-level BPE decode leaves on", () => {
    // RoBERTa and ModernBERT return " Paris"; BERT returns "paris".
    expect(cleanFill(" Paris")).toBe("Paris");
    expect(cleanFill("paris")).toBe("paris");
  });
});
