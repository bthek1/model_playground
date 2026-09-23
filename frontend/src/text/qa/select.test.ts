import { describe, expect, it } from "vitest";

import { contextRange, selectAnswerSpan } from "./select";

// `[CLS] Who built it ? [SEP] Gustave Eiffel did . [SEP]`
const CLS = 101;
const SEP = 102;
const PAD = 0;
const SPECIAL = [CLS, SEP, PAD];
const IDS = [CLS, 10, 11, 12, 13, SEP, 20, 21, 22, 23, SEP];
const MASK = IDS.map(() => 1);

/** Logits that put all the mass on one token index. */
function peak(at: number, length = IDS.length, height = 10): number[] {
  return Array.from({ length }, (_, i) => (i === at ? height : 0));
}

describe("contextRange", () => {
  it("starts one past the first separator and stops at the next special", () => {
    expect(contextRange(IDS, SEP, SPECIAL)).toEqual({ start: 6, end: 10 });
  });

  it("excludes padding, which is a special token", () => {
    const padded = [...IDS, PAD, PAD];
    expect(contextRange(padded, SEP, SPECIAL)).toEqual({ start: 6, end: 10 });
  });

  it("returns an empty range when there is no separator at all", () => {
    expect(contextRange([CLS, 10, 11], SEP, SPECIAL)).toEqual({
      start: 0,
      end: 0,
    });
  });
});

describe("selectAnswerSpan", () => {
  const base = {
    ids: IDS,
    mask: MASK,
    specialIds: SPECIAL,
    sepTokenId: SEP,
  };

  it("picks the highest-scoring start/end pair", () => {
    const choice = selectAnswerSpan({
      ...base,
      startLogits: peak(6),
      endLogits: peak(7),
    })!;
    expect(choice.startToken).toBe(6);
    expect(choice.endToken).toBe(7);
    expect(choice.score).toBeGreaterThan(0.9);
  });

  // The masking is the whole correctness surface: an unmasked question token
  // produces an answer that is a slice of the *question*, which reads as the
  // model misunderstanding rather than as a bug.
  it("never answers from the question half", () => {
    const choice = selectAnswerSpan({
      ...base,
      // Every logit favours a question token.
      startLogits: peak(2),
      endLogits: peak(3),
    })!;
    expect(choice.startToken).toBeGreaterThan(5);
    expect(choice.endToken).toBeGreaterThan(5);
  });

  it("never answers with a separator or a pad", () => {
    const padded = [...IDS, PAD, PAD];
    const choice = selectAnswerSpan({
      ...base,
      ids: padded,
      mask: padded.map((_, i) => (i < IDS.length ? 1 : 0)),
      startLogits: peak(10, padded.length),
      endLogits: peak(11, padded.length),
    })!;
    expect(padded[choice.startToken]).not.toBe(SEP);
    expect(padded[choice.endToken]).not.toBe(PAD);
  });

  it("never returns a span that ends before it starts", () => {
    const choice = selectAnswerSpan({
      ...base,
      startLogits: peak(9),
      endLogits: peak(6),
    })!;
    expect(choice.endToken).toBeGreaterThanOrEqual(choice.startToken);
  });

  // The CLS position is left in the softmax denominator and only then scored
  // zero — matching the pipeline. Masking it instead would rescale every
  // probability the page prints.
  it("scores CLS zero but lets it shrink the other probabilities", () => {
    const withoutCls = selectAnswerSpan({
      ...base,
      startLogits: peak(6),
      endLogits: peak(6),
    })!;
    const withHotCls = selectAnswerSpan({
      ...base,
      startLogits: peak(6).map((v, i) => (i === 0 ? 10 : v)),
      endLogits: peak(6).map((v, i) => (i === 0 ? 10 : v)),
    })!;
    // Same answer either way — CLS can never be it.
    expect(withHotCls.startToken).toBe(6);
    // But a confident CLS takes probability mass from it.
    expect(withHotCls.score).toBeLessThan(withoutCls.score);
  });

  // Transformers.js caps nothing, and neither does this. A cap would silently
  // change the answer on exactly the unsure questions this page is about.
  it("will return a long span when that is what the logits say", () => {
    const choice = selectAnswerSpan({
      ...base,
      startLogits: peak(6),
      endLogits: peak(9),
    })!;
    expect(choice.endToken - choice.startToken).toBe(3);
  });

  it("returns null when there is no separator to answer after", () => {
    expect(
      selectAnswerSpan({
        ...base,
        ids: [CLS, 10, 11],
        mask: [1, 1, 1],
        startLogits: peak(1, 3),
        endLogits: peak(1, 3),
      }),
    ).toBeNull();
  });

  it("returns null when the context side is empty", () => {
    const ids = [CLS, 10, SEP, SEP];
    expect(
      selectAnswerSpan({
        ...base,
        ids,
        mask: [1, 1, 1, 1],
        startLogits: peak(1, 4),
        endLogits: peak(1, 4),
      }),
    ).toBeNull();
  });
});
