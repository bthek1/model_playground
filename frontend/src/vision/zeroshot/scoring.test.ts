import { describe, expect, it } from "vitest";

import {
  dot,
  l2Normalize,
  normalizedRows,
  scoreImage,
  sigmoid,
  softmax,
} from "./scoring";

describe("l2Normalize", () => {
  it("scales a vector to unit length", () => {
    const out = l2Normalize(new Float32Array([3, 4]));
    expect(out[0]).toBeCloseTo(0.6, 6);
    expect(out[1]).toBeCloseTo(0.8, 6);
    expect(Math.hypot(out[0], out[1])).toBeCloseTo(1, 6);
  });

  it("leaves a zero vector alone instead of filling it with NaN", () => {
    // A zero vector has no direction. Dividing by its norm would poison every
    // score downstream with NaN, which renders as a blank table rather than an
    // error anyone would notice.
    const out = l2Normalize(new Float32Array([0, 0, 0]));
    expect(Array.from(out)).toEqual([0, 0, 0]);
  });
});

describe("normalizedRows", () => {
  it("splits a flat matrix into unit rows", () => {
    const rows = normalizedRows([3, 4, 0, 5], 2, 2);
    expect(rows).toHaveLength(2);
    expect(rows[0][0]).toBeCloseTo(0.6, 6);
    expect(rows[1][1]).toBeCloseTo(1, 6);
  });
});

describe("softmax", () => {
  it("produces a distribution that sums to one", () => {
    const p = softmax([1, 2, 3]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
    expect(p[2]).toBeGreaterThan(p[0]);
  });

  it("survives the large logits a scale of 100 produces", () => {
    // CLIP multiplies cosine similarities by ~100, so logits reach ±100 and a
    // naive `exp` overflows to Infinity — every score becomes NaN.
    const p = softmax([100, 30, -100]);
    expect(p.every(Number.isFinite)).toBe(true);
    expect(p[0]).toBeCloseTo(1, 6);
  });

  it("returns nothing for no labels rather than dividing by zero", () => {
    expect(softmax([])).toEqual([]);
  });
});

describe("sigmoid", () => {
  it("is centred at a half and saturates both ways", () => {
    expect(sigmoid(0)).toBeCloseTo(0.5, 9);
    expect(sigmoid(50)).toBeCloseTo(1, 9);
    expect(sigmoid(-50)).toBeCloseTo(0, 9);
  });

  it("does not overflow on a confidently negative logit", () => {
    // SigLIP's bias is around -13, so large negative logits are the common case.
    expect(Number.isFinite(sigmoid(-800))).toBe(true);
    expect(Number.isFinite(sigmoid(800))).toBe(true);
  });
});

describe("dot", () => {
  it("multiplies elementwise and sums", () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32);
  });
});

describe("scoreImage", () => {
  const image = l2Normalize(new Float32Array([1, 0]));
  const texts = [
    l2Normalize(new Float32Array([1, 0])), // identical -> cosine 1
    l2Normalize(new Float32Array([0, 1])), // orthogonal -> cosine 0
  ];

  it("softmaxes over the labels for a CLIP-family model", () => {
    const scores = scoreImage(image, texts, { kind: "softmax", scale: 100 });
    expect(scores.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
    // cosine 1 vs 0, scaled by 100, is a 100-logit gap: effectively certain.
    expect(scores[0]).toBeGreaterThan(0.99);
  });

  it("scores each label independently for a SigLIP-family model", () => {
    const siglip = { kind: "sigmoid" as const, scale: 117.330795, bias: -12.932437 };

    // One matching, one not: the sigmoid saturates in both directions.
    const mixed = scoreImage(image, texts, siglip);
    expect(mixed[0]).toBeGreaterThan(0.99); // 117.3 - 12.9 = +104
    expect(mixed[1]).toBeLessThan(0.01); //     0   - 12.9 =  -12.9

    // **Two labels that both fit both score high**, so the scores sum to ~2.
    // A softmax cannot do that — it would have to split one unit of probability
    // between them — and that is exactly why SigLIP's numbers mean something on
    // their own while CLIP's only mean something relative to the list.
    const bothMatch = scoreImage(image, [texts[0], l2Normalize(new Float32Array([1, 0]))], siglip);
    expect(bothMatch.reduce((a, b) => a + b, 0)).toBeCloseTo(2, 3);
  });

  it("ignores the bias for a softmax model", () => {
    // A constant added to every logit cancels in a softmax; carrying it anyway
    // would be a silent no-op that looks meaningful.
    const withBias = scoreImage(image, texts, {
      kind: "softmax",
      scale: 100,
      bias: -12.9,
    });
    const without = scoreImage(image, texts, { kind: "softmax", scale: 100 });
    expect(withBias[0]).toBeCloseTo(without[0], 12);
  });

  it("makes the scale decide the numbers, not just the ranking", () => {
    // The failure this whole module is arranged around: a wrong `logit_scale`
    // keeps every score in [0, 1] and keeps the order intact, so only the
    // magnitudes betray it. Hence the parity spec against the real pipeline.
    const right = scoreImage(image, texts, { kind: "softmax", scale: 100 });
    const wrong = scoreImage(image, texts, { kind: "softmax", scale: 1 });
    expect(wrong[0]).toBeGreaterThan(wrong[1]); // ranking survives
    expect(wrong[0]).toBeLessThan(0.8); // magnitude does not
    expect(right[0]).toBeGreaterThan(0.99);
  });
});
