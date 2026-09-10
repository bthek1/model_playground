import { describe, expect, it } from "vitest";

import { pooledVerdict, slidingMean } from "./pool";

describe("slidingMean", () => {
  it("leaves a constant series where it was", () => {
    const series = [
      [0.7, 0.3],
      [0.7, 0.3],
      [0.7, 0.3],
    ];
    // Close, not equal: averaging three copies of 0.7 in binary floating point
    // is 0.7000000000000001, and a test that demands exactness here is testing
    // IEEE 754 rather than the pooling.
    for (const [f, row] of slidingMean(series, 3).entries()) {
      expect(row[0], `frame ${f}`).toBeCloseTo(0.7, 12);
      expect(row[1], `frame ${f}`).toBeCloseTo(0.3, 12);
    }
  });

  it("is the identity at a window of one", () => {
    const series = [[0.1], [0.9], [0.2]];
    expect(slidingMean(series, 1)).toEqual(series);
    // …and a copy, so the caller cannot mutate the source through it.
    expect(slidingMean(series, 1)[0]).not.toBe(series[0]);
  });

  it("attenuates a single-frame spike, and more so at a wider window", () => {
    // The whole reason the control exists: a frame-level model produces spiky
    // scores, and pooling is the honest way to ask what the *clip* looks like.
    const series = [[0], [0], [1], [0], [0]];
    const narrow = slidingMean(series, 3)[2][0];
    const wide = slidingMean(series, 5)[2][0];

    expect(narrow).toBeCloseTo(1 / 3, 6);
    expect(wide).toBeCloseTo(1 / 5, 6);
    expect(wide).toBeLessThan(narrow);
  });

  it("clamps the window at the clip's ends rather than padding with zeros", () => {
    // Padding would drag the first and last frames toward nothing and put a dip
    // at both ends of every chart — which reads as the model losing confidence
    // at the start and end of every clip.
    const series = [[1], [1], [1]];
    const out = slidingMean(series, 5);
    expect(out[0][0]).toBeCloseTo(1, 6);
    expect(out[2][0]).toBeCloseTo(1, 6);
  });

  it("centres an even window by taking the extra frame from the past", () => {
    const series = [[0], [4], [0], [0]];
    // Window 2 at frame 1 averages frames 0 and 1.
    expect(slidingMean(series, 2)[1][0]).toBeCloseTo(2, 6);
  });

  it("pools each label independently", () => {
    const out = slidingMean(
      [
        [1, 0],
        [0, 1],
      ],
      2,
    );
    expect(out[1]).toEqual([0.5, 0.5]);
  });

  it("returns nothing for an empty series, and survives a silly window", () => {
    expect(slidingMean([], 5)).toEqual([]);
    expect(slidingMean([[1]], 0)).toEqual([[1]]);
    expect(slidingMean([[1]], -3)).toEqual([[1]]);
  });
});

describe("pooledVerdict", () => {
  it("picks the label with the highest mean over the whole clip", () => {
    const { index, score, means } = pooledVerdict([
      [0.2, 0.8],
      [0.8, 0.2],
      [0.8, 0.2],
    ]);
    expect(index).toBe(0);
    expect(score).toBeCloseTo(0.6, 6);
    expect(means[1]).toBeCloseTo(0.4, 6);
  });

  it("keeps the first label on a tie, so the verdict does not flicker", () => {
    expect(pooledVerdict([[0.5, 0.5]]).index).toBe(0);
  });

  it("can disagree with the loudest single frame", () => {
    // Which is exactly what pooling is for: one spectacular frame is not a
    // clip-level answer.
    const { index } = pooledVerdict([
      [0.05, 0.95], // one spectacular frame for label 1…
      [0.7, 0.3],
      [0.7, 0.3],
      [0.7, 0.3], // …and three quieter ones for label 0, which win the clip
    ]);
    expect(index).toBe(0);
  });

  it("returns a null verdict for an empty clip rather than throwing", () => {
    expect(pooledVerdict([])).toEqual({ index: -1, score: 0, means: [] });
    expect(pooledVerdict([[]])).toEqual({ index: -1, score: 0, means: [] });
  });
});
