import { describe, expect, it } from "vitest";

import {
  argmaxRow,
  classificationMetrics,
  confusionMatrix,
  metricsAtThreshold,
  predictedClasses,
} from "./metrics";

describe("confusionMatrix", () => {
  it("sums to the row count", () => {
    const actual = new Uint8Array([0, 1, 1, 2, 0]);
    const predicted = new Uint8Array([0, 1, 2, 2, 1]);
    const m = confusionMatrix(actual, predicted, ["a", "b", "c"]);
    let total = 0;
    for (const n of m.counts) total += n;
    expect(total).toBe(5);
    expect(m.counts[0 * 3 + 0]).toBe(1);
    expect(m.counts[1 * 3 + 2]).toBe(1);
  });
});

describe("classificationMetrics", () => {
  it("carries the majority baseline from the TRAINING half", () => {
    // The baseline travels inside the metrics object so the two can never come
    // from different splits, and it is taken from the training labels so it is
    // not given a look at the answers.
    const actual = new Uint8Array([0, 0, 0, 1]);
    const predicted = new Uint8Array([0, 0, 1, 1]);
    const train = new Uint8Array([0, 0, 0, 0, 1]);
    const m = classificationMetrics(actual, predicted, ["no", "yes"], train);
    expect(m.accuracy).toBeCloseTo(0.75);
    expect(m.baselineLabel).toBe("no");
    expect(m.baselineAccuracy).toBeCloseTo(0.75);
  });

  it("scores a perfect classifier at one", () => {
    const y = new Uint8Array([0, 1, 2, 1]);
    const m = classificationMetrics(y, y, ["a", "b", "c"], y);
    expect(m.accuracy).toBe(1);
    expect(m.precision).toBeCloseTo(1);
    expect(m.recall).toBeCloseTo(1);
    expect(m.f1).toBeCloseTo(1);
  });

  it("macro-averages over the classes present, not over every label", () => {
    // A label with no held-out rows counted as a zero would drag every macro
    // score down by a factor that has nothing to do with the model.
    const actual = new Uint8Array([0, 0, 1, 1]);
    const predicted = new Uint8Array([0, 0, 1, 1]);
    const m = classificationMetrics(actual, predicted, ["a", "b", "never"], actual);
    expect(m.recall).toBeCloseTo(1);
  });
});

describe("predictedClasses", () => {
  it("uses the threshold on a binary problem and argmax otherwise", () => {
    const binary = Float32Array.from([0.7, 0.3, 0.4, 0.6]);
    expect(Array.from(predictedClasses(binary, 2, 2, 0.5))).toEqual([0, 1]);
    expect(Array.from(predictedClasses(binary, 2, 2, 0.7))).toEqual([0, 0]);
    const three = Float32Array.from([0.2, 0.5, 0.3]);
    expect(Array.from(predictedClasses(three, 1, 3, 0.9))).toEqual([1]);
    expect(argmaxRow(three, 0, 3)).toBe(1);
  });
});

describe("metricsAtThreshold", () => {
  it("moves recall and precision in opposite directions as it falls", () => {
    // The whole justification for the slider: 0.5 is only right when a false
    // positive and a false negative cost the same.
    const probs = Float32Array.from([
      0.9, 0.1, //
      0.6, 0.4,
      0.45, 0.55,
      0.2, 0.8,
    ]);
    const actual = new Uint8Array([0, 1, 1, 1]);
    const train = new Uint8Array([0, 0, 1, 1]);
    const strict = metricsAtThreshold(probs, actual, ["no", "yes"], train, 0.9);
    const loose = metricsAtThreshold(probs, actual, ["no", "yes"], train, 0.3);
    expect(loose.recall).toBeGreaterThan(strict.recall);
    expect(loose.confusion.counts[0 * 2 + 1]).toBeGreaterThanOrEqual(
      strict.confusion.counts[0 * 2 + 1],
    );
  });

  it("keeps the same baseline whatever the threshold", () => {
    const probs = Float32Array.from([0.9, 0.1, 0.2, 0.8]);
    const actual = new Uint8Array([0, 1]);
    const train = new Uint8Array([0, 0, 0, 1]);
    const a = metricsAtThreshold(probs, actual, ["no", "yes"], train, 0.2);
    const b = metricsAtThreshold(probs, actual, ["no", "yes"], train, 0.8);
    expect(a.baselineAccuracy).toBe(b.baselineAccuracy);
    expect(a.baselineLabel).toBe(b.baselineLabel);
  });
});
