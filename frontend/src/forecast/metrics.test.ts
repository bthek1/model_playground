import { describe, expect, it } from "vitest";

import { forecast } from "./baselines";
import { forecastMetrics, naiveScale, spread } from "./metrics";

describe("naiveScale", () => {
  it("is the mean absolute one-step difference over the history", () => {
    expect(naiveScale(Float32Array.from([1, 2, 4, 7]))).toBeCloseTo((1 + 2 + 3) / 3);
  });

  it("uses the season when asked, which is a different number", () => {
    const y = Float32Array.from([1, 5, 1, 5, 1, 5]);
    expect(naiveScale(y, 1)).toBeCloseTo(4);
    // Seasonal differences on a perfectly periodic series are zero — and a zero
    // denominator must not become an Infinity on the page.
    expect(naiveScale(y, 2)).toBeNull();
  });

  it("is null on a history too short or too flat to scale by", () => {
    expect(naiveScale(Float32Array.from([1]))).toBeNull();
    expect(naiveScale(Float32Array.from([3, 3, 3]))).toBeNull();
  });
});

describe("forecastMetrics", () => {
  it("gives a naive forecast MASE 1.0 at horizon 1 on a constant-increment series", () => {
    // The property that makes the number readable — and the one a mis-scaled
    // denominator breaks silently, because a wrong MASE is still a small number
    // near 1. Steps are all 1, so the in-sample naive scale is 1; a one-step
    // naive forecast is out by exactly 1.
    const y = Float32Array.from([1, 2, 3, 4, 5, 6]);
    const history = y.subarray(0, 5);
    const predicted = forecast("naive", Float32Array.from(history), 1);
    const m = forecastMetrics(
      Float32Array.from(y.subarray(5, 6)),
      predicted,
      Float32Array.from(history),
    );
    expect(m.mase).toBeCloseTo(1, 6);
  });

  it("gives a perfect forecast MASE 0 and a doubly-bad one MASE 2", () => {
    const history = Float32Array.from([1, 2, 3, 4, 5]);
    const actual = Float32Array.from([6]);
    expect(forecastMetrics(actual, Float32Array.from([6]), history).mase).toBeCloseTo(0);
    expect(forecastMetrics(actual, Float32Array.from([4]), history).mase).toBeCloseTo(2);
  });

  it("reports MAPE as null rather than Infinity when an actual is zero", () => {
    // Skipping the row quietly is worse than null: the average would then
    // describe a different test set from every other number in the object.
    const m = forecastMetrics(
      Float32Array.from([0, 10]),
      Float32Array.from([1, 11]),
      Float32Array.from([1, 2, 3]),
    );
    expect(m.mape).toBeNull();
    expect(m.mapeSkipped).toBe(1);
    expect(Number.isFinite(m.mae)).toBe(true);
  });

  it("computes MAPE as a fraction when every actual is non-zero", () => {
    const m = forecastMetrics(
      Float32Array.from([100, 200]),
      Float32Array.from([110, 180]),
      Float32Array.from([1, 2, 3]),
    );
    expect(m.mape).toBeCloseTo((0.1 + 0.1) / 2);
  });

  it("returns MASE null instead of Infinity on a flat history", () => {
    const m = forecastMetrics(
      Float32Array.from([1]),
      Float32Array.from([2]),
      Float32Array.from([5, 5, 5]),
    );
    expect(m.mase).toBeNull();
  });
});

describe("spread", () => {
  it("summarises a set of window metrics", () => {
    const s = spread([3, 1, 2, 10]);
    expect(s.min).toBe(1);
    expect(s.max).toBe(10);
    expect(s.mean).toBeCloseTo(4);
    expect(s.median).toBe(3);
  });

  it("is zeroed rather than NaN on no windows", () => {
    expect(spread([])).toEqual({ mean: 0, min: 0, max: 0, median: 0 });
  });
});
