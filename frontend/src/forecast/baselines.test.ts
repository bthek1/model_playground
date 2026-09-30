import { describe, expect, it } from "vitest";

import { forecast, regressionSlope } from "./baselines";

describe("naive", () => {
  it("is exact on a constant series", () => {
    const y = Float32Array.from([5, 5, 5, 5]);
    expect(Array.from(forecast("naive", y, 3))).toEqual([5, 5, 5]);
  });

  it("repeats the last value, not the mean", () => {
    const y = Float32Array.from([0, 0, 0, 10]);
    expect(Array.from(forecast("naive", y, 2))).toEqual([10, 10]);
  });
});

describe("seasonal naive", () => {
  it("repeats the last full season", () => {
    const y = Float32Array.from([1, 2, 3, 4, 5, 6]);
    expect(Array.from(forecast("seasonal-naive", y, 3, 3))).toEqual([4, 5, 6]);
  });

  it("keeps repeating past one season without running off the end", () => {
    // `history[n − m + h]` is the natural thing to write and reads past the end
    // the moment the horizon exceeds the season.
    const y = Float32Array.from([1, 2, 3, 4]);
    expect(Array.from(forecast("seasonal-naive", y, 5, 2))).toEqual([3, 4, 3, 4, 3]);
  });

  it("is wrong in a *named* way when the season is wrong", () => {
    // This is what the control exists for: a wrong season length produces a
    // confident, plausible forecast that is off by a phase, with no symptom.
    const period = [10, 20, 30, 40];
    const y = Float32Array.from([...period, ...period, ...period]);
    expect(Array.from(forecast("seasonal-naive", y, 4, 4))).toEqual(period);
    // At a season of 3 it repeats the wrong three values, in the wrong places.
    expect(Array.from(forecast("seasonal-naive", y, 4, 3))).toEqual([20, 30, 40, 20]);
  });

  it("falls back to naive when the season is longer than the history", () => {
    const y = Float32Array.from([7, 8, 9]);
    expect(Array.from(forecast("seasonal-naive", y, 2, 50))).toEqual([7, 8]);
  });
});

describe("drift", () => {
  it("uses the mean per-step change, checked against a hand-computed case", () => {
    // (last − first)/(n − 1) = (10 − 0)/4 = 2.5
    const y = Float32Array.from([0, 9, 1, 8, 10]);
    expect(Array.from(forecast("drift", y, 2))).toEqual([12.5, 15]);
  });

  it("is NOT the regression slope, and the two agree only on a straight line", () => {
    // The plausible wrong implementation. It matches on the series everyone
    // checks a drift forecast against, which is why it survives review.
    const line = Float32Array.from([1, 2, 3, 4, 5]);
    expect(regressionSlope(line)).toBeCloseTo(1, 6);
    expect(forecast("drift", line, 1)[0]).toBeCloseTo(6, 5);

    const bent = Float32Array.from([0, 9, 1, 8, 10]);
    const meanStep = (bent[4] - bent[0]) / 4;
    expect(regressionSlope(bent)).not.toBeCloseTo(meanStep, 2);
  });

  it("is flat when the series starts and ends in the same place", () => {
    const y = Float32Array.from([5, 100, -20, 5]);
    expect(Array.from(forecast("drift", y, 2))).toEqual([5, 5]);
  });
});

describe("mean", () => {
  it("averages the whole history", () => {
    const y = Float32Array.from([1, 2, 3, 4]);
    expect(Array.from(forecast("mean", y, 2))).toEqual([2.5, 2.5]);
  });
});

describe("every baseline", () => {
  it("refuses an empty history rather than returning zeros", () => {
    for (const id of ["naive", "seasonal-naive", "drift", "mean"] as const) {
      expect(() => forecast(id, new Float32Array(0), 1)).toThrow(/history/i);
    }
  });
});
