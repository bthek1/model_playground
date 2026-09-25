import { describe, expect, it } from "vitest";

import { regressionMetrics } from "./metrics";
import { forwardTarget, inverseTarget, makeTransform } from "./transform";

describe("makeTransform", () => {
  it("round-trips to tolerance", () => {
    const t = makeTransform("log", "price");
    for (const v of [0, 0.5, 1, 10, 1e4]) {
      expect(t.inverse(t.forward(v))).toBeCloseTo(v, 6);
    }
  });

  it("names the units, so the route cannot mislabel them", () => {
    expect(makeTransform("raw", "price").units).toBe("price");
    expect(makeTransform("log", "price").units).toBe("log(1 + price)");
  });

  it("refuses a target the transform is undefined on, with a reason", () => {
    // `log1p(-2)` is NaN, and a gradient loop propagates NaN through every
    // weight while the fit still "completes" and every metric reads NaN.
    const t = makeTransform("log", "profit");
    expect(() => forwardTarget(Float32Array.from([1, -2]), t)).toThrow(/above −1/);
  });

  it("leaves a raw target alone", () => {
    const t = makeTransform("raw", "y");
    const values = Float32Array.from([1, 2, 3]);
    expect(Array.from(forwardTarget(values, t))).toEqual([1, 2, 3]);
    expect(Array.from(inverseTarget(values, t))).toEqual([1, 2, 3]);
  });
});

describe("the incomparability the page exists to demonstrate", () => {
  it("gives a different RMSE in log space from the same errors in target units", () => {
    // This is the page's claim, so it is a test. A log-space RMSE is smaller for
    // the same reason a logarithm is smaller, and reading the two side by side
    // as if they measured the same thing is the mistake people make constantly.
    const t = makeTransform("log", "price");
    const actual = Float32Array.from([10, 100, 1000, 5000]);
    const predicted = Float32Array.from([12, 90, 1300, 4200]);
    const train = Float32Array.from([10, 100, 1000, 5000]);

    const inUnits = regressionMetrics(actual, predicted, train, t.units);
    const inLog = regressionMetrics(
      forwardTarget(actual, t),
      forwardTarget(predicted, t),
      forwardTarget(train, t),
      t.units,
    );

    expect(inLog.rmse).toBeLessThan(inUnits.rmse / 100);
    // And each carries the units it was computed in, so the two can never be
    // rendered side by side unlabelled.
    expect(inUnits.units).toBe("log(1 + price)");
  });
});

describe("regressionMetrics", () => {
  it("carries the train-mean baseline in the same units as the score", () => {
    const actual = Float32Array.from([10, 20, 30]);
    const predicted = Float32Array.from([11, 19, 31]);
    const train = Float32Array.from([0, 0, 0]);
    const m = regressionMetrics(actual, predicted, train, "y");
    expect(m.rmse).toBeCloseTo(1);
    expect(m.mae).toBeCloseTo(1);
    // Predicting the training mean of 0 is terrible here, and the number says so
    // in the target's own units rather than as an abstract R².
    expect(m.baselineRmse).toBeGreaterThan(m.rmse * 10);
    expect(m.units).toBe("y");
  });

  it("gives a model that predicts the test mean an R² of zero", () => {
    const actual = Float32Array.from([1, 2, 3, 4]);
    const predicted = Float32Array.from([2.5, 2.5, 2.5, 2.5]);
    const m = regressionMetrics(actual, predicted, actual, "y");
    expect(m.r2).toBeCloseTo(0, 6);
  });

  it("does not return NaN on a constant target", () => {
    const actual = Float32Array.from([5, 5, 5]);
    const m = regressionMetrics(actual, actual, actual, "y");
    expect(m.r2).toBe(1);
    expect(regressionMetrics(actual, Float32Array.from([1, 1, 1]), actual, "y").r2).toBe(0);
  });
});
