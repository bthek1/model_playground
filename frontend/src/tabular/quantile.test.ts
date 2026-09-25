import { describe, expect, it } from "vitest";

import { mulberry32 } from "@/lib/random";
import { cpuMatmul, type MatmulFn } from "@/webgpu/linearModel";

import { bandCoverage, fitQuantiles, pinball, predictQuantiles } from "./quantile";

const cpu: MatmulFn = async (a, b, m, k, n) => cpuMatmul(a, b, m, k, n);

/** A target with a long right tail, so its median and its mean differ. */
function skewed(n: number, seed = 13) {
  const rand = mulberry32(seed);
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = rand() * 2 - 1;
    x[i] = a;
    // Exponential noise: mean 1, median ln 2 ≈ 0.69.
    y[i] = 2 * a + -Math.log(1 - rand());
  }
  return { x, y };
}

describe("pinball", () => {
  it("is asymmetric, and symmetric at 0.5", () => {
    expect(pinball(2, 0.9)).toBeCloseTo(1.8);
    expect(pinball(-2, 0.9)).toBeCloseTo(0.2);
    expect(pinball(2, 0.5)).toBeCloseTo(pinball(-2, 0.5));
  });
});

describe("fitQuantiles", () => {
  it("puts the 0.5 line nearer the median than the mean on a skewed target", async () => {
    // The reason quantile regression is on the ladder rather than being a
    // presentation choice: least squares fits the conditional *mean*, and on a
    // skewed target that is not the line people read off the chart.
    const n = 1500;
    const { x, y } = skewed(n);
    const model = await fitQuantiles(cpu, x, y, n, 1, {
      quantiles: [0.1, 0.5, 0.9],
      epochs: 200,
      learningRate: 0.2,
      batchSize: 128,
      seed: 3,
    });
    const preds = predictQuantiles(model, x, n);
    // Residuals about the median line should be balanced: half above, half below.
    let above = 0;
    for (let i = 0; i < n; i++) if (y[i] > preds[n + i]) above++;
    expect(above / n).toBeGreaterThan(0.42);
    expect(above / n).toBeLessThan(0.58);
  });

  it("brackets roughly the advertised fraction of held-out rows", async () => {
    // **The coverage, not merely that a band was drawn.** A band of the wrong
    // width is the failure available here and it looks entirely correct.
    const n = 2000;
    const { x, y } = skewed(n, 27);
    const model = await fitQuantiles(cpu, x, y, n, 1, {
      quantiles: [0.1, 0.5, 0.9],
      epochs: 250,
      learningRate: 0.2,
      batchSize: 128,
      seed: 7,
    });
    const preds = predictQuantiles(model, x, n);
    const coverage = bandCoverage(preds, y, n, model.quantiles);
    expect(coverage).toBeGreaterThan(0.7);
    expect(coverage).toBeLessThan(0.9);
  });

  it("orders the lines per row, so a crossing cannot render inside out", async () => {
    const n = 300;
    const { x, y } = skewed(n, 31);
    const model = await fitQuantiles(cpu, x, y, n, 1, {
      quantiles: [0.1, 0.5, 0.9],
      epochs: 20,
      learningRate: 0.05,
      batchSize: 64,
      seed: 5,
    });
    const preds = predictQuantiles(model, x, n);
    for (let i = 0; i < n; i++) {
      expect(preds[i]).toBeLessThanOrEqual(preds[n + i]);
      expect(preds[n + i]).toBeLessThanOrEqual(preds[2 * n + i]);
    }
  });

  it("stops when asked", async () => {
    const n = 100;
    const { x, y } = skewed(n);
    const seen: number[] = [];
    await fitQuantiles(cpu, x, y, n, 1, {
      quantiles: [0.5],
      epochs: 50,
      learningRate: 0.1,
      batchSize: 32,
      seed: 1,
      onProgress: (done) => seen.push(done),
      shouldStop: () => seen.length >= 4,
    });
    expect(seen.length).toBeLessThanOrEqual(5);
  });
});

describe("bandCoverage", () => {
  it("counts a row on the edge as inside", () => {
    const preds = Float32Array.from([0, 0, 1, 1, 2, 2]);
    const actual = Float32Array.from([0, 2]);
    expect(bandCoverage(preds, actual, 2, [0.1, 0.5, 0.9])).toBe(1);
  });

  it("is zero when nothing falls in the band", () => {
    const preds = Float32Array.from([0, 0, 1, 1, 2, 2]);
    const actual = Float32Array.from([9, -9]);
    expect(bandCoverage(preds, actual, 2, [0.1, 0.5, 0.9])).toBe(0);
  });
});

describe("the step size", () => {
  it("behaves the same on a target scaled by a thousand", async () => {
    // The knob has to be scale-invariant, because the target is the *user's*
    // column: a fixed step that behaves on dollars never moves the intercept on
    // a ratio, and explodes on a population count. The sub-gradient of the
    // pinball loss is ±τ whatever the residual, so the step size is the whole
    // story — there is no shrinking gradient to save a badly-scaled one.
    const n = 800;
    const { x, y } = skewed(n, 41);
    const big = Float32Array.from(y, (v) => v * 1000);
    const opts = {
      quantiles: [0.1, 0.5, 0.9],
      epochs: 150,
      learningRate: 0.2,
      batchSize: 64,
      seed: 9,
    };
    const small = await fitQuantiles(cpu, x, y, n, 1, opts);
    const large = await fitQuantiles(cpu, x, big, n, 1, opts);

    const coverSmall = bandCoverage(predictQuantiles(small, x, n), y, n, opts.quantiles);
    const coverLarge = bandCoverage(predictQuantiles(large, x, n), big, n, opts.quantiles);
    expect(Math.abs(coverSmall - coverLarge)).toBeLessThan(0.08);
    expect(coverLarge).toBeGreaterThan(0.65);
  });
});
