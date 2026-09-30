// Quantile regression: fit a *band*, not a number.
//
// A point estimate is a claim the model cannot support. A prediction interval is
// both more honest and better to look at — and getting one costs a single line
// of difference from the squared loss, because the pinball loss has the same
// shape: a per-row residual, a per-row weight, and a gradient that is that
// weight's sign.
//
//   pinball_τ(r) = τ·r          when r ≥ 0   (under-predicted)
//                = (τ − 1)·r    when r < 0   (over-predicted)
//
// At τ = 0.5 that is the absolute error (halved), whose minimiser is the median
// rather than the mean — which is the point: an asymmetric target's median is
// not its mean, and the 0.5 line is the one people expect the model to draw.
//
// Three quantiles are fitted in **one pass over one design matrix**, sharing the
// gradient loop, because three separate fits over the same rows is three times
// the arithmetic for the same numbers.

import { mulberry32, shuffle } from "@/lib/random";
import type { MatmulFn } from "@/webgpu/linearModel";

export interface QuantileOptions {
  /** Ascending. Typically `[0.1, 0.5, 0.9]`. */
  quantiles: number[];
  epochs: number;
  learningRate: number;
  batchSize: number;
  seed: number;
  onProgress?: (done: number, total: number, loss: number | null) => void;
  shouldStop?: () => boolean;
}

export interface QuantileModel {
  quantiles: number[];
  /** `quantiles.length × features`, row-major. */
  weights: Float32Array;
  intercepts: Float32Array;
  features: number;
}

/** The pinball loss of one residual at one quantile. Exported for the tests. */
export function pinball(residual: number, tau: number): number {
  return residual >= 0 ? tau * residual : (tau - 1) * residual;
}

/**
 * Fit one linear model per quantile by sub-gradient descent.
 *
 * The forward pass is a single matmul through the injected seam — the design
 * matrix against a `features × quantiles` weight block — so all three quantiles
 * cost one GPU dispatch per batch rather than three.
 */
export async function fitQuantiles(
  matmul: MatmulFn,
  x: Float32Array,
  y: Float32Array,
  rows: number,
  features: number,
  options: QuantileOptions,
): Promise<QuantileModel> {
  const q = options.quantiles.length;
  const weights = new Float32Array(features * q);
  const intercepts = new Float32Array(q);
  // Start each quantile at the target's mean, so the first steps refine a
  // sensible level rather than travelling to it from zero.
  let mean = 0;
  for (let i = 0; i < rows; i++) mean += y[i];
  mean /= Math.max(1, rows);
  intercepts.fill(mean);

  // **The step size is a fraction of the target's own spread, and it decays.**
  //
  // The pinball loss's sub-gradient does not vanish at the optimum — it is ±τ or
  // ±(1 − τ) whatever the residual — so a constant step size does not converge,
  // it orbits the answer at a radius proportional to the step. Two things are
  // needed and neither is optional: the step is in the *target's* units, so a
  // learning rate that behaves on a target ranging over thousands is enormous on
  // one ranging over 0.9; and 1/√epoch is the standard sub-gradient schedule,
  // which is what closes the orbit. `learningRate` is therefore dimensionless.
  let variance = 0;
  for (let i = 0; i < rows; i++) variance += (y[i] - mean) * (y[i] - mean);
  const targetScale = Math.max(1e-6, Math.sqrt(variance / Math.max(1, rows)));

  const rand = mulberry32(options.seed ^ 0x2f1d);
  const B = Math.min(options.batchSize, Math.max(1, rows));
  const indices = new Int32Array(rows);
  for (let i = 0; i < rows; i++) indices[i] = i;
  const xb = new Float32Array(B * features);
  const yb = new Float32Array(B);
  const grad = new Float32Array(features * q);

  for (let epoch = 0; epoch < options.epochs; epoch++) {
    if (options.shouldStop?.()) break;
    shuffle(indices, rand);
    let total = 0;
    let seen = 0;
    for (let start = 0; start < rows; start += B) {
      if (options.shouldStop?.()) break;
      const cur = Math.min(B, rows - start);
      for (let i = 0; i < cur; i++) {
        const src = indices[start + i] * features;
        xb.set(x.subarray(src, src + features), i * features);
        yb[i] = y[indices[start + i]];
      }
      const preds = await matmul(xb.subarray(0, cur * features), weights, cur, features, q);

      grad.fill(0);
      const gInt = new Float32Array(q);
      for (let i = 0; i < cur; i++) {
        for (let k = 0; k < q; k++) {
          const tau = options.quantiles[k];
          const residual = yb[i] - (preds[i * q + k] + intercepts[k]);
          total += pinball(residual, tau);
          // d/dŷ of the pinball loss is −τ above the line and (1 − τ) below it.
          // Exactly at zero either sub-gradient is valid; the sign convention
          // below picks the upper one, which is what makes τ = 1 behave.
          const slope = residual >= 0 ? -tau : 1 - tau;
          gInt[k] += slope / cur;
          for (let j = 0; j < features; j++) {
            grad[j * q + k] += (slope * xb[i * features + j]) / cur;
          }
        }
      }
      seen += cur;
      const lr = (options.learningRate * targetScale) / Math.sqrt(1 + epoch);
      for (let i = 0; i < grad.length; i++) weights[i] -= lr * grad[i];
      for (let k = 0; k < q; k++) intercepts[k] -= lr * gInt[k];
    }
    options.onProgress?.(epoch + 1, options.epochs, seen ? total / (seen * q) : null);
  }

  return { quantiles: options.quantiles, weights, intercepts, features };
}

/**
 * Predict every quantile for `rows` rows — `quantiles.length × rows`, one
 * contiguous row per quantile.
 *
 * The quantile lines are fitted independently, so nothing stops a low quantile
 * crossing a high one on some rows (the "quantile crossing" problem). They are
 * sorted per row here, which is the standard and honest repair: it changes no
 * line's *level*, only the labelling of which is which where they have crossed,
 * and an interval whose lower edge is above its upper edge is not renderable.
 */
export function predictQuantiles(
  model: QuantileModel,
  x: Float32Array,
  rows: number,
): Float32Array {
  const q = model.quantiles.length;
  const out = new Float32Array(q * rows);
  const row = new Float32Array(q);
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < q; k++) {
      let sum = model.intercepts[k];
      for (let j = 0; j < model.features; j++) {
        sum += model.weights[j * q + k] * x[i * model.features + j];
      }
      row[k] = sum;
    }
    row.sort();
    for (let k = 0; k < q; k++) out[k * rows + i] = row[k];
  }
  return out;
}

/**
 * Fraction of held-out rows falling inside the outer band.
 *
 * The number the page must show, and the one a test must assert: a band of the
 * wrong width looks entirely correct on screen. A 0.1–0.9 band should cover
 * about 80% of held-out rows; covering 99% means it is uselessly wide and
 * covering 40% means it is a decoration.
 */
export function bandCoverage(
  predictions: Float32Array,
  actuals: Float32Array,
  rows: number,
  quantiles: number[],
): number {
  if (rows === 0 || quantiles.length < 2) return 0;
  const lo = 0;
  const hi = quantiles.length - 1;
  let inside = 0;
  for (let i = 0; i < rows; i++) {
    const low = predictions[lo * rows + i];
    const high = predictions[hi * rows + i];
    if (actuals[i] >= low && actuals[i] <= high) inside++;
  }
  return inside / rows;
}
