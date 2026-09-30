// Ridge regression in closed form: `(XᵀX + λI) w = Xᵀy`.
//
// This file is where the category's GPU/CPU split stops being rhetoric and
// becomes arithmetic. Building `XᵀX` and `Xᵀy` is `O(n·d²)` and `O(n·d)` in the
// **row count**, which is the large number — so both go through the same
// `MatmulFn` seam `webgpu/linearModel.ts` uses, and run as WGSL on the GPU.
// Factorising the resulting `d×d` matrix is microseconds with `d` in the tens,
// and stays on the CPU. Dispatching a kernel to invert a matrix smaller than one
// workgroup is the trees lesson pointing the other way.
//
// **And it is a solve, not an inverse.** `(XᵀX + λI)` is symmetric positive
// definite for any `λ > 0`, which is exactly what ridge's penalty buys — so a
// Cholesky factorisation is both the fastest route and the stable one, where the
// unregularised normal equations are neither. That is the real reason ridge is
// this category's linear model rather than plain least squares, and the page
// says so rather than leaving it as an implementation detail.

import type { MatmulFn } from "@/webgpu/linearModel";

export interface RidgeResult {
  /** One weight per design column, plus the intercept in `intercept`. */
  weights: Float32Array;
  intercept: number;
  /**
   * True when the design is (numerically) rank-deficient — two columns carrying
   * the same information, or more columns than rows. At `λ = 0` the
   * factorisation cannot complete on such a design; reporting that beats
   * returning the plausible-looking garbage an inverse would produce.
   */
  rankDeficient: boolean;
  /**
   * `max(diag)/min(diag)` of the Cholesky factor, squared — a cheap stand-in for
   * the condition number of the normal equations. Large means the coefficients
   * are unstable even where they are computable, which is worth saying next to
   * them.
   */
  condition: number;
  /** The λ that was actually used, after any fallback. */
  lambda: number;
}

/**
 * In-place Cholesky. Returns false the moment a pivot is non-positive, which is
 * what "not positive definite" looks like from inside the algorithm.
 */
export function cholesky(a: Float64Array, d: number): boolean {
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = a[i * d + j];
      for (let k = 0; k < j; k++) sum -= a[i * d + k] * a[j * d + k];
      if (i === j) {
        if (!(sum > 1e-10)) return false;
        a[i * d + i] = Math.sqrt(sum);
      } else {
        a[i * d + j] = sum / a[j * d + j];
      }
    }
    // The strict upper triangle is not part of the factor; zero it so the
    // result is a lower-triangular matrix rather than a half-overwritten one.
    for (let j = i + 1; j < d; j++) a[i * d + j] = 0;
  }
  return true;
}

/** Solve `L Lᵀ w = b` given the lower-triangular factor `L`, in place of `b`. */
export function choleskySolve(l: Float64Array, b: Float64Array, d: number): Float64Array {
  const w = Float64Array.from(b);
  // Forward substitution: L z = b.
  for (let i = 0; i < d; i++) {
    let sum = w[i];
    for (let k = 0; k < i; k++) sum -= l[i * d + k] * w[k];
    w[i] = sum / l[i * d + i];
  }
  // Back substitution: Lᵀ w = z.
  for (let i = d - 1; i >= 0; i--) {
    let sum = w[i];
    for (let k = i + 1; k < d; k++) sum -= l[k * d + i] * w[k];
    w[i] = sum / l[i * d + i];
  }
  return w;
}

/**
 * Fit ridge regression on a design matrix.
 *
 * The intercept is fitted as an extra column of ones and is **not penalised**:
 * shrinking it toward zero would shrink the predictions toward zero rather than
 * toward the mean, which is a different (and wrong) model.
 */
export async function fitRidge(
  matmul: MatmulFn,
  x: Float32Array,
  y: Float32Array,
  rows: number,
  features: number,
  lambda: number,
): Promise<RidgeResult> {
  const d = features + 1;

  // Augment with the intercept column and transpose once. The transpose is what
  // lets both products be a single matmul: `Xᵀ·X` is (d×n)·(n×d) and `Xᵀ·y` is
  // (d×n)·(n×1).
  const xa = new Float32Array(rows * d);
  for (let i = 0; i < rows; i++) {
    xa.set(x.subarray(i * features, (i + 1) * features), i * d);
    xa[i * d + features] = 1;
  }
  const xt = new Float32Array(d * rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < d; j++) xt[j * rows + i] = xa[i * d + j];
  }

  const gram = await matmul(xt, xa, d, rows, d);
  const xty = await matmul(xt, y, d, rows, 1);

  // Float64 from here: the factorisation is where precision is actually spent,
  // and it is `d×d` with `d` in the tens, so it costs nothing to do properly.
  const a = new Float64Array(d * d);
  for (let i = 0; i < d * d; i++) a[i] = gram[i];
  const b = new Float64Array(d);
  for (let i = 0; i < d; i++) b[i] = xty[i];

  let used = lambda;
  let rankDeficient = false;
  let factor = penalised(a, d, used, features);
  if (!cholesky(factor, d)) {
    // Only reachable at λ ≈ 0 on a design that is exactly collinear. The page
    // says so; silently substituting a pseudo-inverse would hand back
    // coefficients that look like a model and are one of infinitely many.
    rankDeficient = true;
    used = Math.max(lambda, 1e-6 * trace(a, d) / d);
    factor = penalised(a, d, used, features);
    if (!cholesky(factor, d)) {
      throw new Error(
        "The design matrix is singular even with a penalty — two feature columns are identical, or there are more columns than rows.",
      );
    }
  }

  const solved = choleskySolve(factor, b, d);
  let min = Infinity;
  let max = 0;
  for (let i = 0; i < d; i++) {
    const v = factor[i * d + i];
    min = Math.min(min, v);
    max = Math.max(max, v);
  }

  const weights = new Float32Array(features);
  for (let i = 0; i < features; i++) weights[i] = solved[i];
  return {
    weights,
    intercept: solved[features],
    rankDeficient,
    condition: min > 0 ? (max / min) ** 2 : Infinity,
    lambda: used,
  };
}

/** `A + λI`, with the intercept's diagonal entry left alone. */
function penalised(a: Float64Array, d: number, lambda: number, features: number): Float64Array {
  const out = Float64Array.from(a);
  for (let i = 0; i < features; i++) out[i * d + i] += lambda;
  // A tiny floor on the intercept's own pivot: with a design of all-zero
  // columns the ones column alone is still fine, but a degenerate frame would
  // otherwise fail on a pivot that is only zero by rounding.
  out[features * d + features] += 1e-12;
  return out;
}

function trace(a: Float64Array, d: number): number {
  let sum = 0;
  for (let i = 0; i < d; i++) sum += a[i * d + i];
  return sum;
}

/** Predictions for a design matrix, given a fitted ridge model. */
export function predictRidge(
  model: Pick<RidgeResult, "weights" | "intercept">,
  x: Float32Array,
  rows: number,
  features: number,
): Float32Array {
  const out = new Float32Array(rows);
  for (let i = 0; i < rows; i++) {
    let sum = model.intercept;
    for (let j = 0; j < features; j++) sum += model.weights[j] * x[i * features + j];
    out[i] = sum;
  }
  return out;
}
