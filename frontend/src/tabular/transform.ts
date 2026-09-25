// The log-transform, and the units it owns.
//
// Fitting a regression on `log1p(target)` is a standard and often correct move
// on a target with a long right tail. The mistake people make constantly is what
// happens next: they read the RMSE off the logged fit, compare it with the RMSE
// of the raw fit, and conclude the transform helped. **Those two numbers are not
// comparable.** One is in log units and one is in the target's units, and the
// logged one is smaller for the same reason a number's logarithm is smaller.
//
// So this module owns the units rather than the route. The route renders what it
// is handed, which makes it structurally unable to put an RMSE in log space next
// to an RMSE in dollars without the labels saying which is which. The honest
// comparison back-transforms the predictions and re-scores **in the original
// units**, and the page shows both numbers with their units named.

export type TargetSpace = "raw" | "log";

export interface TargetTransform {
  space: TargetSpace;
  /** What the model was fitted on. */
  forward: (v: number) => number;
  /** Back to the target's own units. */
  inverse: (v: number) => number;
  /**
   * The units any metric computed in this space is in, as a phrase that can be
   * rendered directly — "log units" or the column's own name.
   */
  units: string;
}

/** `log1p` rather than `log`: a target of exactly 0 is ordinary and `log(0)` is not. */
export function makeTransform(space: TargetSpace, targetName: string): TargetTransform {
  if (space === "log") {
    return {
      space,
      forward: (v) => Math.log1p(v),
      inverse: (v) => Math.expm1(v),
      units: `log(1 + ${targetName})`,
    };
  }
  return { space, forward: (v) => v, inverse: (v) => v, units: targetName };
}

/**
 * Apply the forward transform to a target column.
 *
 * `log1p` is undefined below −1, and a target with negative values is a real
 * thing (a profit column, a temperature). Refusing with a reason is better than
 * quietly producing `NaN`s that a gradient loop then propagates through every
 * weight — the fit still "completes", and every metric reads `NaN`.
 */
export function forwardTarget(values: Float32Array, t: TargetTransform): Float32Array {
  if (t.space === "log") {
    for (let i = 0; i < values.length; i++) {
      if (values[i] <= -1) {
        throw new Error(
          "The log transform needs a target above −1 (it fits log(1 + y)). This column has values at or below −1.",
        );
      }
    }
  }
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = t.forward(values[i]);
  return out;
}

/** Back-transform predictions into the target's own units. */
export function inverseTarget(values: Float32Array, t: TargetTransform): Float32Array {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = t.inverse(values[i]);
  return out;
}
