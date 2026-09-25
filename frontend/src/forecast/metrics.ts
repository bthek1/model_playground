// Forecast error, and the one scaling that makes it readable.
//
// MAE and RMSE are in the series' own units, which makes them meaningless
// across series and nearly meaningless on their own — is an MAE of 42 good?
// MAPE fixes the units and introduces a hole: it divides by the actual, so one
// zero in the target makes it infinite and a near-zero makes it enormous. And
// **MASE** fixes both: it divides by the in-sample naive error, so 1.0 means
// "no better than repeating the last value" and everything below 1 is a real
// improvement over the thing you have to beat.
//
// That denominator is the whole value of the number and it fails silently: a
// MASE scaled by the wrong thing still looks like a small number near 1.

export interface ForecastMetrics {
  mae: number;
  rmse: number;
  /**
   * Mean absolute percentage error, 0–1 (not percent). `null` when any actual
   * is zero — which is honest, where `Infinity` renders as a broken chart and a
   * silently-skipped row makes the number describe a different test set.
   */
  mape: number | null;
  /** How many rows MAPE had to be computed without. Zero, or it is null. */
  mapeSkipped: number;
  /**
   * Mean absolute scaled error. 1.0 means "no better than the in-sample naive
   * forecast"; below 1 is an improvement. `null` when the history is too short
   * or flat for the denominator to exist.
   */
  mase: number | null;
}

/**
 * The scale MASE divides by: the mean absolute `season`-step difference over the
 * **training** history.
 *
 * Exported because it is the part that goes wrong. Computing it over the test
 * window instead, or with a season of 1 when the metric is meant to be
 * seasonal, changes every MASE on the page and breaks nothing visible.
 */
export function naiveScale(history: Float32Array, season = 1): number | null {
  const m = Math.max(1, season);
  if (history.length <= m) return null;
  let sum = 0;
  let count = 0;
  for (let i = m; i < history.length; i++) {
    sum += Math.abs(history[i] - history[i - m]);
    count++;
  }
  if (count === 0) return null;
  const scale = sum / count;
  // A perfectly flat history has scale 0, and every MASE would be Infinity.
  return scale > 0 ? scale : null;
}

export function forecastMetrics(
  actual: Float32Array,
  predicted: Float32Array,
  /** The training history MASE is scaled by. */
  history: Float32Array,
  season = 1,
): ForecastMetrics {
  const n = Math.min(actual.length, predicted.length);
  if (n === 0) {
    return { mae: 0, rmse: 0, mape: null, mapeSkipped: 0, mase: null };
  }

  let sae = 0;
  let sse = 0;
  let ape = 0;
  let apeCount = 0;
  let skipped = 0;
  for (let i = 0; i < n; i++) {
    const e = predicted[i] - actual[i];
    sae += Math.abs(e);
    sse += e * e;
    if (actual[i] === 0) skipped++;
    else {
      ape += Math.abs(e / actual[i]);
      apeCount++;
    }
  }

  const mae = sae / n;
  const scale = naiveScale(history, season);
  return {
    mae,
    rmse: Math.sqrt(sse / n),
    // Reported as null rather than as an average over the rows that happened to
    // be non-zero: that average describes a different test set from every other
    // number in this object.
    mape: skipped > 0 || apeCount === 0 ? null : ape / apeCount,
    mapeSkipped: skipped,
    mase: scale == null ? null : mae / scale,
  };
}

/** Mean and spread of one metric across backtest windows. */
export function spread(values: number[]): {
  mean: number;
  min: number;
  max: number;
  median: number;
} {
  if (values.length === 0) return { mean: 0, min: 0, max: 0, median: 0 };
  const sorted = [...values].sort((a, b) => a - b);
  let sum = 0;
  for (const v of values) sum += v;
  return {
    mean: sum / values.length,
    min: sorted[0],
    max: sorted[sorted.length - 1],
    median: sorted[Math.floor(sorted.length / 2)],
  };
}
