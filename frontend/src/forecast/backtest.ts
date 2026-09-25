// Rolling-origin backtesting: the page's actual argument.
//
// **One split lies.** A single train/test cut reports one number, that number
// depends entirely on where the cut fell, and nothing about it says so. Slide
// the origin forward a few times and the metric moves — often by more than the
// difference between the methods being compared, which is the finding.
//
// The split must be rolling-origin and **never k-fold**. Shuffled
// cross-validation on a time series trains on the future and scores the past,
// which is the `/link-prediction` leakage lesson in its oldest form — and it
// fails *upward*: the metric improves, the page looks better, and nothing
// throws. Hence `noLeakage()` below and the test that calls it on every window.

import { forecast, type BaselineId } from "./baselines";
import { forecastMetrics, type ForecastMetrics } from "./metrics";

export type BacktestMode = "expanding" | "sliding";

export interface BacktestOptions {
  horizon: number;
  /** How many origins to evaluate. */
  windows: number;
  /** Steps between consecutive origins. */
  stride: number;
  mode: BacktestMode;
  /** Sliding-window training length. Ignored when expanding. */
  windowLength: number;
  season: number;
}

export interface Window {
  /** Index one past the last training point — the origin. */
  origin: number;
  trainStart: number;
  /** Exclusive. */
  trainEnd: number;
  testStart: number;
  /** Exclusive. */
  testEnd: number;
  metrics: ForecastMetrics;
}

export interface BacktestResult {
  windows: Window[];
  /**
   * The single classic split — train on everything but the last `horizon`
   * points, score on those.
   *
   * It comes out of **this** call rather than a second one, which is the only
   * way the comparison the page makes is honest: two runs could differ in the
   * season, the horizon or the method and the gap would be attributed to the
   * splitting.
   */
  single: ForecastMetrics;
  /** The single split's own indices, so the chart can mark it. */
  singleWindow: Window;
  /** Why fewer windows were produced than asked for, if any. */
  note: string | null;
}

/** Enumerate the windows a configuration implies, without scoring them. */
export function planWindows(
  length: number,
  options: BacktestOptions,
): { origin: number; trainStart: number }[] {
  const { horizon, windows, stride, mode, windowLength } = options;
  const out: { origin: number; trainStart: number }[] = [];
  // The last origin is the one whose test window ends exactly at the series'
  // end; earlier origins step back by `stride`. Counting forward from the front
  // instead silently drops the most recent — and most relevant — window.
  const lastOrigin = length - horizon;
  for (let k = windows - 1; k >= 0; k--) {
    const origin = lastOrigin - k * stride;
    const trainStart = mode === "sliding" ? Math.max(0, origin - windowLength) : 0;
    // A window needs enough history to forecast from at all.
    if (origin - trainStart < 2) continue;
    if (origin <= 0 || origin + horizon > length) continue;
    out.push({ origin, trainStart });
  }
  return out;
}

/**
 * The leakage invariant, as a function so both the test and the page can call
 * it: no training index may be greater than or equal to any test index in the
 * same window.
 */
export function noLeakage(w: Window): boolean {
  return w.trainEnd <= w.testStart && w.trainStart < w.trainEnd && w.testStart < w.testEnd;
}

export function backtest(
  values: Float32Array,
  method: BaselineId,
  options: BacktestOptions,
): BacktestResult {
  const n = values.length;
  const { horizon, season } = options;
  if (n < horizon + 2) {
    throw new Error(
      `Need at least ${horizon + 2} points to hold out ${horizon} of them.`,
    );
  }

  const score = (trainStart: number, origin: number): Window => {
    const history = values.subarray(trainStart, origin);
    const actual = values.subarray(origin, origin + horizon);
    const predicted = forecast(method, history, horizon, season);
    return {
      origin,
      trainStart,
      trainEnd: origin,
      testStart: origin,
      testEnd: origin + horizon,
      metrics: forecastMetrics(
        Float32Array.from(actual),
        predicted,
        Float32Array.from(history),
        season,
      ),
    };
  };

  const planned = planWindows(n, options);
  const windows = planned.map((p) => score(p.trainStart, p.origin));

  // The notebook split: everything but the last `horizon` points.
  const singleWindow = score(0, n - horizon);

  return {
    windows,
    single: singleWindow.metrics,
    singleWindow,
    note:
      windows.length < options.windows
        ? `Only ${windows.length} of the ${options.windows} windows fit in ${n} points — earlier origins would have had too little history.`
        : null,
  };
}
