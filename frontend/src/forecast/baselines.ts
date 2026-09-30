// The three baselines you have to beat, and they are the page.
//
// Twenty lines of arithmetic each, which is exactly why they are worth shipping:
// a forecasting model that cannot beat "the last value, repeated" has not earned
// its download, and most of them cannot on most series. There is no model on
// this page at all — the baselines *are* the null models, so the comparison on
// screen is between them rather than against them.
//
// Each of the three has a specific way of being got wrong, and each of those is
// a test rather than a comment.

export type BaselineId = "naive" | "seasonal-naive" | "drift" | "mean";

export interface BaselineInfo {
  id: BaselineId;
  label: string;
  blurb: string;
}

export const BASELINES: BaselineInfo[] = [
  {
    id: "naive",
    label: "Naive",
    blurb:
      "The last value, repeated. On anything close to a random walk this is genuinely hard to beat, which is the whole point of putting it first.",
  },
  {
    id: "seasonal-naive",
    label: "Seasonal naive",
    blurb:
      "The value one season back. Strong on anything with a real period — and confidently wrong by a phase if the season length is wrong, which is why it is a control rather than something detected for you.",
  },
  {
    id: "drift",
    label: "Drift",
    blurb:
      "The last value plus the average per-step change over the history — a straight line through the first and last points. Not the regression slope through all of them, which is the version people reach for and a different forecast.",
  },
  {
    id: "mean",
    label: "Historical mean",
    blurb:
      "The average of everything so far. Almost always the worst of the four on real data, and worth seeing lose.",
  },
];

/**
 * Forecast `horizon` steps ahead from `history`.
 *
 * `season` is used only by seasonal naive and is **never inferred**: guessing
 * the period and being wrong produces a confident, plausible, wrong forecast
 * with no symptom anywhere. The control belongs to the user.
 */
export function forecast(
  id: BaselineId,
  history: Float32Array,
  horizon: number,
  season = 1,
): Float32Array {
  const n = history.length;
  if (n === 0) throw new Error("No history to forecast from.");
  const out = new Float32Array(horizon);

  switch (id) {
    case "naive": {
      out.fill(history[n - 1]);
      return out;
    }
    case "seasonal-naive": {
      const m = Math.max(1, Math.min(season, n));
      // ŷ_{T+h} = y_{T+h−m(k+1)} with k = ⌊(h−1)/m⌋ — that is, walk the last
      // full season forward and repeat it. Reaching for `history[n − m + h]`
      // instead runs off the end the moment the horizon exceeds one season.
      for (let h = 1; h <= horizon; h++) {
        const k = Math.floor((h - 1) / m);
        out[h - 1] = history[n + h - m * (k + 1) - 1];
      }
      return out;
    }
    case "drift": {
      // The **mean per-step change**, which telescopes to (last − first)/(n−1).
      // The version people write instead is the least-squares slope through the
      // whole series, which is a different number on anything but a straight
      // line — and gives a different forecast with nothing to notice.
      const slope = n > 1 ? (history[n - 1] - history[0]) / (n - 1) : 0;
      for (let h = 1; h <= horizon; h++) out[h - 1] = history[n - 1] + h * slope;
      return out;
    }
    case "mean": {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += history[i];
      out.fill(sum / n);
      return out;
    }
  }
}

/**
 * The least-squares slope through the whole history.
 *
 * Exported **only** so a test can assert that drift is not this. It is the
 * plausible wrong implementation, and the two agree exactly on a straight line —
 * which is the series everyone checks a drift forecast against.
 */
export function regressionSlope(history: Float32Array): number {
  const n = history.length;
  if (n < 2) return 0;
  const meanX = (n - 1) / 2;
  let meanY = 0;
  for (let i = 0; i < n; i++) meanY += history[i];
  meanY /= n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const dx = i - meanX;
    num += dx * (history[i] - meanY);
    den += dx * dx;
  }
  return den === 0 ? 0 : num / den;
}
