// Pooling per-frame scores over time.
//
// This is the only thing `/video-classification` does that a still-image page
// does not, and it is deliberately the *cheapest* thing on the page: pure,
// synchronous, and over numbers already in hand. Changing the window re-derives
// the whole chart without touching the model — the same rule the detection
// threshold and the VAD threshold follow, and it matters more here because a
// re-run is N CLIP passes over a whole clip.
//
// A sliding mean is also the honest pooling for this page's thesis. The route is
// a **frame-level baseline**: it has no way to know that a frame follows the one
// before it, and averaging a neighbourhood is the most a model that cannot see
// motion is entitled to do with time. Anything cleverer would be smuggling in
// temporal modelling the model does not have.

/** Per-frame scores: `series[frame][label]`. */
export type ScoreSeries = readonly (readonly number[])[];

/**
 * Centred sliding mean over frames, per label.
 *
 * The window is clamped at the clip's ends rather than padded with zeros:
 * padding would drag the first and last frames' scores toward nothing and put a
 * dip at both ends of every chart, which reads as the model losing confidence
 * at the start and end of every clip.
 *
 * An even window is centred by taking the extra frame from the past.
 */
export function slidingMean(series: ScoreSeries, window: number): number[][] {
  const frames = series.length;
  if (frames === 0) return [];
  const w = Math.max(1, Math.floor(window));
  if (w === 1) return series.map((row) => [...row]);

  const labels = series[0].length;
  // The extra frame of an even window comes from the *past*: a viewer reading
  // a point on the chart is asking "what did the clip look like up to here",
  // and biasing toward the future makes the line lead the picture.
  const before = Math.floor(w / 2);
  const after = w - 1 - before;

  const out: number[][] = [];
  for (let f = 0; f < frames; f++) {
    const lo = Math.max(0, f - before);
    const hi = Math.min(frames - 1, f + after);
    const n = hi - lo + 1;
    const row = new Array<number>(labels).fill(0);
    for (let k = lo; k <= hi; k++) {
      const source = series[k];
      for (let l = 0; l < labels; l++) row[l] += source[l] ?? 0;
    }
    for (let l = 0; l < labels; l++) row[l] /= n;
    out.push(row);
  }
  return out;
}

export interface PooledVerdict {
  /** Index of the winning label. -1 when there is nothing to decide. */
  index: number;
  /** The winner's mean score over the whole clip. */
  score: number;
  /** Every label's clip-level mean, in the label list's own order. */
  means: number[];
}

/**
 * The clip-level answer: each label's mean over every frame, and the winner.
 *
 * Computed from the **unpooled** series on purpose. A sliding mean is a
 * smoothing for the chart; averaging an already-averaged series would weight
 * the middle of the clip more heavily than its ends, for no reason anyone could
 * defend.
 */
export function pooledVerdict(series: ScoreSeries): PooledVerdict {
  if (series.length === 0 || series[0].length === 0) {
    return { index: -1, score: 0, means: [] };
  }
  const labels = series[0].length;
  const means = new Array<number>(labels).fill(0);
  for (const row of series) {
    for (let l = 0; l < labels; l++) means[l] += row[l] ?? 0;
  }
  for (let l = 0; l < labels; l++) means[l] /= series.length;

  let index = 0;
  for (let l = 1; l < labels; l++) if (means[l] > means[index]) index = l;
  return { index, score: means[index], means };
}
