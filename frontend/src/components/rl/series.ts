// The arithmetic behind the return chart, kept out of the component so it can
// be tested without a chart — and so the running mean the page quotes and the
// one it draws are the same function.

/** Trailing mean over `window` episodes — the first `window − 1` average what exists. */
export function runningMean(values: readonly number[], window: number): number[] {
  const out = new Array<number>(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= window) sum -= values[i - window];
    out[i] = sum / Math.min(i + 1, window);
  }
  return out;
}

/** Largest number of points drawn per series; longer runs are strided. */
export const MAX_POINTS = 2000;

/** Every k-th index so a series has at most `MAX_POINTS`, always keeping the last. */
export function strideIndices(n: number, max = MAX_POINTS): number[] {
  if (n <= max) return Array.from({ length: n }, (_, i) => i);
  const step = n / max;
  const out: number[] = [];
  for (let k = 0; k < max - 1; k++) out.push(Math.floor(k * step));
  out.push(n - 1);
  return out;
}
