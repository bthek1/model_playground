// Waveform peak extraction — the pure math behind the audio waveform canvases
// (components/audio/Waveform.tsx). Kept separate from the drawing code so it can
// be unit-tested without a canvas, mirroring the heatmap split in
// components/viz/heatmap.tsx (paintDiverging vs HeatmapTile).

/**
 * Reduce `samples` to `buckets` [min, max] pairs — one pair per horizontal pixel
 * of a waveform. Returned interleaved as `Float32Array` of length `buckets * 2`
 * (`[min0, max0, min1, max1, …]`). Buckets past the end of a short clip are
 * `[0, 0]`, so a fixed-width canvas renders a partial take flush-left.
 */
export function computePeaks(samples: Float32Array, buckets: number): Float32Array {
  const peaks = new Float32Array(buckets * 2);
  if (samples.length === 0 || buckets === 0) return peaks;

  const perBucket = samples.length / buckets;
  for (let b = 0; b < buckets; b++) {
    const start = Math.floor(b * perBucket);
    const end = Math.min(samples.length, Math.max(start + 1, Math.floor((b + 1) * perBucket)));
    let min = Infinity;
    let max = -Infinity;
    for (let i = start; i < end; i++) {
      const v = samples[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    peaks[b * 2] = min === Infinity ? 0 : min;
    peaks[b * 2 + 1] = max === -Infinity ? 0 : max;
  }
  return peaks;
}

/** Duration of a clip in seconds, formatted `m:ss.t` (e.g. `0:07.3`). */
export function formatDuration(sampleCount: number, sampleRate: number): string {
  const total = sampleCount / sampleRate;
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  return `${m}:${s < 10 ? "0" : ""}${s.toFixed(1)}`;
}
