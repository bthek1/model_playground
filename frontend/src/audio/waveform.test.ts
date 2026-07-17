import { describe, expect, it } from "vitest";

import { computePeaks, formatDuration } from "./waveform";

describe("computePeaks", () => {
  it("returns one interleaved [min, max] pair per bucket", () => {
    // Two buckets of two samples each (values exactly representable in f32).
    const peaks = computePeaks(new Float32Array([-0.5, 0.25, 0.125, -0.875]), 2);
    expect(Array.from(peaks)).toEqual([-0.5, 0.25, -0.875, 0.125]);
  });

  it("handles more buckets than samples (single-sample buckets, zero tail)", () => {
    const peaks = computePeaks(new Float32Array([0.4]), 3);
    // Bucket 0 sees the sample; every bucket's floor(start) is 0 for tiny
    // inputs, so all buckets read sample 0 — no NaNs, no out-of-range reads.
    expect(peaks).toHaveLength(6);
    for (const v of peaks) expect(Number.isFinite(v)).toBe(true);
    expect(peaks[0]).toBeCloseTo(0.4);
    expect(peaks[1]).toBeCloseTo(0.4);
  });

  it("returns zeros for empty input or zero buckets", () => {
    expect(Array.from(computePeaks(new Float32Array(0), 4))).toEqual([
      0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    expect(computePeaks(new Float32Array([1]), 0)).toHaveLength(0);
  });

  it("covers every sample exactly once across buckets", () => {
    // A lone spike must land in exactly one bucket's max.
    const samples = new Float32Array(100);
    samples[37] = 1;
    const peaks = computePeaks(samples, 10);
    const maxes = Array.from({ length: 10 }, (_, b) => peaks[b * 2 + 1]);
    expect(maxes.filter((m) => m === 1)).toHaveLength(1);
    expect(maxes[3]).toBe(1); // sample 37 → bucket 3 (samples 30–39)
  });
});

describe("formatDuration", () => {
  it("formats seconds under a minute with a leading zero", () => {
    expect(formatDuration(7.3 * 16000, 16000)).toBe("0:07.3");
  });

  it("formats minutes + seconds", () => {
    expect(formatDuration(75 * 16000, 16000)).toBe("1:15.0");
  });
});
