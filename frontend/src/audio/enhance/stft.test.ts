import { describe, expect, it } from "vitest";

import {
  FFT_BINS,
  FFT_SIZE,
  HOP_SIZE,
  frameCount,
  istft,
  powerSpectrum,
  stft,
  vorbisWindow,
} from "./stft";

function randomSignal(n: number, seed = 1): Float32Array {
  let x = seed;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x = (x * 1103515245 + 12345) % 2147483648;
    out[i] = x / 2147483648 - 0.5;
  }
  return out;
}

/** Worst-case absolute error between two signals. */
function maxError(a: Float32Array, b: Float32Array): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

describe("vorbisWindow", () => {
  it("is symmetric", () => {
    const w = vorbisWindow(FFT_SIZE);
    for (let i = 0; i < w.length; i++) {
      expect(w[i]).toBeCloseTo(w[w.length - 1 - i], 6);
    }
  });

  it("is power-complementary at 50% overlap", () => {
    // This is the property that makes analysis+synthesis lossless, and it is
    // how we recognise the window shipped in the model's auxiliary file.
    const w = vorbisWindow(FFT_SIZE);
    const half = FFT_SIZE / 2;
    for (let i = 0; i < half; i++) {
      expect(w[i] ** 2 + w[i + half] ** 2).toBeCloseTo(1, 5);
    }
  });

  it("stays within [0, 1]", () => {
    const w = vorbisWindow(FFT_SIZE);
    for (const v of w) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});

describe("frameCount", () => {
  it("covers the signal, including a partial tail", () => {
    expect(frameCount(HOP_SIZE)).toBe(1);
    expect(frameCount(HOP_SIZE + 1)).toBe(2);
    expect(frameCount(0)).toBe(1);
  });
});

describe("stft", () => {
  it("produces the configured bin count per frame", () => {
    const spec = stft(randomSignal(48000));
    expect(spec.bins).toBe(FFT_BINS);
    expect(spec.data.length).toBe(spec.frames * FFT_BINS * 2);
    expect(spec.frames).toBeGreaterThan(1);
  });

  it("puts a steady tone at the expected bin", () => {
    // 1000 Hz at 48 kHz with a 960-point window → bin 20 (1000 / (48000/960)).
    const n = 48000;
    const sig = new Float32Array(n);
    for (let i = 0; i < n; i++) sig[i] = Math.sin((2 * Math.PI * 1000 * i) / 48000);

    const spec = stft(sig);
    const power = powerSpectrum(spec);
    // Use a middle frame; the edges are ramped by the padding.
    const f = Math.floor(spec.frames / 2);
    const frame = power.subarray(f * FFT_BINS, (f + 1) * FFT_BINS);

    let peak = 0;
    for (let b = 1; b < FFT_BINS; b++) if (frame[b] > frame[peak]) peak = b;
    // A true 960-point transform at 48 kHz has 50 Hz bins, so 1000 Hz is bin 20
    // exactly. (Padding to 1024 would put it near 21 — the bug this catches.)
    expect(peak).toBe(20);
  });
});

describe("istft", () => {
  it("round-trips random noise", () => {
    const sig = randomSignal(12000, 9);
    const restored = istft(stft(sig), sig.length);
    expect(restored.length).toBe(sig.length);
    expect(maxError(restored, sig)).toBeLessThan(1e-4);
  });

  it("round-trips a tone, including the signal edges", () => {
    const n = 9600;
    const sig = new Float32Array(n);
    for (let i = 0; i < n; i++) sig[i] = 0.8 * Math.sin((2 * Math.PI * 440 * i) / 48000);

    const restored = istft(stft(sig), n);
    expect(maxError(restored, sig)).toBeLessThan(1e-4);
    // Edges specifically — overlap-add normalisation is weakest there.
    expect(restored[0]).toBeCloseTo(sig[0], 4);
    expect(restored[n - 1]).toBeCloseTo(sig[n - 1], 4);
  });

  it("round-trips a signal shorter than one window", () => {
    const sig = randomSignal(100, 21);
    const restored = istft(stft(sig), sig.length);
    expect(maxError(restored, sig)).toBeLessThan(1e-4);
  });

  it("scales linearly — halving the spectrum halves the output", () => {
    const sig = randomSignal(4800, 13);
    const spec = stft(sig);
    for (let i = 0; i < spec.data.length; i++) spec.data[i] *= 0.5;
    const restored = istft(spec, sig.length);
    for (let i = 0; i < sig.length; i++) {
      expect(restored[i]).toBeCloseTo(sig[i] * 0.5, 4);
    }
  });
});

describe("powerSpectrum", () => {
  it("is |re|^2 + |im|^2 per bin", () => {
    const spec = stft(randomSignal(2400, 4));
    const power = powerSpectrum(spec);
    expect(power.length).toBe(spec.frames * spec.bins);
    const re = spec.data[10 * 2];
    const im = spec.data[10 * 2 + 1];
    expect(power[10]).toBeCloseTo(re * re + im * im, 6);
  });

  it("is never negative", () => {
    const power = powerSpectrum(stft(randomSignal(2400, 8)));
    for (const v of power) expect(v).toBeGreaterThanOrEqual(0);
  });
});
