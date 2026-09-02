// STFT / ISTFT with overlap-add, at DeepFilterNet3's framing: 960-point window,
// 480-sample hop (50% overlap), 481 retained bins.
//
// The window matters as much as the transform. DFN3 ships its own (Vorbis) in
// `deepfilter-auxiliary.bin`; that is the model's ground truth and Phase 2 uses
// it. `vorbisWindow()` here reproduces it so the DSP can be tested before the
// network is involved — the two are asserted equal once the aux file loads.

import { dft, fft, irfft, rfft } from "./fft";

/** DeepFilterNet3 framing (its config.json — do not hard-code these elsewhere). */
export const FFT_SIZE = 960;
export const HOP_SIZE = 480;
export const FFT_BINS = 481;
export const SAMPLE_RATE = 48000;

/**
 * The Vorbis power-complementary window: `sin(pi/2 * sin^2(pi/N * (n + 0.5)))`.
 * Squared and overlap-added at 50% it sums to exactly 1, which is what makes
 * analysis-then-synthesis lossless when the same window is used both ways.
 */
export function vorbisWindow(size: number): Float32Array {
  const w = new Float32Array(size);
  for (let n = 0; n < size; n++) {
    const inner = Math.sin((Math.PI / size) * (n + 0.5));
    w[n] = Math.sin((Math.PI / 2) * inner * inner);
  }
  return w;
}

/** How many hops cover `length` samples, padding the tail frame. */
export function frameCount(length: number, hop = HOP_SIZE): number {
  return Math.max(1, Math.ceil(length / hop));
}

export interface Stft {
  /** `frames * bins * 2` interleaved complex values, frame-major. */
  data: Float32Array;
  frames: number;
  bins: number;
}

/**
 * Analyse `signal` into overlapping windowed spectra.
 *
 * The signal is zero-padded at both ends so every sample is covered by two
 * frames and overlap-add reconstructs the original including its edges.
 * `padFront` also *chooses the framing*: frame `k` starts at `k * hop -
 * padFront` in the original signal. DeepFilterNet3's streaming analysis holds
 * one hop of history, so it needs `padFront = hop`, not a whole window — see
 * `deepFilterNet.ts`. `istft` must be given the same `padFront` to line up.
 */
export function stft(
  signal: Float32Array,
  window: Float32Array = vorbisWindow(FFT_SIZE),
  hop = HOP_SIZE,
  bins = FFT_BINS,
  padFront = window.length,
  padBack = window.length,
): Stft {
  const size = window.length;
  const padded = new Float32Array(padFront + signal.length + padBack);
  padded.set(signal, padFront);

  const frames = Math.max(1, Math.floor((padded.length - size) / hop) + 1);
  const data = new Float32Array(frames * bins * 2);
  const frame = new Float32Array(size);

  for (let f = 0; f < frames; f++) {
    const start = f * hop;
    for (let i = 0; i < size; i++) frame[i] = padded[start + i] * window[i];
    const spec = rfft(frame, bins);
    data.set(spec, f * bins * 2);
  }

  return { data, frames, bins };
}

/**
 * Overlap-add synthesis — the inverse of {@link stft} with the same window,
 * hop and `padFront`. `length` is the original signal length, used to strip the
 * padding back off.
 */
export function istft(
  spec: Stft,
  length: number,
  window: Float32Array = vorbisWindow(FFT_SIZE),
  hop = HOP_SIZE,
  padFront = window.length,
): Float32Array {
  const size = window.length;
  const out = new Float32Array(Math.max((spec.frames - 1) * hop + size, padFront + length));
  // Track the summed window energy so a partially covered edge can be corrected.
  const norm = new Float32Array(out.length);

  for (let f = 0; f < spec.frames; f++) {
    const slice = spec.data.subarray(
      f * spec.bins * 2,
      (f + 1) * spec.bins * 2,
    );
    const frame = irfft(slice, size);
    const start = f * hop;
    for (let i = 0; i < size; i++) {
      const at = start + i;
      if (at >= out.length) break;
      out[at] += frame[i] * window[i];
      norm[at] += window[i] * window[i];
    }
  }

  for (let i = 0; i < out.length; i++) {
    if (norm[i] > 1e-8) out[i] /= norm[i];
  }
  return out.slice(padFront, padFront + length);
}

/** Magnitude-squared of every bin — the input to ERB band analysis. */
export function powerSpectrum(spec: Stft): Float32Array {
  const out = new Float32Array(spec.frames * spec.bins);
  for (let i = 0; i < out.length; i++) {
    const re = spec.data[i * 2];
    const im = spec.data[i * 2 + 1];
    out[i] = re * re + im * im;
  }
  return out;
}

// Re-exported so callers touch one module for the transform layer.
export { dft, fft, rfft, irfft };
