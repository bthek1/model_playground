// DeepFilterNet3's auxiliary constants — `deepfilter-auxiliary.bin` from the
// model repo (124 KB of little-endian float32). The published ONNX graph is the
// *neural* part only; these three arrays are the DSP contract it was trained
// against, so we use the shipped copies rather than re-deriving the ERB band
// layout ourselves.
//
// Layout, verified against the reference implementation (`libDF`'s `DF::new`
// with sr 48000 / fft 960 / 32 bands / min_nb_erb_freqs 2):
//
//   offset (floats)   count     contents
//   0                 481 x 32  forward ERB matrix, [bin][band], 1/width in band
//   15392             32 x 481  inverse ERB matrix, [band][bin], 1 in band
//   30784             960       Vorbis analysis/synthesis window
//
// Note the two matrices are transposes of each other in *layout*, not just in
// value: forward is bin-major, inverse is band-major. Reading `inv` as
// `[bin][band]` yields a matrix that is silently wrong — every band would draw
// its gain from the wrong bins — which is why `parseAux` asserts the invariants
// below instead of trusting the file's size alone.

import { FFT_BINS } from "./stft";

/** ERB bands DeepFilterNet3 uses (its `config.json`: `erb_bands`). */
export const ERB_BANDS = 32;

/** Byte length of a well-formed `deepfilter-auxiliary.bin`. */
export const AUX_BYTES = (FFT_BINS * ERB_BANDS * 2 + 960) * 4;

export interface DeepFilterAux {
  /** `[bin * ERB_BANDS + band]` — 1/bandWidth inside the band, 0 elsewhere. */
  fwd: Float32Array;
  /** `[band * FFT_BINS + bin]` — 1 inside the band, 0 elsewhere. */
  inv: Float32Array;
  /** The 960-point Vorbis window used for both analysis and synthesis. */
  window: Float32Array;
  /** Bins per ERB band, derived from `inv`; sums to `FFT_BINS`. */
  widths: Int32Array;
}

/**
 * Parse and validate the auxiliary file. Throws on anything that would make the
 * DSP silently wrong — a truncated file, a transposed matrix, a window that is
 * not power-complementary. Wrong DSP here does not produce an error, it produces
 * plausible audio with artefacts, so the checks are worth their cost (they run
 * once per model load).
 */
export function parseAux(buffer: ArrayBuffer): DeepFilterAux {
  if (buffer.byteLength !== AUX_BYTES) {
    throw new Error(
      `deepfilter-auxiliary.bin: expected ${AUX_BYTES} bytes, got ${buffer.byteLength}`,
    );
  }

  const all = new Float32Array(buffer);
  const matrix = FFT_BINS * ERB_BANDS;
  const fwd = all.slice(0, matrix);
  const inv = all.slice(matrix, matrix * 2);
  const window = all.slice(matrix * 2);

  // Forward: each bin belongs to exactly one band, and each band's column sums
  // to 1 (it is a band *mean*). A transposed read fails this immediately.
  for (let band = 0; band < ERB_BANDS; band++) {
    let sum = 0;
    for (let bin = 0; bin < FFT_BINS; bin++) sum += fwd[bin * ERB_BANDS + band];
    if (Math.abs(sum - 1) > 1e-4) {
      throw new Error(
        `deepfilter-auxiliary.bin: forward ERB band ${band} sums to ${sum}, expected 1`,
      );
    }
  }

  // Inverse: a broadcast matrix of ones, one *contiguous* run per band, the
  // runs laid end to end in increasing bin order. Contiguity is the invariant
  // that catches a transposed read — a [bin][band] read of the same bytes still
  // holds 481 ones and still looks like a partition, but they are scattered,
  // and `erbSynthesis` (which walks bins in order) would hand every band the
  // wrong gains.
  const widths = new Int32Array(ERB_BANDS);
  let covered = 0;
  for (let band = 0; band < ERB_BANDS; band++) {
    let width = 0;
    for (let bin = 0; bin < FFT_BINS; bin++) {
      const v = inv[band * FFT_BINS + bin];
      if (v === 0) continue;
      if (Math.abs(v - 1) > 1e-6) {
        throw new Error(
          `deepfilter-auxiliary.bin: inverse ERB entry [${band}][${bin}] is ${v}, expected 0 or 1`,
        );
      }
      if (bin !== covered + width) {
        throw new Error(
          `deepfilter-auxiliary.bin: inverse ERB band ${band} is not a contiguous run ` +
            `(bin ${bin} outside [${covered}, ${covered + width}]) — matrix is transposed?`,
        );
      }
      width++;
    }
    if (width === 0) {
      throw new Error(`deepfilter-auxiliary.bin: inverse ERB band ${band} is empty`);
    }
    widths[band] = width;
    covered += width;
  }
  if (covered !== FFT_BINS) {
    throw new Error(
      `deepfilter-auxiliary.bin: ERB bands cover ${covered} bins, expected ${FFT_BINS}`,
    );
  }

  // Cross-check: the forward matrix stores 1/width, so it knows the band widths
  // independently. If the two disagree the file is not internally consistent.
  let bin = 0;
  for (let band = 0; band < ERB_BANDS; band++) {
    const expected = 1 / widths[band];
    for (let j = 0; j < widths[band]; j++, bin++) {
      if (Math.abs(fwd[bin * ERB_BANDS + band] - expected) > 1e-6) {
        throw new Error(
          `deepfilter-auxiliary.bin: forward ERB [${bin}][${band}] disagrees with band width ` +
            `${widths[band]}`,
        );
      }
    }
  }

  // Window: symmetric, and power-complementary at 50% overlap (w[n]^2 +
  // w[n + hop]^2 === 1). That identity is what makes our overlap-add exact.
  const size = window.length;
  for (let i = 0; i < size / 2; i++) {
    if (Math.abs(window[i] - window[size - 1 - i]) > 1e-6) {
      throw new Error("deepfilter-auxiliary.bin: window is not symmetric");
    }
  }
  const hop = size / 2;
  for (let i = 0; i < hop; i++) {
    const sum = window[i] * window[i] + window[i + hop] * window[i + hop];
    if (Math.abs(sum - 1) > 1e-5) {
      throw new Error(
        `deepfilter-auxiliary.bin: window is not power-complementary at ${i} (${sum})`,
      );
    }
  }

  return { fwd, inv, window, widths };
}

/**
 * Band energies for one frame: the mean `|X|^2` over each ERB band. `spec` is
 * interleaved complex, `FFT_BINS` long. This is `fwd^T @ |X|^2`, but written as
 * a loop because `fwd` is 99% zeros — a dense product would be 32x the work.
 */
export function erbAnalysis(
  spec: Float32Array,
  widths: Int32Array,
  out = new Float32Array(ERB_BANDS),
): Float32Array {
  let bin = 0;
  for (let band = 0; band < ERB_BANDS; band++) {
    const width = widths[band];
    let sum = 0;
    for (let j = 0; j < width; j++, bin++) {
      const re = spec[bin * 2];
      const im = spec[bin * 2 + 1];
      sum += re * re + im * im;
    }
    out[band] = sum / width;
  }
  return out;
}

/**
 * Broadcast a per-band gain back to all `FFT_BINS` bins — the inverse ERB
 * matrix applied as the flat copy it is.
 */
export function erbSynthesis(
  gains: Float32Array,
  widths: Int32Array,
  out = new Float32Array(FFT_BINS),
): Float32Array {
  let bin = 0;
  for (let band = 0; band < ERB_BANDS; band++) {
    const g = gains[band];
    for (let j = 0; j < widths[band]; j++, bin++) out[bin] = g;
  }
  return out;
}
