// DeepFilterNet3's input features. The published graph takes two normalised
// tensors, and the normalisation is *stateful*: an exponential running mean per
// band that starts from a fixed init and decays with `alpha` per frame. Get the
// state wrong and nothing errors — the network simply sees a signal at the wrong
// apparent level and masks it away as noise.
//
// Everything here is a term-by-term port of the reference (`libDF`'s
// `band_mean_norm_erb` / `band_unit_norm` in `libDF/src/lib.rs`) and is asserted
// against arrays captured from that implementation in `features.test.ts`.

import { ERB_BANDS, erbAnalysis } from "./aux";
import type { Stft } from "./stft";
import { FFT_BINS } from "./stft";

/** Deep-filter bins — the low `df_bins` the complex branch sees. */
export const DF_BINS = 96;

/**
 * `normalization_alpha` from the model config. It is `exp(-hop / (sr * tau))`
 * rounded to 3 decimals: `exp(-0.01) = 0.99005` → `0.99`. The rounding is part
 * of the trained contract, so use the constant, not the formula.
 */
export const NORM_ALPHA = 0.99;

/** ERB mean state ramps linearly from -60 dB to -90 dB across the bands. */
const MEAN_NORM_INIT: [number, number] = [-60, -90];
/** Complex unit-norm state ramps from 0.001 to 0.0001 across the DF bins. */
const UNIT_NORM_INIT: [number, number] = [0.001, 0.0001];

function linspace(from: number, to: number, count: number): Float32Array {
  const out = new Float32Array(count);
  const step = (to - from) / (count - 1);
  for (let i = 0; i < count; i++) out[i] = from + i * step;
  return out;
}

/**
 * The per-frame normalisation state. Kept as an object rather than folded into
 * the feature function because it is exactly what a future streaming path has
 * to carry across chunks — offline processing is just one long run of it.
 */
export class FeatureNormalizer {
  private readonly meanState = linspace(
    MEAN_NORM_INIT[0],
    MEAN_NORM_INIT[1],
    ERB_BANDS,
  );
  private readonly unitState = linspace(
    UNIT_NORM_INIT[0],
    UNIT_NORM_INIT[1],
    DF_BINS,
  );

  constructor(private readonly alpha = NORM_ALPHA) {}

  /**
   * ERB band energies in dB, mean-normalised and scaled to roughly [-1, 1].
   * `bands` is overwritten in place with the feature and returned.
   */
  normErb(bands: Float32Array): Float32Array {
    const a = this.alpha;
    for (let b = 0; b < ERB_BANDS; b++) {
      const db = 10 * Math.log10(bands[b] + 1e-10);
      const s = db * (1 - a) + this.meanState[b] * a;
      this.meanState[b] = s;
      bands[b] = (db - s) / 40;
    }
    return bands;
  }

  /**
   * Unit-normalise the first `DF_BINS` complex bins of one frame. Note the
   * divisor is `sqrt(state)`, not `state`: the feature keeps a square-root
   * dependence on level, so the spectrum handed in must carry the reference
   * implementation's scaling (see `SPEC_SCALE` in `deepFilterNet.ts`).
   * `out` receives interleaved `[re, im, …]`, `DF_BINS * 2` long.
   */
  normUnit(spec: Float32Array, offset: number, out: Float32Array): Float32Array {
    const a = this.alpha;
    for (let f = 0; f < DF_BINS; f++) {
      const re = spec[offset + f * 2];
      const im = spec[offset + f * 2 + 1];
      const mag = Math.hypot(re, im);
      const s = mag * (1 - a) + this.unitState[f] * a;
      this.unitState[f] = s;
      const d = Math.sqrt(s);
      out[f * 2] = re / d;
      out[f * 2 + 1] = im / d;
    }
    return out;
  }
}

export interface DeepFilterFeatures {
  /** `[1, 1, frames, 32]`, frame-major. */
  featErb: Float32Array;
  /** `[1, 2, frames, 96]` — the whole real plane, then the whole imaginary one. */
  featSpec: Float32Array;
  frames: number;
}

/**
 * Build both input tensors for a whole clip in one pass, sharing the one
 * normalisation state across frames (the frames are not independent).
 */
export function extractFeatures(
  spec: Stft,
  widths: Int32Array,
  alpha = NORM_ALPHA,
): DeepFilterFeatures {
  if (spec.bins !== FFT_BINS) {
    throw new Error(`extractFeatures: expected ${FFT_BINS} bins, got ${spec.bins}`);
  }
  const { frames } = spec;
  const norm = new FeatureNormalizer(alpha);
  const featErb = new Float32Array(frames * ERB_BANDS);
  // The channel axis is outermost, so real and imaginary live in two separate
  // contiguous planes rather than interleaved per bin.
  const featSpec = new Float32Array(2 * frames * DF_BINS);
  const bands = new Float32Array(ERB_BANDS);
  const unit = new Float32Array(DF_BINS * 2);

  for (let t = 0; t < frames; t++) {
    const offset = t * FFT_BINS * 2;
    erbAnalysis(spec.data.subarray(offset, offset + FFT_BINS * 2), widths, bands);
    norm.normErb(bands);
    featErb.set(bands, t * ERB_BANDS);

    norm.normUnit(spec.data, offset, unit);
    for (let f = 0; f < DF_BINS; f++) {
      featSpec[t * DF_BINS + f] = unit[f * 2];
      featSpec[frames * DF_BINS + t * DF_BINS + f] = unit[f * 2 + 1];
    }
  }

  return { featErb, featSpec, frames };
}
