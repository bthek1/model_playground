// DeepFilterNet3 speech enhancement: the audio-in / audio-out pipeline around
// the published ONNX graph (`soniqo/DeepFilterNet3-ONNX`).
//
// This is the one audio task with no Transformers.js path. The repo ships "the
// neural graph only" — it takes two normalised feature tensors and returns an
// ERB mask plus complex deep-filter coefficients. STFT, ERB analysis, feature
// normalisation, mask application, deep filtering and overlap-add synthesis are
// all ours, and every one of them fails *quietly*: a wrong constant produces
// plausible audio with artefacts, not an exception.
//
// The whole chain was therefore validated against the official implementation
// (DeepFilterNet v0.5.6 + libDF) rather than by ear. On a 22 s clip our output
// matches `df.enhance()` at 50–57 dB SI-SDR (max abs sample error ~1e-3) — the
// same order the model card reports for its own reference integration.
//
// The four constants that had to be right, and what each one silently breaks:
//
//   SPEC_SCALE   The reference scales the analysis spectrum by
//                `2 * hop / fft^2 = 1/960`. The unit-norm feature divides by
//                `sqrt(state)`, so it is NOT level-invariant: drop this and the
//                network sees a signal ~31x too loud and masks it away.
//   PAD_FRONT    One *hop*, not one window. The streaming reference holds a
//                single hop of history, so frame k spans `[k-1, k+1) * hop`.
//   PAD_BACK     One full window of silence, so the last real samples are
//                flushed through the two-frame lookahead.
//   DF_OFFSET    `df_order - 1 - df_lookahead = 2`. Output frame t is filtered
//                from noisy frames t-2 … t+2. Off by one here is the classic
//                metallic-artefact bug.
//
// The model's `conv_lookahead` needs no handling: the export bakes the feature
// shift into the graph, so mask/coefficient frame t applies to spectrum frame t.

import { ERB_BANDS, erbSynthesis, type DeepFilterAux } from "./aux";
import { DF_BINS, extractFeatures } from "./features";
import { FFT_BINS, FFT_SIZE, HOP_SIZE, SAMPLE_RATE, istft, stft, type Stft } from "./stft";

export { SAMPLE_RATE };

/** Order of the complex deep filter (`df_order`). */
export const DF_ORDER = 5;
/** Frames of lookahead the deep filter is allowed (`df_lookahead`). */
export const DF_LOOKAHEAD = 2;
/** Taps before the current frame: `DF_ORDER - 1 - DF_LOOKAHEAD`. */
const DF_OFFSET = DF_ORDER - 1 - DF_LOOKAHEAD;

/** `libDF`'s `wnorm` — see the header. */
export const SPEC_SCALE = (2 * HOP_SIZE) / (FFT_SIZE * FFT_SIZE);

const PAD_FRONT = HOP_SIZE;
const PAD_BACK = FFT_SIZE;

/** What the ONNX graph returns for a whole clip. */
export interface DeepFilterOutputs {
  /** `[frames, 32]` per-band gains. */
  erbMask: Float32Array;
  /** `[DF_ORDER, frames, DF_BINS, 2]` complex coefficients. */
  dfCoefs: Float32Array;
}

/** Runs the graph. Abstracted so the DSP is testable without ONNX Runtime. */
export type DeepFilterInference = (
  featErb: Float32Array,
  featSpec: Float32Array,
  frames: number,
) => Promise<DeepFilterOutputs>;

/** Analyse `signal` at DeepFilterNet3's framing and scaling. */
export function analyse(signal: Float32Array, window: Float32Array): Stft {
  const spec = stft(signal, window, HOP_SIZE, FFT_BINS, PAD_FRONT, PAD_BACK);
  for (let i = 0; i < spec.data.length; i++) spec.data[i] *= SPEC_SCALE;
  return spec;
}

/** Overlap-add back to time domain, undoing {@link analyse}'s scaling. */
export function synthesise(
  spec: Stft,
  length: number,
  window: Float32Array,
): Float32Array {
  const scaled: Stft = { ...spec, data: new Float32Array(spec.data.length) };
  for (let i = 0; i < spec.data.length; i++) scaled.data[i] = spec.data[i] / SPEC_SCALE;
  return istft(scaled, length, window, HOP_SIZE, PAD_FRONT);
}

/**
 * Apply the ERB mask and the deep filter to `spec`, in place.
 *
 * Order matters and is not the obvious one: the mask is applied to the whole
 * spectrum first, then bins below `DF_BINS` are *overwritten* by the deep
 * filter — which reads from an untouched copy of the **noisy** spectrum, not
 * from the masked result. Filtering the already-masked bins would apply the
 * suppression twice.
 */
export function applyEnhancement(
  spec: Stft,
  { erbMask, dfCoefs }: DeepFilterOutputs,
  widths: Int32Array,
): void {
  const { frames } = spec;
  // The deep filter's taps span frames t-2 … t+2, so it must read the noisy
  // spectrum from before the mask touched it.
  const noisy = spec.data.slice();
  const gains = new Float32Array(FFT_BINS);

  for (let t = 0; t < frames; t++) {
    erbSynthesis(erbMask.subarray(t * ERB_BANDS, (t + 1) * ERB_BANDS), widths, gains);
    const base = t * FFT_BINS * 2;
    for (let bin = 0; bin < FFT_BINS; bin++) {
      spec.data[base + bin * 2] *= gains[bin];
      spec.data[base + bin * 2 + 1] *= gains[bin];
    }
  }

  for (let t = 0; t < frames; t++) {
    const base = t * FFT_BINS * 2;
    for (let f = 0; f < DF_BINS; f++) {
      let re = 0;
      let im = 0;
      for (let i = 0; i < DF_ORDER; i++) {
        const frame = t - DF_OFFSET + i;
        if (frame < 0 || frame >= frames) continue;
        const c = ((i * frames + t) * DF_BINS + f) * 2;
        const cRe = dfCoefs[c];
        const cIm = dfCoefs[c + 1];
        const sRe = noisy[frame * FFT_BINS * 2 + f * 2];
        const sIm = noisy[frame * FFT_BINS * 2 + f * 2 + 1];
        re += cRe * sRe - cIm * sIm;
        im += cRe * sIm + cIm * sRe;
      }
      spec.data[base + f * 2] = re;
      spec.data[base + f * 2 + 1] = im;
    }
  }
}

/**
 * Enhance a whole clip: 48 kHz mono in, 48 kHz mono out, same length.
 *
 * Offline (one pass over every frame) rather than streaming. The normalisation
 * state and the filter's lookahead both make a live path a separate problem;
 * see the plan.
 */
export async function enhanceAudio(
  signal: Float32Array,
  aux: DeepFilterAux,
  infer: DeepFilterInference,
): Promise<Float32Array> {
  if (signal.length === 0) return new Float32Array(0);
  const spec = analyse(signal, aux.window);
  const { featErb, featSpec, frames } = extractFeatures(spec, aux.widths);
  const outputs = await infer(featErb, featSpec, frames);
  applyEnhancement(spec, outputs, aux.widths);
  return synthesise(spec, signal.length, aux.window);
}
