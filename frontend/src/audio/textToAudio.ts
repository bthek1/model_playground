// Text-to-Audio (music generation) catalogue. This is the plan's Phase 5
// "partial" task: MusicGen-small *can* run in the browser, but it is
// autoregressive — roughly 50 tokens of audio per second of compute on WASM —
// and the weights are an order of magnitude heavier than anything else we ship
// (571 MB quantized, ~1.05 GB at fp16). So it is deliberately gated behind an
// explicit opt-in in the route and never auto-loads. AudioLDM / Stable Audio are
// `diffusers` latent-diffusion with no Transformers.js path at all — those stay
// server-side. See docs/plans/completed/audio-models-in-browser.md.
//
// It reuses the **TTS worker**, not a new one: the modality is the same
// (text in → audio out) and `TtsSynthesizer` already describes it exactly.

import type { MeasuredBytes } from "./size";

export interface MusicModel {
  id: string;
  label: string;
  hint: string;
  /** Parameter count in millions (the headline decoder size). */
  params: number;
  /** Measured download — the params estimate is hopeless for a 3-graph model. */
  bytes: MeasuredBytes;
  /** Audio sample rate the model emits. */
  sampleRate: number;
}

export const MUSIC_MODELS: MusicModel[] = [
  {
    id: "Xenova/musicgen-small",
    label: "MusicGen small",
    hint: "Text-prompted music. Autoregressive and slow — seconds of compute per second of audio.",
    params: 300,
    // Three graphs: text encoder + decoder + EnCodec vocoder.
    // q8  105.3 + 408.6 + 57.1 MB = 571 MB
    // fp16 209.3 + 808.9 + 56.4 MB = 1074.6 MB
    bytes: { wasm: 598_736_896, webgpu: 1_126_825_984 },
    sampleRate: 32_000,
  },
];

export const DEFAULT_MUSIC_MODEL = MUSIC_MODELS[0].id;

/**
 * Measured on WASM: 60 tokens produced 36,480 samples at 32 kHz (1.14 s of
 * audio). Used to turn a requested duration into `max_new_tokens` and to warn
 * the user roughly how long they will wait.
 */
export const TOKENS_PER_SECOND = 50;

/** Keep the toy short — generation time grows linearly with this. */
export const MAX_SECONDS = 15;
export const DEFAULT_SECONDS = 5;

export function tokensForSeconds(seconds: number): number {
  return Math.max(16, Math.round(seconds * TOKENS_PER_SECOND));
}
