// Text-to-Speech model catalogue + worker protocol. Two engines behind one
// uniform interface: **kokoro-js** (Kokoro-82M — best small-model quality,
// purpose-built for the browser) and the Transformers.js **`text-to-speech`
// pipeline** (MMS-VITS, SpeechT5). TTS gets its own worker (`tts.worker.ts`)
// rather than the generic pipeline worker because the modality differs
// (text in → audio out) and kokoro-js is a separate heavy dependency that would
// bloat the classification worker bundle. See docs/plans/…/audio-models-in-browser.md.

import type { ModelRequest, ModelResponse } from "@/model/types";

import type { LoadOpts } from "./backend";

export interface TtsVoice {
  id: string;
  label: string;
}

export interface TtsModel {
  id: string;
  label: string;
  hint: string;
  /** Parameter count in millions — drives the size-before-load estimate. */
  params: number;
  engine: "kokoro" | "pipeline";
  /** Named voices — Kokoro only. */
  voices?: TtsVoice[];
  /** x-vector speaker-embedding URL — SpeechT5 only. */
  speakerEmbeddings?: string;
}

export const TTS_MODELS: TtsModel[] = [
  {
    id: "onnx-community/Kokoro-82M-v1.0-ONNX",
    label: "Kokoro 82M",
    hint: "Best small-model quality; purpose-built for the browser (WebGPU).",
    params: 82,
    engine: "kokoro",
    voices: [
      { id: "af_heart", label: "Heart — US female" },
      { id: "af_bella", label: "Bella — US female" },
      { id: "af_nicole", label: "Nicole — US female (soft)" },
      { id: "am_michael", label: "Michael — US male" },
      { id: "bf_emma", label: "Emma — UK female" },
      { id: "bm_george", label: "George — UK male" },
    ],
  },
  {
    id: "Xenova/mms-tts-eng",
    label: "MMS English",
    hint: "Tiny end-to-end VITS; swap the model id for 1,000+ languages.",
    params: 36,
    engine: "pipeline",
  },
  {
    id: "Xenova/speecht5_tts",
    label: "SpeechT5",
    hint: "Microsoft SpeechT5 with an x-vector speaker embedding.",
    params: 144,
    engine: "pipeline",
    speakerEmbeddings:
      "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin",
  },
];

export const DEFAULT_TTS_MODEL = TTS_MODELS[0].id;

/** Synthesised speech: mono samples + their rate (varies by model). */
export interface TtsAudio {
  audio: Float32Array;
  sampleRate: number;
}

export interface TtsRunOpts {
  /** Kokoro voice id (ignored by pipeline models). */
  voice?: string;
  /** Kokoro speaking speed multiplier. */
  speed?: number;
  /**
   * MusicGen only: how many audio tokens to generate (~50 per second of audio).
   * Ignored by the speech models.
   */
  maxNewTokens?: number;
}

// --- Worker message protocol -------------------------------------------------
//
// The envelope is shared with every other model worker (`model/types.ts`); only
// the load/run payloads below are TTS-specific.

/** Main thread → worker. */
export type TtsRequest = ModelRequest<
  { model: string; opts?: LoadOpts },
  { text: string; opts?: TtsRunOpts }
>;

/** Worker → main thread. The result's audio buffer is transferred (zero-copy). */
export type TtsResponse = ModelResponse<TtsAudio>;
