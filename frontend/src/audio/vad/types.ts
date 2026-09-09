// Types for voice activity detection. Like audio-to-audio (and unlike every
// pipeline route) this task has no Transformers.js path — Silero publishes a
// bare ONNX graph — so the payloads carry raw samples and the result is an
// array of numbers rather than a label.

import type { ModelProgress, ModelRequest, ModelResponse } from "@/model/types";

import type { Backend } from "@/model/backend";
import type { MeasuredBytes } from "@/model/size";
import { FRAME_SAMPLES } from "./vad";

/** 16 kHz mono, the default for `decodeToMono` — this is not the 48 kHz route. */
export const SAMPLE_RATE = 16000;

export { FRAME_SAMPLES };

export interface VadModel {
  id: string;
  label: string;
  hint: string;
  /** Hugging Face repo, or `null` for the baseline that downloads nothing. */
  repo: string | null;
  /** File within the repo. */
  file?: string;
  /** Parameter count in millions — the shared `ModelPicker` wants it. */
  params: number;
  /** Measured download; both backends quote the same number (see below). */
  bytes: MeasuredBytes;
}

/**
 * The catalogue.
 *
 * Silero is loaded as the **fp32 `model.onnx`** rather than one of the repo's
 * quantized variants. At 2.2 MB the download is already noise next to every
 * other model in the app, an LSTM graph is exactly the kind of thing q8 export
 * bugs bite (see `asrLoadOpts` for the ASR version of that story), and the
 * measured cost of running it is 0.3 ms per 32 ms frame — there is nothing here
 * worth trading accuracy for.
 */
export const VAD_MODELS: VadModel[] = [
  {
    id: "onnx-community/silero-vad",
    label: "Silero VAD v5",
    hint: "The standard. Frame-level speech probability, robust to noise and music.",
    repo: "onnx-community/silero-vad",
    file: "onnx/model.onnx",
    // 2_243_022 B fp32. Same file on both backends: this one always runs on
    // WASM (see `session.ts`), so there is no fp16 variant to quote.
    params: 0.56,
    bytes: { webgpu: 2_243_022, wasm: 2_243_022 },
  },
  {
    id: "energy",
    label: "Energy VAD",
    hint: "No model, no download — short-time level, as a baseline to beat.",
    repo: null,
    params: 0,
    bytes: { webgpu: 0, wasm: 0 },
  },
];

export const DEFAULT_VAD_MODEL = VAD_MODELS[0].id;

export const ENERGY_VAD_MODEL = "energy";

export interface VadResult {
  /** Speech probability per `frameSamples`-long frame, 0–1. */
  probabilities: Float32Array;
  frameSamples: number;
  sampleRate: number;
  /** Length of the clip these frames cover, in samples. */
  samples: number;
}

export type VadProgress = ModelProgress;

/** Main thread → worker. */
export type VadRequest = ModelRequest<
  { model: string; backend?: Backend },
  { audio: Float32Array }
>;

/** Worker → main thread. */
export type VadResponse = ModelResponse<VadResult>;
