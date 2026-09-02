// Shared types for the in-browser ASR pipeline: the model catalogue, the result
// shape, and the worker message protocol used by both the engine (worker side)
// and `useAsr` (main-thread side).

import type { LoadOpts } from "./backend";
import type { MeasuredBytes } from "./size";

/** A selectable ASR model. Both are ONNX-exported with WebGPU + WASM support. */
export interface AsrModel {
  id: string;
  label: string;
  hint: string;
  /** Parameter count in millions — drives the size-before-load estimate. */
  params: number;
  /**
   * Measured download bytes. ASR needs these because the WASM path keeps the
   * decoder at fp32 (`asrLoadOpts`), so a uniform-q8 estimate understates it ~3x.
   */
  bytes: MeasuredBytes;
}

/** ONNX ASR models with first-class Transformers.js support (see the plan). */
export const ASR_MODELS: AsrModel[] = [
  {
    id: "onnx-community/whisper-base",
    label: "Whisper base",
    hint: "Timestamps, 99 languages, translate.",
    params: 74,
    // fp16 39.4 + 99.9 MB · WASM q8 encoder 22.1 + fp32 decoder 198.9 MB
    bytes: { webgpu: 146_276_352, wasm: 231_735_296 },
  },
  {
    id: "onnx-community/moonshine-tiny-ONNX",
    label: "Moonshine tiny",
    hint: "English, low latency — best for live captioning.",
    params: 27,
    // WASM q8 encoder 7.6 + fp32 decoder 74.6 MB
    bytes: { wasm: 86_179_840 },
  },
];

export const DEFAULT_ASR_MODEL = ASR_MODELS[0].id;

/** One timestamped segment of a transcript. `end` is null while open-ended. */
export interface AsrChunk {
  text: string;
  timestamp: [number, number | null];
}

export interface AsrResult {
  text: string;
  chunks?: AsrChunk[];
}

/** Transformers.js `progress_callback` payload (loosely typed — many variants). */
export interface AsrProgress {
  status: string;
  name?: string;
  file?: string;
  /** 0–100 while a file downloads. */
  progress?: number;
  loaded?: number;
  total?: number;
}

/** Options forwarded to the ASR pipeline call; mirrors the notebook's flags. */
export interface AsrRunArgs {
  return_timestamps?: boolean;
  chunk_length_s?: number;
  language?: string;
  task?: "transcribe" | "translate";
}

// --- Worker message protocol -------------------------------------------------

/** Main thread → worker. */
export type AsrRequest =
  | { type: "load"; model: string; opts?: LoadOpts }
  | { type: "run"; id: number; audio: Float32Array; args?: AsrRunArgs };

/** Worker → main thread. */
export type AsrResponse =
  | { type: "progress"; progress: AsrProgress }
  | { type: "ready"; model: string; backend: LoadOpts["device"] }
  | { type: "result"; id: number; result: AsrResult }
  | { type: "error"; id?: number; error: string };
