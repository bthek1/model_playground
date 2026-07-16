// Shared types for the in-browser ASR pipeline: the model catalogue, the result
// shape, and the worker message protocol used by both the engine (worker side)
// and `useAsr` (main-thread side).

import type { LoadOpts } from "./backend";

/** A selectable ASR model. Both are ONNX-exported with WebGPU + WASM support. */
export interface AsrModel {
  id: string;
  label: string;
  hint: string;
}

/** ONNX ASR models with first-class Transformers.js support (see the plan). */
export const ASR_MODELS: AsrModel[] = [
  {
    id: "onnx-community/whisper-base",
    label: "Whisper base",
    hint: "Timestamps, 99 languages, translate. ~150 MB.",
  },
  {
    id: "onnx-community/moonshine-tiny-ONNX",
    label: "Moonshine tiny",
    hint: "English, low latency — best for live captioning.",
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
