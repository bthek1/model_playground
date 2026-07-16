// Task-agnostic worker protocol for in-browser Transformers.js pipelines. One
// generic worker (`pipeline.worker.ts`) serves any discriminative audio task —
// the pipeline `task` is carried in the `load` message rather than baked into a
// per-task worker file. ASR keeps its own specialised worker because it drives
// the real-time capture loop (`useLiveAsr`); everything else routes through here.

import type { LoadOpts } from "./backend";

/** Transformers.js pipeline task strings this generic worker supports. */
export type PipelineTask =
  | "audio-classification"
  | "zero-shot-audio-classification";

/** Transformers.js `progress_callback` payload (loosely typed — many variants). */
export interface PipelineProgress {
  status: string;
  name?: string;
  file?: string;
  /** 0–100 while a file downloads. */
  progress?: number;
  loaded?: number;
  total?: number;
}

/** One `{ label, score }` prediction (audio-classification + zero-shot). */
export interface ClassLabel {
  label: string;
  score: number;
}

// --- Worker message protocol -------------------------------------------------

/** Main thread → worker. `args` are spread as positional args to the pipeline. */
export type PipelineRequest =
  | { type: "load"; task: PipelineTask; model: string; opts?: LoadOpts }
  | { type: "run"; id: number; input: Float32Array; args?: unknown[] };

/** Worker → main thread. */
export type PipelineResponse =
  | { type: "progress"; progress: PipelineProgress }
  | { type: "ready"; model: string; backend: LoadOpts["device"] }
  | { type: "result"; id: number; result: unknown }
  | { type: "error"; id?: number; error: string };
