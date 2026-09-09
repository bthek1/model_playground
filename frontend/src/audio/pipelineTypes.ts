// Task-agnostic worker protocol for in-browser Transformers.js pipelines. One
// generic worker (`pipeline.worker.ts`) serves any discriminative audio task —
// the pipeline `task` is carried in the `load` message rather than baked into a
// per-task worker file. ASR keeps its own specialised worker because it drives
// the real-time capture loop (`useLiveAsr`); everything else routes through here.

import type { ModelProgress, ModelRequest, ModelResponse } from "@/model/types";

import type { LoadOpts } from "@/model/backend";

/** Transformers.js pipeline task strings this generic worker supports. */
export type PipelineTask =
  | "audio-classification"
  | "zero-shot-audio-classification";

/** Load/warm-up progress. Alias of the shared `ModelProgress`. */
export type PipelineProgress = ModelProgress;

/** One `{ label, score }` prediction (audio-classification + zero-shot). */
export interface ClassLabel {
  label: string;
  score: number;
}

// --- Worker message protocol -------------------------------------------------
//
// The envelope is shared with every other model worker (`model/types.ts`); only
// the load/run payloads below are pipeline-specific.

/** Main thread → worker. `args` are spread as positional args to the pipeline. */
export type PipelineRequest = ModelRequest<
  { task: PipelineTask; model: string; opts?: LoadOpts },
  { input: Float32Array; args?: unknown[] }
>;

/** Worker → main thread. */
export type PipelineResponse = ModelResponse<unknown>;
