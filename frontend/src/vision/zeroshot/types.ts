// Worker protocol for the zero-shot route's split-tower engine.
//
// This task gets an engine of its own rather than riding the generic vision
// worker, and the roadmap already named the criterion: a task that is not a
// plain `pipeline()` call — SAM's encode-once/decode-many is the other example —
// owns its engine, the way `audio/enhance/` and `audio/vad/` do. Encoding the
// labels once and reusing them across frames is exactly that shape, and the
// generic worker has nowhere to keep the cache.
//
// The envelope is still the shared `ModelRequest`/`ModelResponse` from
// `model/types.ts`. Only the payloads below are specific to this task.

import type { Backend, DtypeSpec, LoadOpts } from "@/model/backend";
import type { ModelRequest, ModelResponse } from "@/model/types";

import type { ImagePayload } from "../image";
import type { ScoringSpec } from "./scoring";

export interface ZeroShotLoad {
  model: string;
  /** Picks the tower classes and the tokenizer padding. */
  family: "clip" | "siglip";
  scoring: ScoringSpec;
  opts?: LoadOpts;
  dtypes?: Partial<Record<Backend, DtypeSpec>>;
}

export interface ZeroShotRun {
  image: ImagePayload;
  /** The fully-templated strings to score against. Order is the answer's order. */
  prompts: string[];
}

export interface ZeroShotResult {
  /** One score per prompt, in the order the prompts were given. */
  scores: number[];
  /**
   * True when the label embeddings came from the cache rather than the text
   * tower. Surfaced in the UI: an optimisation nobody can see is an optimisation
   * nobody can check.
   */
  textCached: boolean;
  /** Milliseconds spent in the text tower this run. Zero on a cache hit. */
  textMs: number;
  /** Milliseconds spent in the vision tower. Paid on every frame regardless. */
  imageMs: number;
}

export type ZeroShotRequest = ModelRequest<ZeroShotLoad, ZeroShotRun>;
export type ZeroShotResponse = ModelResponse<ZeroShotResult>;
