// The shared vocabulary for every model page: one status enum, one progress
// shape, one worker protocol, one hook contract. Specified in
// docs/standards/model-page-pattern.md — read §2 and §3 before changing this.
//
// The point is that a reviewer who has read one task hook has read all of them.
// Task-specific detail lives in the *payload* generics, never in renamed fields.

import type { Backend, LoadOpts } from "@/audio/backend";

/**
 * Machine A — the model lifecycle, one per worker.
 *
 *   idle ──load()──▶ loading ──ready──▶ ready
 *     ▲                 │ progress ↺
 *     │                 │ error (no id)
 *     │                 ▼
 *     └── model change ─ error ──retry()──▶ loading
 *
 * `idle` is the default once a page opts out of `autoLoad`: weights are the
 * user's bandwidth, so the download starts on an action. `progress` events are a
 * self-loop on `loading` — they never move `status`. An *inference* failure does
 * not leave `ready`; only a load failure reaches `error`.
 */
export type ModelStatus = "idle" | "loading" | "ready" | "error";

/**
 * Weight-download / warm-up progress. Loosely typed on purpose: this mirrors
 * Transformers.js's `progress_callback` payload, which has many variants. Our
 * engines also post a synthetic `{ status: "warmup" }`.
 */
export interface ModelProgress {
  status: string;
  name?: string;
  file?: string;
  /** 0–100 while a file downloads. */
  progress?: number;
  loaded?: number;
  total?: number;
}

// --- Worker protocol ---------------------------------------------------------
//
// Every model worker speaks this. `TLoad` and `TRun` carry the task-specific
// payload (a model id, a pipeline task, the audio or text to run on) so the
// envelope — the message names, the `id` correlation, the response union — is
// identical across ASR, the generic pipeline worker, and TTS.

/** Main thread → worker. */
export type ModelRequest<TLoad, TRun> =
  | ({ type: "load" } & TLoad)
  | ({ type: "run"; id: number } & TRun);

/**
 * Worker → main thread.
 *
 * `id` on an error is the discriminator between the two machines: `id != null`
 * is a request failure (Machine B, status stays `ready`), `id == null` is a load
 * failure (Machine A).
 */
export type ModelResponse<TResult> =
  | { type: "progress"; progress: ModelProgress }
  | { type: "ready"; model: string; backend: LoadOpts["device"] }
  | { type: "result"; id: number; result: TResult }
  | { type: "error"; id?: number; error: string };

// --- Hook contract -----------------------------------------------------------

/**
 * What every task hook returns (docs/standards/model-page-pattern.md §3).
 * Task-specific fields are additive; nothing here is renamed or dropped per
 * task. Hooks that predate this contract expose a task-named alias for `run`
 * (`transcribe`, `synthesize`) — those are being migrated route by route.
 */
export interface ModelTask<TInput, TOutput, TOpts = void> {
  // Machine A — load
  status: ModelStatus;
  idle: boolean;
  loading: boolean;
  ready: boolean;
  /** Null outside `loading`. */
  progress: ModelProgress | null;
  /** Resolved execution backend once `ready`. */
  backend: Backend | null;
  /** Start the download. No-op unless `idle`. */
  load: () => void;
  /** Re-attempt a failed load. No-op unless `error`. */
  retry: () => void;

  // Machine B — inference
  run: (input: TInput, opts?: TOpts) => Promise<TOutput>;
  /** True while ANY request is in flight — an inflight count, not a boolean. */
  running: boolean;
  /** Latest successful output, for pages that show one result at a time. */
  result: TOutput | null;

  /** Load error (`status === "error"`) or the most recent run error. */
  error: string | null;
}
