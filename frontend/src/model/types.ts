// The shared vocabulary for every model page: one status enum, one progress
// shape, one worker protocol, one hook contract. Specified in
// docs/standards/model-page-pattern.md — read §2 and §3 before changing this.
//
// The point is that a reviewer who has read one task hook has read all of them.
// Task-specific detail lives in the *payload* generics, never in renamed fields.

import type { Backend, LoadOpts } from "./backend";
import type { LoadProgress } from "./progress";

/**
 * Machine A — the model lifecycle, one per worker.
 *
 *   idle ──load()──▶ loading ──ready──▶ ready
 *     ▲     ◀──cancel()──┘ progress ↺
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

/** One `{ label, score }` prediction. Every classifier in the app returns these. */
export interface ClassLabel {
  label: string;
  score: number;
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
 *
 * **`partial` is progress *within one run*, and it is not a fifth status.** A
 * generative decoder produces its answer over seconds — a VLM encodes the image
 * before a single token exists — so a page that can only render the finished
 * result shows an unlabelled multi-second pause, which is indistinguishable from
 * a hang. The worker reports the intermediate state against the request `id`
 * that will eventually carry the result.
 *
 * Three things it deliberately is not:
 *
 *   not a `ModelStatus`   Machine A is untouched. `ready` means the weights are
 *                         loaded; a run in progress does not move it.
 *   not `progress`        that variant is Machine A's download/warm-up self-loop
 *                         and carries no `id`. Overloading it would make
 *                         "downloading" and "generating" the same event.
 *   not a running flag    `running` stays an inflight *count*. A partial does
 *                         not open or close a request; only `run`, `result` and
 *                         `error` do.
 *
 * `TPartial` defaults to `never`, which is what keeps this change free for every
 * worker that does not stream: the arm is uninhabited, so existing exhaustive
 * switches stay exhaustive and no current engine or test is touched.
 */
export type ModelResponse<TResult, TPartial = never> =
  | { type: "progress"; progress: ModelProgress }
  | { type: "ready"; model: string; backend: LoadOpts["device"] }
  | { type: "partial"; id: number; partial: TPartial }
  | { type: "result"; id: number; result: TResult }
  | { type: "error"; id?: number; error: string };

// --- Hook contract -----------------------------------------------------------

/**
 * What every task hook returns (docs/standards/model-page-pattern.md §3).
 * Task-specific fields are additive; nothing here is renamed or dropped per
 * task. Hooks that predate this contract expose a task-named alias for `run`
 * (`transcribe`, `synthesize`) — those are being migrated route by route.
 */
export interface ModelTask<TInput, TOutput, TOpts = void, TPartial = never> {
  // Machine A — load
  status: ModelStatus;
  idle: boolean;
  loading: boolean;
  ready: boolean;
  /** Null outside `loading`. */
  progress: ModelProgress | null;
  /** Aggregate, monotonic load progress. Null outside `loading`. */
  loadProgress: LoadProgress | null;
  /** Duration of the load that produced `ready`, in ms. */
  loadedInMs: number | null;
  /** Resolved execution backend once `ready`. */
  backend: Backend | null;
  /** Start the download. No-op unless `idle`. */
  load: () => void;
  /** Re-attempt a failed load. No-op unless `error`. `overrides` ride along on
   *  the `load` message — `{ backend: "wasm" }` is the "retry on CPU" path. */
  retry: (overrides?: Record<string, unknown>) => void;
  /** Abandon a load in flight, returning to `idle`. No-op unless `loading`. */
  cancel: () => void;

  // Machine B — inference
  run: (input: TInput, opts?: TOpts) => Promise<TOutput>;
  /** True while ANY request is in flight — an inflight count, not a boolean. */
  running: boolean;
  /** Latest successful output, for pages that show one result at a time. */
  result: TOutput | null;
  /**
   * Latest in-run progress from a streaming worker: partial text, an encode/generate
   * stage, whatever the task reports. Null unless a run is in flight, and cleared the
   * moment its result lands so OUTPUT never renders a half-finished answer beside the
   * finished one. Optional because most tasks are single-shot and never post one.
   */
  partial?: TPartial | null;

  /** Load error (`status === "error"`) or the most recent run error. */
  error: string | null;
}
