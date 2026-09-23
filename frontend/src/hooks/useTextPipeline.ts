// The text counterpart of `useVisionPipeline`: loads a Transformers.js text
// pipeline in the generic text worker and exposes `run` as a promise. All the
// plumbing — worker lifecycle, the id-correlated pending table, and the two
// state machines of docs/standards/model-page-pattern.md §2 — lives in
// `model/useModelWorker.ts`. Task hooks (`useTextClassifier`, and the NER / QA /
// fill-mask hooks that follow) are thin wrappers around this and do not
// re-derive any of it.
//
// No transfer list and no payload conversion, unlike the audio and vision
// hooks: a string is structured-cloneable and the page usually still needs it.

import { useCallback } from "react";

import type { LoadProgress } from "@/model/progress";
import type { ModelProgress, ModelStatus } from "@/model/types";
import { useModelWorker } from "@/model/useModelWorker";
import { createTextWorker } from "@/text/client";
import type { TextInput, TextModel, TextTask } from "@/text/types";

export interface UseTextPipelineResult {
  status: ModelStatus;
  /** True before the user has asked for the weights. */
  idle: boolean;
  loading: boolean;
  ready: boolean;
  progress: ModelProgress | null;
  /** Aggregate load progress across every file. Null outside `loading`. */
  loadProgress: LoadProgress | null;
  /** Duration of the load that produced `ready`, in ms. */
  loadedInMs: number | null;
  /** The chosen backend once loaded (`"webgpu"` | `"wasm"`). */
  backend: string | null;
  running: boolean;
  error: string | null;
  /**
   * Run the loaded pipeline. `args` are spread positionally after the input, so
   * callers pass e.g. `[{ top_k: 5 }]` or `[candidateLabels, { multi_label }]`.
   * Resolves with the raw pipeline output.
   */
  run: (
    input: TextInput,
    args?: unknown[],
    /**
     * The mask literal in `input`, for `fill-mask` only. It is beside `args`
     * rather than inside them because the pipeline never sees it: the engine
     * swaps it for the loaded tokenizer's own `mask_token` first.
     */
    mask?: string,
  ) => Promise<unknown>;
  /** Start the download (no-op unless idle) — see model-page-pattern.md §3. */
  load: () => void;
  /** Re-attempt a failed load. */
  retry: (overrides?: Record<string, unknown>) => void;
  /** Abandon a load in flight, returning to `idle`. */
  cancel: () => void;
}

export function useTextPipeline(
  task: TextTask,
  model: string,
  autoLoad = false,
  /** Per-backend precision override from the catalogue entry, if it has one. */
  dtypes?: TextModel["dtypes"],
): UseTextPipelineResult {
  const worker = useModelWorker<unknown>({
    createWorker: createTextWorker,
    key: `${task}:${model}`,
    loadMessage: dtypes ? { task, model, dtypes } : { task, model },
    autoLoad,
    notReadyMessage: "Text worker not ready",
  });

  const { run: post } = worker;
  const run = useCallback(
    (input: TextInput, args?: unknown[], mask?: string): Promise<unknown> =>
      post({ input, args, mask }),
    [post],
  );

  return {
    status: worker.status,
    idle: worker.idle,
    loading: worker.loading,
    ready: worker.ready,
    progress: worker.progress,
    loadProgress: worker.loadProgress,
    loadedInMs: worker.loadedInMs,
    backend: worker.backend,
    running: worker.running,
    error: worker.error,
    run,
    load: worker.load,
    retry: worker.retry,
    cancel: worker.cancel,
  };
}
