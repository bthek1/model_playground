// The vision counterpart of `usePipeline`: loads a Transformers.js vision
// pipeline in the generic vision worker and exposes `run` as a promise. All the
// plumbing — worker lifecycle, the id-correlated pending table, and the two
// state machines of docs/standards/model-page-pattern.md §2 — lives in
// `model/useModelWorker.ts`. Task hooks (`useImageClassifier`, and the depth /
// detection / segmentation hooks that follow) are thin wrappers around this and
// do not re-derive any of it.

import { useCallback } from "react";
import type { RawImage } from "@huggingface/transformers";

import type { LoadProgress } from "@/model/progress";
import type { ModelProgress, ModelStatus } from "@/model/types";
import { useModelWorker } from "@/model/useModelWorker";
import { createVisionWorker } from "@/vision/client";
import { toPayload, transferablesOf } from "@/vision/image";
import type { VisionModel, VisionTask } from "@/vision/types";

export interface RunImageOptions {
  /**
   * Hand the pixel buffer to the worker instead of copying it. Faster, but it
   * detaches the caller's `RawImage` — use it for a webcam frame the page has
   * finished with, never for an image still on screen.
   */
  consume?: boolean;
}

export interface UseVisionPipelineResult {
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
   * Run the loaded pipeline on an image. `args` are spread positionally onto the
   * pipeline call, so callers pass e.g. `[{ top_k: 5 }]` or `[candidateLabels]`.
   * Resolves with the raw pipeline output.
   */
  run: (
    image: RawImage,
    args?: unknown[],
    opts?: RunImageOptions,
  ) => Promise<unknown>;
  /** Start the download (no-op unless idle) — see model-page-pattern.md §3. */
  load: () => void;
  /** Re-attempt a failed load. */
  retry: (overrides?: Record<string, unknown>) => void;
  /** Abandon a load in flight, returning to `idle`. */
  cancel: () => void;
}

export function useVisionPipeline(
  task: VisionTask,
  model: string,
  autoLoad = false,
  /** Per-backend precision override from the catalogue entry, if it has one. */
  dtypes?: VisionModel["dtypes"],
): UseVisionPipelineResult {
  const worker = useModelWorker<unknown>({
    createWorker: createVisionWorker,
    key: `${task}:${model}`,
    loadMessage: dtypes ? { task, model, dtypes } : { task, model },
    autoLoad,
    notReadyMessage: "Vision worker not ready",
  });

  const { run: post } = worker;
  const run = useCallback(
    (
      image: RawImage,
      args?: unknown[],
      { consume = false }: RunImageOptions = {},
    ): Promise<unknown> => {
      const payload = toPayload(image, { copy: !consume });
      return post({ image: payload, args }, transferablesOf(payload));
    },
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
