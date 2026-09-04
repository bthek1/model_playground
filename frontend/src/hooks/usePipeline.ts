import { useCallback } from "react";

import { createPipelineWorker } from "@/audio/pipelineClient";
import type { PipelineProgress, PipelineTask } from "@/audio/pipelineTypes";
import { useModelWorker } from "@/model/useModelWorker";

export type PipelineStatus = "idle" | "loading" | "ready" | "error";

export interface UsePipelineResult {
  status: PipelineStatus;
  /** True before the user has asked for the weights. */
  idle: boolean;
  loading: boolean;
  ready: boolean;
  progress: PipelineProgress | null;
  /** The chosen backend once loaded (`"webgpu"` | `"wasm"`). */
  backend: string | null;
  running: boolean;
  error: string | null;
  /**
   * Run the loaded pipeline on `input` (mono 16 kHz Float32). `args` are spread
   * positionally onto the pipeline call, so callers pass e.g. `[{ top_k: 6 }]`
   * or `[candidateLabels]`. Resolves with the raw pipeline output.
   */
  run: (input: Float32Array, args?: unknown[]) => Promise<unknown>;
  /** Start the download (no-op unless idle) — see model-page-pattern.md §3. */
  load: () => void;
  /** Re-attempt a failed load. */
  retry: () => void;
}

/**
 * Loads a Transformers.js pipeline for `task`/`model` in the generic pipeline
 * worker and exposes `run` as a promise. The worker lifecycle, id correlation
 * and state machine live in `useModelWorker`. ASR uses its own hook (`useAsr`)
 * for the real-time loop; discriminative tasks build on this.
 */
export function usePipeline(
  task: PipelineTask,
  model: string,
  autoLoad = true,
): UsePipelineResult {
  const worker = useModelWorker<unknown>({
    createWorker: createPipelineWorker,
    key: `${task}:${model}`,
    loadMessage: { task, model },
    autoLoad,
    notReadyMessage: "Pipeline worker not ready",
  });

  const { run: post } = worker;
  const run = useCallback(
    (input: Float32Array, args?: unknown[]): Promise<unknown> =>
      // Transfer the input buffer to avoid a copy; the caller's array is consumed.
      post({ input, args }, [input.buffer]),
    [post],
  );

  return {
    status: worker.status,
    idle: worker.idle,
    loading: worker.loading,
    ready: worker.ready,
    progress: worker.progress,
    backend: worker.backend,
    running: worker.running,
    error: worker.error,
    run,
    load: worker.load,
    retry: worker.retry,
  };
}
