import { useCallback } from "react";

import type { Backend } from "@/audio/backend";
import { createVadWorker } from "@/audio/vad/vadClient";
import { DEFAULT_VAD_MODEL, type VadResult } from "@/audio/vad/types";
import { useModelWorker } from "@/model/useModelWorker";
import type { ModelTask } from "@/model/types";

/**
 * Voice activity detection (audio → per-frame speech probability) in a Web
 * Worker.
 *
 * Returns the shared `ModelTask` contract verbatim — `run`, not a task-named
 * alias like `useTts`'s `synthesize` (docs/standards/model-page-pattern.md §3).
 *
 * `autoLoad` defaults to **false**: even a 2 MB model is the user's bandwidth,
 * and the route quotes the size before it spends any.
 */
export function useVad(
  model: string = DEFAULT_VAD_MODEL,
  autoLoad = false,
): ModelTask<Float32Array, VadResult> {
  const worker = useModelWorker<VadResult>({
    createWorker: createVadWorker,
    key: model,
    loadMessage: { model },
    autoLoad,
    notReadyMessage: "VAD worker not ready",
  });

  const { run: post } = worker;
  const run = useCallback(
    (audio: Float32Array): Promise<VadResult> =>
      // Transfer rather than copy — the caller's array is detached, which is why
      // the route keeps its own copy for the waveform.
      post({ audio }, [audio.buffer]),
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
    backend: worker.backend as Backend | null,
    load: worker.load,
    retry: worker.retry,
    cancel: worker.cancel,
    run,
    running: worker.running,
    result: worker.result,
    error: worker.error,
  };
}
