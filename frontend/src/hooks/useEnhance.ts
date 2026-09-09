import { useCallback } from "react";

import type { Backend } from "@/model/backend";
import { createEnhanceWorker } from "@/audio/enhance/enhanceClient";
import {
  DEFAULT_ENHANCE_MODEL,
  type EnhanceResult,
} from "@/audio/enhance/types";
import { useModelWorker } from "@/model/useModelWorker";
import type { ModelTask } from "@/model/types";

/**
 * Speech enhancement (audio → audio) with DeepFilterNet3 in a Web Worker.
 *
 * Returns the shared `ModelTask` contract verbatim — `run`, not a task-named
 * alias like `useTts`'s `synthesize` or `useAsr`'s `transcribe`. Those predate
 * the contract in docs/standards/model-page-pattern.md §3; this hook is what the
 * rest are being migrated towards, so don't add an `enhance()` alias here.
 *
 * `autoLoad` defaults to **false**: the weights are the user's bandwidth, so the
 * route quotes the size first and downloads on an explicit action.
 */
export function useEnhance(
  model: string = DEFAULT_ENHANCE_MODEL,
  autoLoad = false,
): ModelTask<Float32Array, EnhanceResult> {
  const worker = useModelWorker<EnhanceResult>({
    createWorker: createEnhanceWorker,
    key: model,
    loadMessage: { model },
    autoLoad,
    notReadyMessage: "Enhancement worker not ready",
  });

  const { run: post } = worker;
  const run = useCallback(
    (audio: Float32Array): Promise<EnhanceResult> =>
      // Transfer rather than copy: a minute of 48 kHz mono is 11 MB. The
      // caller's array is detached, which is why the route keeps its own copy
      // of the input for A/B playback.
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
