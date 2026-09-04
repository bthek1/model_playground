import { useCallback } from "react";

import { createAsrWorker } from "@/audio/asrClient";
import {
  DEFAULT_ASR_MODEL,
  type AsrProgress,
  type AsrResult,
  type AsrRunArgs,
} from "@/audio/types";
import type { LoadProgress } from "@/model/progress";
import { useModelWorker } from "@/model/useModelWorker";

export type AsrStatus = "idle" | "loading" | "ready" | "error";

export interface UseAsrResult {
  status: AsrStatus;
  /** True before the user has asked for the weights. */
  idle: boolean;
  /** True until the pipeline reports `ready`. */
  loading: boolean;
  ready: boolean;
  /** Latest model download/load progress event, if any. */
  progress: AsrProgress | null;
  /** Aggregate load progress across every file. Null outside `loading`. */
  loadProgress: LoadProgress | null;
  /** Duration of the load that produced `ready`, in ms. */
  loadedInMs: number | null;
  /** The chosen backend once loaded (`"webgpu"` | `"wasm"`). */
  backend: string | null;
  /** Latest transcription result. */
  result: AsrResult | null;
  running: boolean;
  error: string | null;
  /** Transcribe mono 16 kHz Float32 audio; resolves with the transcript. */
  transcribe: (audio: Float32Array, args?: AsrRunArgs) => Promise<AsrResult>;
  /** Start the download (no-op unless idle) — see model-page-pattern.md §3. */
  load: () => void;
  /** Re-attempt a failed load. */
  retry: () => void;
  /** Abandon a load in flight, returning to `idle`. */
  cancel: () => void;
}

/**
 * Loads an ASR model in a Web Worker and exposes transcription as a promise.
 * The worker lifecycle, id correlation and state machine live in
 * `useModelWorker`; this wrapper only adds the ASR run payload and its
 * task-named alias (`transcribe`).
 */
export function useAsr(
  model: string = DEFAULT_ASR_MODEL,
  autoLoad = true,
): UseAsrResult {
  const worker = useModelWorker<AsrResult>({
    createWorker: createAsrWorker,
    key: model,
    loadMessage: { model },
    autoLoad,
    notReadyMessage: "ASR worker not ready",
  });

  const { run } = worker;
  const transcribe = useCallback(
    (audio: Float32Array, args?: AsrRunArgs): Promise<AsrResult> =>
      // Transfer the audio buffer to avoid a copy; the caller's array is consumed.
      run({ audio, args }, [audio.buffer]),
    [run],
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
    result: worker.result,
    running: worker.running,
    error: worker.error,
    transcribe,
    load: worker.load,
    retry: worker.retry,
    cancel: worker.cancel,
  };
}
