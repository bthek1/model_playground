import { useCallback } from "react";

import type { PipelineProgress } from "@/audio/pipelineTypes";
import {
  DEFAULT_TTS_MODEL,
  type TtsAudio,
  type TtsRunOpts,
} from "@/audio/tts";
import { createTtsWorker } from "@/audio/ttsClient";
import { useModelWorker } from "@/model/useModelWorker";

export type TtsStatus = "idle" | "loading" | "ready" | "error";

export interface UseTtsResult {
  status: TtsStatus;
  loading: boolean;
  ready: boolean;
  progress: PipelineProgress | null;
  /** The chosen backend once loaded (`"webgpu"` | `"wasm"`). */
  backend: string | null;
  /** Latest synthesised audio (also resolved by `synthesize`). */
  result: TtsAudio | null;
  running: boolean;
  error: string | null;
  /** Synthesise audio from text; resolves with the samples + sample rate. */
  synthesize: (text: string, opts?: TtsRunOpts) => Promise<TtsAudio>;
  /** Start the download (no-op unless idle) — see model-page-pattern.md §3. */
  load: () => void;
  /** Re-attempt a failed load. */
  retry: () => void;
}

/**
 * Loads a text→audio model (Kokoro via kokoro-js, MMS/SpeechT5 via the
 * Transformers.js `text-to-speech` pipeline, or MusicGen) in the TTS Web Worker
 * and exposes synthesis as a promise. The worker lifecycle, id correlation and
 * state machine live in `useModelWorker`.
 */
export function useTts(
  model: string = DEFAULT_TTS_MODEL,
  autoLoad = true,
): UseTtsResult {
  const worker = useModelWorker<TtsAudio>({
    createWorker: createTtsWorker,
    key: model,
    loadMessage: { model },
    autoLoad,
    notReadyMessage: "TTS worker not ready",
  });

  const { run } = worker;
  const synthesize = useCallback(
    (text: string, opts?: TtsRunOpts): Promise<TtsAudio> =>
      run({ text, opts }),
    [run],
  );

  return {
    status: worker.status,
    loading: worker.loading,
    ready: worker.ready,
    progress: worker.progress,
    backend: worker.backend,
    result: worker.result,
    running: worker.running,
    error: worker.error,
    synthesize,
    load: worker.load,
    retry: worker.retry,
  };
}
