import { useCallback, useEffect, useRef, useState } from "react";

import { createTtsWorker } from "@/audio/ttsClient";
import type { PipelineProgress } from "@/audio/pipelineTypes";
import {
  DEFAULT_TTS_MODEL,
  type TtsAudio,
  type TtsResponse,
  type TtsRunOpts,
} from "@/audio/tts";

export type TtsStatus = "loading" | "ready" | "error";

export interface UseTtsResult {
  status: TtsStatus;
  loading: boolean;
  ready: boolean;
  progress: PipelineProgress | null;
  /** The chosen backend once loaded (`"webgpu"` | `"wasm"`). */
  backend: string | null;
  /** Latest synthesised speech (also resolved by `synthesize`). */
  result: TtsAudio | null;
  running: boolean;
  error: string | null;
  /** Synthesise speech from text; resolves with the audio + sample rate. */
  synthesize: (text: string, opts?: TtsRunOpts) => Promise<TtsAudio>;
}

/**
 * Loads a TTS model (Kokoro via kokoro-js, or MMS/SpeechT5 via the
 * Transformers.js `text-to-speech` pipeline) in the TTS Web Worker and exposes
 * synthesis as a promise. The worker is created once per `model` and terminated
 * on unmount / model change (freeing the model + backend context). Requests are
 * correlated by an incrementing id so overlapping calls resolve independently.
 */
export function useTts(model: string = DEFAULT_TTS_MODEL): UseTtsResult {
  const workerRef = useRef<Worker | null>(null);
  const nextId = useRef(0);
  const pending = useRef(
    new Map<
      number,
      { resolve: (r: TtsAudio) => void; reject: (e: Error) => void }
    >(),
  );

  const [status, setStatus] = useState<TtsStatus>("loading");
  const [progress, setProgress] = useState<PipelineProgress | null>(null);
  const [backend, setBackend] = useState<string | null>(null);
  const [result, setResult] = useState<TtsAudio | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setStatus("loading");
    setProgress(null);
    setBackend(null);
    setError(null);

    const worker = createTtsWorker();
    workerRef.current = worker;
    const inflight = pending.current;

    worker.onmessage = (event: MessageEvent<TtsResponse>) => {
      const data = event.data;
      switch (data.type) {
        case "progress":
          setProgress(data.progress);
          break;
        case "ready":
          setStatus("ready");
          setBackend(data.backend);
          break;
        case "result":
          setResult(data.result);
          setRunning(false);
          inflight.get(data.id)?.resolve(data.result);
          inflight.delete(data.id);
          break;
        case "error":
          if (data.id != null) {
            setRunning(false);
            inflight.get(data.id)?.reject(new Error(data.error));
            inflight.delete(data.id);
          } else {
            setStatus("error");
          }
          setError(data.error);
          break;
      }
    };

    worker.postMessage({ type: "load", model });

    return () => {
      worker.terminate();
      workerRef.current = null;
      inflight.forEach(({ reject }) => reject(new Error("Worker terminated")));
      inflight.clear();
    };
  }, [model]);

  const synthesize = useCallback(
    (text: string, opts?: TtsRunOpts): Promise<TtsAudio> => {
      const worker = workerRef.current;
      if (!worker) return Promise.reject(new Error("TTS worker not ready"));
      const id = ++nextId.current;
      setRunning(true);
      setError(null);
      return new Promise<TtsAudio>((resolve, reject) => {
        pending.current.set(id, { resolve, reject });
        worker.postMessage({ type: "run", id, text, opts });
      });
    },
    [],
  );

  return {
    status,
    loading: status === "loading",
    ready: status === "ready",
    progress,
    backend,
    result,
    running,
    error,
    synthesize,
  };
}
