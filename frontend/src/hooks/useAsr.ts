import { useCallback, useEffect, useRef, useState } from "react";

import { createAsrWorker } from "@/audio/asrClient";
import {
  DEFAULT_ASR_MODEL,
  type AsrProgress,
  type AsrResponse,
  type AsrResult,
  type AsrRunArgs,
} from "@/audio/types";

export type AsrStatus = "loading" | "ready" | "error";

export interface UseAsrResult {
  status: AsrStatus;
  /** True until the pipeline reports `ready`. */
  loading: boolean;
  ready: boolean;
  /** Latest model download/load progress event, if any. */
  progress: AsrProgress | null;
  /** The chosen backend once loaded (`"webgpu"` | `"wasm"`). */
  backend: string | null;
  /** Latest transcription result. */
  result: AsrResult | null;
  running: boolean;
  error: string | null;
  /** Transcribe mono 16 kHz Float32 audio; resolves with the transcript. */
  transcribe: (audio: Float32Array, args?: AsrRunArgs) => Promise<AsrResult>;
}

/**
 * Loads an ASR model in a Web Worker and exposes transcription as a promise.
 * The worker is created once per `model`, and terminated on unmount / model
 * change (freeing the model + backend context). Transcription requests are
 * correlated by an incrementing id so overlapping calls resolve independently.
 */
export function useAsr(model: string = DEFAULT_ASR_MODEL): UseAsrResult {
  const workerRef = useRef<Worker | null>(null);
  const nextId = useRef(0);
  const pending = useRef(
    new Map<
      number,
      { resolve: (r: AsrResult) => void; reject: (e: Error) => void }
    >(),
  );

  const [status, setStatus] = useState<AsrStatus>("loading");
  const [progress, setProgress] = useState<AsrProgress | null>(null);
  const [backend, setBackend] = useState<string | null>(null);
  const [result, setResult] = useState<AsrResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setStatus("loading");
    setProgress(null);
    setBackend(null);
    setError(null);

    const worker = createAsrWorker();
    workerRef.current = worker;
    const inflight = pending.current;

    worker.onmessage = (event: MessageEvent<AsrResponse>) => {
      const data = event.data;
      switch (data.type) {
        case "progress":
          setProgress(data.progress);
          break;
        case "ready":
          setStatus("ready");
          setBackend(data.backend);
          break;
        case "result": {
          setResult(data.result);
          setRunning(false);
          inflight.get(data.id)?.resolve(data.result);
          inflight.delete(data.id);
          break;
        }
        case "error": {
          if (data.id != null) {
            setRunning(false);
            inflight.get(data.id)?.reject(new Error(data.error));
            inflight.delete(data.id);
          } else {
            // A load-time error — the model never became usable.
            setStatus("error");
          }
          setError(data.error);
          break;
        }
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

  const transcribe = useCallback(
    (audio: Float32Array, args?: AsrRunArgs): Promise<AsrResult> => {
      const worker = workerRef.current;
      if (!worker) return Promise.reject(new Error("ASR worker not ready"));
      const id = ++nextId.current;
      setRunning(true);
      setError(null);
      return new Promise<AsrResult>((resolve, reject) => {
        pending.current.set(id, { resolve, reject });
        // Transfer the audio buffer to avoid a copy; the caller's array is consumed.
        worker.postMessage({ type: "run", id, audio, args }, [audio.buffer]);
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
    transcribe,
  };
}
