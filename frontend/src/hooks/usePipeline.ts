import { useCallback, useEffect, useRef, useState } from "react";

import { createPipelineWorker } from "@/audio/pipelineClient";
import type {
  PipelineProgress,
  PipelineResponse,
  PipelineTask,
} from "@/audio/pipelineTypes";

export type PipelineStatus = "loading" | "ready" | "error";

export interface UsePipelineResult {
  status: PipelineStatus;
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
}

/**
 * Loads a Transformers.js pipeline for `task`/`model` in the generic pipeline
 * worker and exposes `run` as a promise. The worker is created once per
 * (task, model) and terminated on unmount / change (freeing the model + backend
 * context). Requests are correlated by an incrementing id so overlapping calls
 * resolve independently. ASR uses its own hook (`useAsr`) for the real-time loop;
 * discriminative tasks build on this.
 */
export function usePipeline(
  task: PipelineTask,
  model: string,
): UsePipelineResult {
  const workerRef = useRef<Worker | null>(null);
  const nextId = useRef(0);
  const pending = useRef(
    new Map<
      number,
      { resolve: (r: unknown) => void; reject: (e: Error) => void }
    >(),
  );

  const [status, setStatus] = useState<PipelineStatus>("loading");
  const [progress, setProgress] = useState<PipelineProgress | null>(null);
  const [backend, setBackend] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setStatus("loading");
    setProgress(null);
    setBackend(null);
    setError(null);

    const worker = createPipelineWorker();
    workerRef.current = worker;
    const inflight = pending.current;

    worker.onmessage = (event: MessageEvent<PipelineResponse>) => {
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

    worker.postMessage({ type: "load", task, model });

    return () => {
      worker.terminate();
      workerRef.current = null;
      inflight.forEach(({ reject }) => reject(new Error("Worker terminated")));
      inflight.clear();
    };
  }, [task, model]);

  const run = useCallback(
    (input: Float32Array, args?: unknown[]): Promise<unknown> => {
      const worker = workerRef.current;
      if (!worker) return Promise.reject(new Error("Pipeline worker not ready"));
      const id = ++nextId.current;
      setRunning(true);
      setError(null);
      return new Promise<unknown>((resolve, reject) => {
        pending.current.set(id, { resolve, reject });
        // Transfer the input buffer to avoid a copy; the caller's array is consumed.
        worker.postMessage({ type: "run", id, input, args }, [input.buffer]);
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
    running,
    error,
    run,
  };
}
