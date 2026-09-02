// The worker plumbing every task hook used to duplicate: creation and teardown,
// the id-correlated pending table, the progress/ready/result/error switch, and
// the two state machines from docs/standards/model-page-pattern.md §2.
//
// `useTts`, `useAsr` and `usePipeline` are thin typed wrappers around this.
// Anything genuinely task-specific (ASR's capture loop, TTS's voice option)
// stays in the wrapper.

import { useCallback, useEffect, useRef, useState } from "react";

import type { ModelProgress, ModelResponse, ModelStatus } from "./types";

export interface UseModelWorkerOptions {
  /** Spawns the Worker. Called on load, and again on retry. */
  createWorker: () => Worker;
  /**
   * Worker identity. Changing it tears the current worker down and resets the
   * machine — typically the model id, or `task + model`.
   */
  key: string;
  /** Merged into the `{ type: "load" }` message. */
  loadMessage: Record<string, unknown>;
  /**
   * Start loading as soon as the hook mounts. `true` preserves the historical
   * behaviour; pages that download weights pass `false` so the user consents
   * first (model-page-pattern.md §1.2). Compile-only tasks keep `true`.
   */
  autoLoad?: boolean;
  /** Rejection message when `run()` is called with no live worker. */
  notReadyMessage?: string;
}

export interface UseModelWorkerResult<TResult> {
  status: ModelStatus;
  idle: boolean;
  loading: boolean;
  ready: boolean;
  progress: ModelProgress | null;
  backend: string | null;
  running: boolean;
  result: TResult | null;
  error: string | null;
  load: () => void;
  retry: () => void;
  /**
   * Post a `run` message and resolve with its correlated result. `payload` is
   * spread into the message, so a caller supplies only its task fields
   * (`{ audio }`, `{ text, opts }`, …).
   */
  run: (
    payload: Record<string, unknown>,
    transfer?: Transferable[],
  ) => Promise<TResult>;
}

interface Pending<TResult> {
  resolve: (r: TResult) => void;
  reject: (e: Error) => void;
}

export function useModelWorker<TResult>({
  createWorker,
  key,
  loadMessage,
  autoLoad = true,
  notReadyMessage = "Model worker not ready",
}: UseModelWorkerOptions): UseModelWorkerResult<TResult> {
  const workerRef = useRef<Worker | null>(null);
  const nextId = useRef(0);
  const pending = useRef(new Map<number, Pending<TResult>>());

  const [status, setStatus] = useState<ModelStatus>(
    autoLoad ? "loading" : "idle",
  );
  const [progress, setProgress] = useState<ModelProgress | null>(null);
  const [backend, setBackend] = useState<string | null>(null);
  const [result, setResult] = useState<TResult | null>(null);
  // A count, not a boolean: with two overlapping requests a boolean reports idle
  // as soon as the first returns, while the second is still running.
  const [inflight, setInflight] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Kept in refs so the worker effect depends only on `key`/`autoLoad` — a new
  // inline `loadMessage` object each render must not respawn the worker.
  const optionsRef = useRef({ createWorker, loadMessage, notReadyMessage });
  optionsRef.current = { createWorker, loadMessage, notReadyMessage };
  const statusRef = useRef(status);
  statusRef.current = status;

  /** Spawn a worker, wire the response switch, and post `load`. */
  const start = useCallback(() => {
    const { createWorker: spawn, loadMessage: message } = optionsRef.current;
    const worker = spawn();
    workerRef.current = worker;
    const table = pending.current;

    worker.onmessage = (event: MessageEvent<ModelResponse<TResult>>) => {
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
          setInflight((n) => Math.max(0, n - 1));
          table.get(data.id)?.resolve(data.result);
          table.delete(data.id);
          break;
        case "error":
          if (data.id != null) {
            // Machine B: one request failed; the model is still loaded.
            setInflight((n) => Math.max(0, n - 1));
            table.get(data.id)?.reject(new Error(data.error));
            table.delete(data.id);
          } else {
            // Machine A: the model never became usable.
            setStatus("error");
          }
          setError(data.error);
          break;
      }
    };

    setStatus("loading");
    setProgress(null);
    setError(null);
    worker.postMessage({ type: "load", ...message });
  }, []);

  useEffect(() => {
    setBackend(null);
    setError(null);
    setProgress(null);
    setStatus(autoLoad ? "loading" : "idle");
    if (autoLoad) start();

    const table = pending.current;
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
      table.forEach(({ reject }) => reject(new Error("Worker terminated")));
      table.clear();
      setInflight(0);
    };
  }, [key, autoLoad, start]);

  /** Start the download. No-op unless idle (§3). */
  const load = useCallback(() => {
    if (statusRef.current !== "idle") return;
    start();
  }, [start]);

  /** Re-attempt a failed load without changing the selected model. */
  const retry = useCallback(() => {
    if (statusRef.current !== "error") return;
    workerRef.current?.terminate();
    workerRef.current = null;
    start();
  }, [start]);

  const run = useCallback(
    (
      payload: Record<string, unknown>,
      transfer?: Transferable[],
    ): Promise<TResult> => {
      const worker = workerRef.current;
      if (!worker) {
        return Promise.reject(new Error(optionsRef.current.notReadyMessage));
      }
      const id = ++nextId.current;
      setInflight((n) => n + 1);
      setError(null);
      return new Promise<TResult>((resolve, reject) => {
        pending.current.set(id, { resolve, reject });
        const message = { type: "run", id, ...payload };
        // Only pass a transfer list when there is one — some workers take none.
        if (transfer) worker.postMessage(message, transfer);
        else worker.postMessage(message);
      });
    },
    [],
  );

  return {
    status,
    idle: status === "idle",
    loading: status === "loading",
    ready: status === "ready",
    progress,
    backend,
    running: inflight > 0,
    result,
    error,
    load,
    retry,
    run,
  };
}
