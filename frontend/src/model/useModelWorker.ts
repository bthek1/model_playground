// The worker plumbing every task hook used to duplicate: creation and teardown,
// the id-correlated pending table, the progress/ready/result/error switch, and
// the two state machines from docs/standards/model-page-pattern.md §2.
//
// `useTts`, `useAsr` and `usePipeline` are thin typed wrappers around this.
// Anything genuinely task-specific (ASR's capture loop, TTS's voice option)
// stays in the wrapper.

import { useCallback, useEffect, useRef, useState } from "react";

import { reportDownload, reportInflight } from "@/telemetry/activity";

import {
  initialProgress,
  reduceProgress,
  summarize,
  type LoadProgress,
  type ProgressState,
} from "./progress";
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
   * Start loading as soon as the hook mounts. **Defaults to `false`, and every
   * task hook leaves it there.** Weights are the user's bandwidth and the tab's
   * memory: the LOAD button is the only thing allowed to spend either
   * (model-page-pattern.md §1.2). The option survives for a hypothetical
   * compile-only task with nothing to download — not as a convenience.
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
  /** The raw worker event — the last one seen, unaggregated. */
  progress: ModelProgress | null;
  /** Aggregate download/warm-up progress across every file. */
  loadProgress: LoadProgress | null;
  /** How long the load that produced `ready` took, in ms. Null until ready. */
  loadedInMs: number | null;
  backend: string | null;
  running: boolean;
  result: TResult | null;
  error: string | null;
  load: () => void;
  /**
   * Re-attempt a failed load. `overrides` are merged into the `load` message —
   * that is how "Retry on CPU" pins a backend without the page reaching into
   * the worker protocol.
   */
  retry: (overrides?: Record<string, unknown>) => void;
  /** Abandon a load in flight and return to `idle`. No-op unless loading. */
  cancel: () => void;
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
  autoLoad = false,
  notReadyMessage = "Model worker not ready",
}: UseModelWorkerOptions): UseModelWorkerResult<TResult> {
  const workerRef = useRef<Worker | null>(null);
  const nextId = useRef(0);
  const pending = useRef(new Map<number, Pending<TResult>>());

  const [status, setStatus] = useState<ModelStatus>(
    autoLoad ? "loading" : "idle",
  );
  const [progress, setProgress] = useState<ModelProgress | null>(null);
  const [progressState, setProgressState] =
    useState<ProgressState>(initialProgress);
  const [loadedInMs, setLoadedInMs] = useState<number | null>(null);
  // Re-rendered once a second while loading, purely so the elapsed counter in
  // the LOAD slot moves. Progress events alone can be minutes apart on a slow
  // link, which is exactly when the user most needs to see something ticking.
  const [elapsedMs, setElapsedMs] = useState(0);
  const startedAt = useRef<number | null>(null);
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
  const start = useCallback((overrides?: Record<string, unknown>) => {
    const { createWorker: spawn, loadMessage: message } = optionsRef.current;
    const worker = spawn();
    workerRef.current = worker;
    const table = pending.current;

    worker.onmessage = (event: MessageEvent<ModelResponse<TResult>>) => {
      const data = event.data;
      switch (data.type) {
        case "progress":
          setProgress(data.progress);
          setProgressState((prev) => reduceProgress(prev, data.progress));
          break;
        case "ready":
          setStatus("ready");
          setBackend(data.backend);
          setLoadedInMs(
            startedAt.current == null ? null : Date.now() - startedAt.current,
          );
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
    setProgressState(initialProgress);
    setLoadedInMs(null);
    setElapsedMs(0);
    startedAt.current = Date.now();
    setError(null);
    worker.postMessage({ type: "load", ...message, ...overrides });
  }, []);

  useEffect(() => {
    setBackend(null);
    setError(null);
    setProgress(null);
    setProgressState(initialProgress);
    setLoadedInMs(null);
    setElapsedMs(0);
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

  /**
   * Re-attempt a failed load without changing the selected model. `overrides`
   * ride along on the `load` message — e.g. `{ backend: "wasm" }` after a GPU
   * failure. §6 still holds: the backend is resolved once per worker, the user
   * has just chosen it instead of the probe.
   */
  const retry = useCallback(
    (overrides?: Record<string, unknown>) => {
      if (statusRef.current !== "error") return;
      workerRef.current?.terminate();
      workerRef.current = null;
      start(overrides);
    },
    [start],
  );

  /**
   * Abandon a download in flight. Machine A gains one transition,
   * `loading --cancel()--> idle`; the partial download stays in the browser
   * cache, so re-loading later picks up where this left off.
   */
  const cancel = useCallback(() => {
    if (statusRef.current !== "loading") return;
    workerRef.current?.terminate();
    workerRef.current = null;
    pending.current.forEach(({ reject }) => reject(new Error("Load cancelled")));
    pending.current.clear();
    setInflight(0);
    startedAt.current = null;
    setProgress(null);
    setProgressState(initialProgress);
    setError(null);
    setStatus("idle");
  }, []);

  // The elapsed-time ticker. Only runs while a load is actually in flight.
  useEffect(() => {
    if (status !== "loading") return;
    const started = startedAt.current ?? Date.now();
    const id = setInterval(() => setElapsedMs(Date.now() - started), 1000);
    return () => clearInterval(id);
  }, [status]);

  const loadProgress =
    status === "loading" ? summarize(progressState, elapsedMs) : null;

  // The system panel's two app-derived numbers. Machine B's in-flight count and
  // the byte aggregate above are already here and nowhere else, so this hook
  // reports them rather than the panel measuring them a second time (it
  // couldn't: an inference is invisible from outside the hook that started it).
  // Both clean up to "nothing happening" — a route unmounted mid-run must not
  // leave the panel reading busy for the rest of the session.
  const loadedBytes = loadProgress?.loaded ?? 0;
  const totalBytes = loadProgress?.total ?? 0;

  useEffect(() => {
    reportInflight(key, inflight);
    return () => reportInflight(key, 0);
  }, [key, inflight]);

  useEffect(() => {
    reportDownload(key, totalBytes > 0 ? { loadedBytes, totalBytes } : null);
    return () => reportDownload(key, null);
  }, [key, loadedBytes, totalBytes]);

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
    loadProgress,
    loadedInMs,
    backend,
    running: inflight > 0,
    result,
    error,
    load,
    retry,
    cancel,
    run,
  };
}
