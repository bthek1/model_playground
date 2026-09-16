// Owns the /link-prediction route's worker, its split dataset and a streaming
// run.
//
// The shape of useGraphTraining, with one real difference: the **held-out
// fraction is a load parameter**, not a training one. Changing it is a different
// split, which is a different graph and a different layout, so `load(fraction)`
// re-splits where `/graph`'s `load()` is idempotent. The page says so rather
// than quietly spending a second of layout on a slider drag.
//
// There is no per-epoch picture here either. `/graph` streams predictions so the
// colours settle while it trains; the candidates on this page are scored once
// after the last epoch, so what streams is the metrics and nothing else.

import { useCallback, useEffect, useRef, useState } from "react";

import type { GnnArch } from "@/webgpu/gnn";
import type { GraphBackend } from "@/webgpu/graphSession";
import type { LinkMetrics } from "@/webgpu/linkPredictor";
import type {
  LinkLoadOptions,
  LinkSummary,
  LinkTrainRequest,
  LinkTrainResult,
} from "@/webgpu/linkSession";
import {
  createWebGPUWorker,
  loadLinkGraphInWorker,
  trainLinkInWorker,
  type LinkTrainingHandle,
} from "@/webgpu/workerClient";

export type LinkLoadStatus = "idle" | "loading" | "ready" | "error";

/** One completed run, so two architectures can be compared on the same split. */
export interface LinkRunPoint {
  arch: GnnArch;
  layers: number;
  testAuc: number;
  testAp: number;
}

export interface LinkPredictionState {
  status: LinkLoadStatus;
  summary: LinkSummary | null;
  loadError: string | null;
  backend: GraphBackend | null;
  load: (options?: LinkLoadOptions) => void;
  retry: () => void;

  training: boolean;
  metrics: LinkMetrics[];
  /** Everything the last completed run produced; null until one finishes. */
  result: LinkTrainResult | null;
  elapsedMs: number | null;
  trainError: string | null;
  history: LinkRunPoint[];
  start: (request: LinkTrainRequest) => void;
  stop: () => void;
}

export function useLinkPrediction(): LinkPredictionState {
  const workerRef = useRef<Worker | null>(null);
  const handleRef = useRef<LinkTrainingHandle | null>(null);

  const [status, setStatus] = useState<LinkLoadStatus>("idle");
  const statusRef = useRef(status);
  statusRef.current = status;
  const [summary, setSummary] = useState<LinkSummary | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [training, setTraining] = useState(false);
  const [metrics, setMetrics] = useState<LinkMetrics[]>([]);
  const [result, setResult] = useState<LinkTrainResult | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [history, setHistory] = useState<LinkRunPoint[]>([]);

  // rAF-batched, as /graph is: a fast run emits metrics faster than React can
  // usefully re-render, and only the newest one is ever drawn.
  const metricsRef = useRef<LinkMetrics[]>([]);
  const flushScheduled = useRef(false);
  const scheduleFlush = useCallback(() => {
    if (flushScheduled.current) return;
    flushScheduled.current = true;
    requestAnimationFrame(() => {
      flushScheduled.current = false;
      setMetrics(metricsRef.current.slice());
    });
  }, []);

  useEffect(() => {
    return () => {
      handleRef.current?.cancel();
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  // A ref rather than `status`: two clicks in one React batch would both read
  // the same stale state and both start a split plus a layout.
  const loadingRef = useRef(false);

  const load = useCallback((options: LinkLoadOptions = {}) => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    setStatus("loading");
    setLoadError(null);

    workerRef.current ??= createWebGPUWorker();
    loadLinkGraphInWorker(workerRef.current, options)
      .then((loaded) => {
        setSummary(loaded);
        setStatus("ready");
        // A new split is a new graph: the run that was on screen was scored
        // against edges this one does not hold out, so it is not an answer to
        // the question now being asked.
        setResult(null);
        setMetrics([]);
        metricsRef.current = [];
        setHistory([]);
        loadingRef.current = false;
      })
      .catch((error: unknown) => {
        loadingRef.current = false;
        setLoadError(error instanceof Error ? error.message : String(error));
        setStatus("error");
      });
  }, []);

  const retry = useCallback(() => {
    loadingRef.current = false;
    setStatus("idle");
    setLoadError(null);
  }, []);

  const start = useCallback(
    (request: LinkTrainRequest) => {
      const worker = workerRef.current;
      if (!worker || statusRef.current !== "ready") {
        setTrainError("Load the graph first.");
        return;
      }

      metricsRef.current = [];
      setMetrics([]);
      setResult(null);
      setElapsedMs(null);
      setTrainError(null);
      setTraining(true);

      const handle = trainLinkInWorker(worker, request, (m) => {
        metricsRef.current.push(m);
        scheduleFlush();
      });
      handleRef.current = handle;

      handle.promise
        .then((completed) => {
          // Stopped before the first epoch: nothing to show and nothing wrong.
          const final = completed.metrics;
          if (!final) return;
          setMetrics(metricsRef.current.slice());
          setResult(completed);
          setElapsedMs(completed.elapsedMs);
          setHistory((past) => [
            ...past.filter(
              (p) => !(p.arch === request.arch && p.layers === request.layers),
            ),
            {
              arch: request.arch,
              layers: request.layers,
              testAuc: final.testAuc,
              testAp: final.testAp,
            },
          ]);
        })
        .catch((error: unknown) => {
          setTrainError(error instanceof Error ? error.message : String(error));
        })
        .finally(() => {
          setTraining(false);
          handleRef.current = null;
        });
    },
    [scheduleFlush],
  );

  const stop = useCallback(() => handleRef.current?.cancel(), []);

  return {
    status,
    summary,
    loadError,
    backend: summary?.backend ?? null,
    load,
    retry,
    training,
    metrics,
    result,
    elapsedMs,
    trainError,
    history,
    start,
    stop,
  };
}
