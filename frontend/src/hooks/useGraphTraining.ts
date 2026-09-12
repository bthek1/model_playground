// Owns the /graph route's worker, its dataset, and a streaming GNN training run.
//
// The shape follows useLinearTraining: a worker held in a ref, metrics batched to
// animation frames so a fast run cannot flood React with renders, and a cancel
// handle for the transport. What it adds is the **depth history** — the point of
// the page — which accumulates one entry per (architecture, depth) the user has
// actually trained, so dragging the depth slider builds the oversmoothing curve
// instead of discarding each previous answer.

import { useCallback, useEffect, useRef, useState } from "react";

import type { GnnArch, GnnMetrics } from "@/webgpu/gnn";
import type {
  GraphBackend,
  GraphSummary,
  GraphTrainRequest,
} from "@/webgpu/graphSession";
import {
  createWebGPUWorker,
  type GraphTrainingHandle,
  loadGraphInWorker,
  trainGraphInWorker,
} from "@/webgpu/workerClient";

/** Machine A of the page pattern: the dataset and the device, not a download. */
export type GraphLoadStatus = "idle" | "loading" | "ready" | "error";

/** One completed run, keyed by the two things the depth chart plots against. */
export interface DepthPoint {
  arch: GnnArch;
  layers: number;
  testAcc: number;
  smoothness: number;
  deadFraction: number;
}

export interface GraphTrainingState {
  status: GraphLoadStatus;
  summary: GraphSummary | null;
  loadError: string | null;
  backend: GraphBackend | null;
  load: () => void;
  retry: () => void;

  training: boolean;
  /** Per-epoch metrics for the run in progress (or the last one). */
  metrics: GnnMetrics[];
  /** The class each node is currently predicted to be; drives the canvas. */
  predictions: Uint8Array | null;
  elapsedMs: number | null;
  trainError: string | null;
  /** One point per (architecture, depth) trained this session. */
  depthHistory: DepthPoint[];
  start: (request: GraphTrainRequest) => void;
  stop: () => void;
}

export function useGraphTraining(): GraphTrainingState {
  const workerRef = useRef<Worker | null>(null);
  const handleRef = useRef<GraphTrainingHandle | null>(null);

  const [status, setStatus] = useState<GraphLoadStatus>("idle");
  // `start` reads the status without depending on it, so its identity is stable
  // across a load — a changing callback would remount the transport controls.
  const statusRef = useRef(status);
  statusRef.current = status;
  const [summary, setSummary] = useState<GraphSummary | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [training, setTraining] = useState(false);
  const [metrics, setMetrics] = useState<GnnMetrics[]>([]);
  const [predictions, setPredictions] = useState<Uint8Array | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [depthHistory, setDepthHistory] = useState<DepthPoint[]>([]);

  // rAF-batched streaming. A 200-epoch run on the GPU emits faster than React
  // can usefully re-render, and the canvas only needs the newest frame anyway.
  const metricsRef = useRef<GnnMetrics[]>([]);
  const pendingPredictions = useRef<Uint8Array | null>(null);
  const flushScheduled = useRef(false);
  const scheduleFlush = useCallback(() => {
    if (flushScheduled.current) return;
    flushScheduled.current = true;
    requestAnimationFrame(() => {
      flushScheduled.current = false;
      setMetrics(metricsRef.current.slice());
      if (pendingPredictions.current) setPredictions(pendingPredictions.current);
    });
  }, []);

  useEffect(() => {
    return () => {
      handleRef.current?.cancel();
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  // Guards `load` against re-entry. A ref rather than the `status` state because
  // two clicks in one React batch would both read the same stale `status` and
  // both start a fetch — and the layout behind it is the second most expensive
  // thing on the page.
  const loadingRef = useRef(false);

  const load = useCallback(() => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    setStatus("loading");
    setLoadError(null);

    workerRef.current ??= createWebGPUWorker();
    loadGraphInWorker(workerRef.current)
      .then((result) => {
        setSummary(result);
        setStatus("ready");
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
    (request: GraphTrainRequest) => {
      const worker = workerRef.current;
      if (!worker || statusRef.current !== "ready") {
        setTrainError("Load the graph first.");
        return;
      }

      metricsRef.current = [];
      pendingPredictions.current = null;
      setMetrics([]);
      setPredictions(null);
      setElapsedMs(null);
      setTrainError(null);
      setTraining(true);

      const handle = trainGraphInWorker(worker, request, (m, pred) => {
        metricsRef.current.push(m);
        pendingPredictions.current = pred;
        scheduleFlush();
      });
      handleRef.current = handle;

      handle.promise
        .then((result) => {
          // Stopped before the first epoch finished: there is nothing to show,
          // and nothing went wrong. Leave the panel as it was.
          const final = result.metrics;
          if (!final) return;
          setMetrics(metricsRef.current.slice());
          setPredictions(result.predictions);
          setElapsedMs(result.elapsedMs);
          // One point per (architecture, depth): re-running the same pair
          // replaces its entry rather than stacking duplicates on the chart.
          setDepthHistory((history) => [
            ...history.filter(
              (p) => !(p.arch === request.arch && p.layers === request.layers),
            ),
            {
              arch: request.arch,
              layers: request.layers,
              testAcc: final.testAcc,
              smoothness: final.smoothness,
              deadFraction: final.deadFraction,
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
    predictions,
    elapsedMs,
    trainError,
    depthHistory,
    start,
    stop,
  };
}
