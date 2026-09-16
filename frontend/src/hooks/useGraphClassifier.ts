// Owns the /graph-classification route's worker, its downloaded dataset and a
// streaming run.
//
// The shape of useGraphTraining and useLinkPrediction, with one addition: this is
// the only one of the three whose LOAD actually goes to the network, so it also
// owns the **lazy layouts** the gallery asks for. A tile requests the graph it is
// about to draw; the worker computes that one layout and remembers it, so
// scrolling the gallery never re-lays-out anything it has already shown.

import { useCallback, useEffect, useRef, useState } from "react";

import type { GnnArch } from "@/webgpu/gnn";
import type { GraphClassMetrics, ReadoutMode } from "@/webgpu/graphPool";
import type { GraphBackend } from "@/webgpu/graphSession";
import type {
  GraphLayoutPayload,
  ProteinSummary,
  ProteinTrainRequest,
} from "@/webgpu/proteinSession";
import {
  createWebGPUWorker,
  layoutProteinInWorker,
  loadProteinsInWorker,
  trainProteinsInWorker,
  type ProteinTrainingHandle,
} from "@/webgpu/workerClient";

export type ProteinLoadStatus = "idle" | "loading" | "ready" | "error";

/** One completed run, so two architectures can be compared on the same split. */
export interface ProteinRunPoint {
  arch: GnnArch;
  readout: ReadoutMode;
  testAcc: number;
}

export interface GraphClassifierState {
  status: ProteinLoadStatus;
  summary: ProteinSummary | null;
  loadError: string | null;
  backend: GraphBackend | null;
  load: () => void;
  retry: () => void;

  training: boolean;
  metrics: GraphClassMetrics[];
  /** The class each of the graphs is currently predicted to be. */
  predicted: Uint8Array | null;
  elapsedMs: number | null;
  trainError: string | null;
  history: ProteinRunPoint[];
  start: (request: ProteinTrainRequest) => void;
  stop: () => void;

  /** Layouts the gallery has asked for, by graph index. */
  layouts: Map<number, GraphLayoutPayload>;
  requestLayout: (graph: number) => void;
}

export function useGraphClassifier(): GraphClassifierState {
  const workerRef = useRef<Worker | null>(null);
  const handleRef = useRef<ProteinTrainingHandle | null>(null);

  const [status, setStatus] = useState<ProteinLoadStatus>("idle");
  const statusRef = useRef(status);
  statusRef.current = status;
  const [summary, setSummary] = useState<ProteinSummary | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [training, setTraining] = useState(false);
  const [metrics, setMetrics] = useState<GraphClassMetrics[]>([]);
  const [predicted, setPredicted] = useState<Uint8Array | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [history, setHistory] = useState<ProteinRunPoint[]>([]);
  const [layouts, setLayouts] = useState<Map<number, GraphLayoutPayload>>(
    () => new Map(),
  );
  // Which layouts are already asked for, so a gallery that re-renders while one
  // is in flight does not ask again. A ref because it must be read and written
  // between renders, not across them.
  const pendingLayouts = useRef<Set<number>>(new Set());

  const metricsRef = useRef<GraphClassMetrics[]>([]);
  const pendingPredicted = useRef<Uint8Array | null>(null);
  const flushScheduled = useRef(false);
  const scheduleFlush = useCallback(() => {
    if (flushScheduled.current) return;
    flushScheduled.current = true;
    requestAnimationFrame(() => {
      flushScheduled.current = false;
      setMetrics(metricsRef.current.slice());
      if (pendingPredicted.current) setPredicted(pendingPredicted.current);
    });
  }, []);

  useEffect(() => {
    return () => {
      handleRef.current?.cancel();
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  const loadingRef = useRef(false);

  const load = useCallback(() => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    setStatus("loading");
    setLoadError(null);

    workerRef.current ??= createWebGPUWorker();
    loadProteinsInWorker(workerRef.current)
      .then((loaded) => {
        setSummary(loaded);
        setStatus("ready");
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

  const requestLayout = useCallback((graph: number) => {
    const worker = workerRef.current;
    if (!worker || pendingLayouts.current.has(graph)) return;
    pendingLayouts.current.add(graph);
    layoutProteinInWorker(worker, graph)
      .then((layout) => {
        setLayouts((current) => new Map(current).set(graph, layout));
      })
      .catch(() => {
        // A tile that cannot be drawn stays a placeholder. It is a picture, not
        // a result — failing the page over one would be the wrong trade.
        pendingLayouts.current.delete(graph);
      });
  }, []);

  const start = useCallback(
    (request: ProteinTrainRequest) => {
      const worker = workerRef.current;
      if (!worker || statusRef.current !== "ready") {
        setTrainError("Load the dataset first.");
        return;
      }

      metricsRef.current = [];
      pendingPredicted.current = null;
      setMetrics([]);
      setPredicted(null);
      setElapsedMs(null);
      setTrainError(null);
      setTraining(true);

      const handle = trainProteinsInWorker(worker, request, (m, pred) => {
        metricsRef.current.push(m);
        pendingPredicted.current = pred;
        scheduleFlush();
      });
      handleRef.current = handle;

      handle.promise
        .then((result) => {
          const final = result.metrics;
          if (!final) return;
          setMetrics(metricsRef.current.slice());
          setPredicted(result.predicted);
          setElapsedMs(result.elapsedMs);
          setHistory((past) => [
            ...past.filter(
              (p) =>
                !(p.arch === request.arch && p.readout === request.readout),
            ),
            {
              arch: request.arch,
              readout: request.readout,
              testAcc: final.testAcc,
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
    predicted,
    elapsedMs,
    trainError,
    history,
    start,
    stop,
    layouts,
    requestLayout,
  };
}
