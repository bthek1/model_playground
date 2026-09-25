// The Tabular category's task hook. Thin, like every other one — the worker
// lifecycle, the id-correlated pending table and the two state machines all
// live in `model/useModelWorker.ts` and are not re-derived here.
//
// Two things about it are unlike the other task hooks, and both follow from the
// category rather than from taste:
//
// **`load` spends nothing, and the FIT button is a `run`.** Everywhere else
// `load` is a weight download and is the expensive click. Here it hands typed
// arrays that are already in the tab to a worker in the same tab; no bytes
// cross a network and nothing is allocated that was not allocated already. What
// costs is the *fit*, which is a request — so its iteration metrics ride
// `partial` (progress inside one run, correlated to its id), Machine A stays
// `ready` throughout, and `running` stays an inflight count. No new envelope and
// no new status: `docs/standards/model-page-pattern.md` §7.
//
// **`fit()` does both.** The page has one button, and pressing it on a cold
// worker must not require a second press. So `fit()` posts `load` when idle and
// the `run` immediately behind it; the engine serialises the two, which is why
// it keeps a chain rather than trusting message order.

import { useCallback, useRef, useState } from "react";

import type { LoadProgress } from "@/model/progress";
import type { ModelProgress, ModelStatus } from "@/model/types";
import { useModelWorker } from "@/model/useModelWorker";
import { createFitWorker } from "@/tabular/client";
import type {
  Dataset,
  FitPartial,
  FitResult,
  FitSpec,
  PredictResult,
} from "@/tabular/types";

export interface UseTabularFitResult {
  status: ModelStatus;
  idle: boolean;
  loading: boolean;
  ready: boolean;
  progress: ModelProgress | null;
  loadProgress: LoadProgress | null;
  loadedInMs: number | null;
  backend: string | null;
  /** True while a fit or a prediction is in flight. */
  running: boolean;
  /** The fit's own determinate counter — never a byte total. */
  partial: FitPartial | null;
  /** The most recent completed fit. Predictions do not replace it. */
  result: FitResult | null;
  error: string | null;
  load: () => void;
  retry: (overrides?: Record<string, unknown>) => void;
  cancel: () => void;
  /** Fit the ladder rung named by `spec`. The only thing on this page that spends. */
  fit: (spec: FitSpec) => Promise<FitResult>;
  /** Predict one hand-typed row against the fit in hand. */
  predict: (values: (number | string | null)[]) => Promise<PredictResult>;
  /**
   * Interrupt the fit in flight. Out of band, not a request: it sets a flag the
   * fit loop polls, so the run it stops still settles with whatever was built.
   */
  stop: () => void;
}

export function useTabularFit(
  dataset: Dataset | null,
  createWorker: () => Worker = createFitWorker,
): UseTabularFitResult {
  // The worker instance, captured on the way past, so `stop()` can post to it
  // without `useModelWorker` growing an out-of-band send for one category.
  const workerRef = useRef<Worker | null>(null);
  const spawn = useCallback(() => {
    const worker = createWorker();
    workerRef.current = worker;
    return worker;
  }, [createWorker]);

  const [fitResult, setFitResult] = useState<FitResult | null>(null);

  const worker = useModelWorker<FitResult | PredictResult, FitPartial>({
    createWorker: spawn,
    // Identity is the dataset: a new file is a new worker and a fresh machine.
    // Hyperparameters deliberately are **not** in here — changing one must
    // leave the fit already on screen alone until the user presses FIT again.
    key: dataset ? `${dataset.name}:${dataset.rowCount}:${dataset.columns.length}` : "none",
    loadMessage: { dataset },
    notReadyMessage: "Fit worker not ready",
  });

  const { run: post, load, status } = worker;
  const statusRef = useRef(status);
  statusRef.current = status;

  const fit = useCallback(
    async (spec: FitSpec): Promise<FitResult> => {
      if (statusRef.current === "idle") load();
      const result = (await post({ spec })) as FitResult;
      setFitResult(result);
      return result;
    },
    [load, post],
  );

  const predict = useCallback(
    async (values: (number | string | null)[]): Promise<PredictResult> =>
      (await post({ predict: { values } })) as PredictResult,
    [post],
  );

  const stop = useCallback(() => {
    workerRef.current?.postMessage({ type: "stop" });
  }, []);

  return {
    status: worker.status,
    idle: worker.idle,
    loading: worker.loading,
    ready: worker.ready,
    progress: worker.progress,
    loadProgress: worker.loadProgress,
    loadedInMs: worker.loadedInMs,
    backend: worker.backend,
    running: worker.running,
    partial: worker.partial,
    result: fitResult,
    error: worker.error,
    load: worker.load,
    retry: worker.retry,
    cancel: worker.cancel,
    fit,
    predict,
    stop,
  };
}
