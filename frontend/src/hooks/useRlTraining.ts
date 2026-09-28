// Owns the /rl route's worker and its streaming training runs.
//
// The shape is `useGraphTraining`'s: a worker held in a ref, metrics batched to
// animation frames, `start`/`stop`. Three differences, each deliberate:
//
//   - **There is no load.** Nothing is downloaded and nothing is decoded, so
//     the worker is created on the first Train rather than on a LOAD press —
//     and never on mount, which the route test asserts.
//   - **`control` changes a run in progress without restarting it** — ε and the
//     simulation speed. Everything else on the page is a choice that spends
//     nothing until Train.
//   - **The history is per configuration.** `/graph`'s `depthHistory` keeps one
//     point per (architecture, depth); here one entry per request actually
//     trained, so a second seed adds to the comparison instead of replacing the
//     first — which is the comparison the page exists for.

import { useCallback, useEffect, useRef, useState } from "react";

import { createRlWorker, type RlTrainingHandle, trainRlInWorker } from "@/rl/client";
import type {
  RlControl,
  RlProgress,
  RlRenderState,
  RlTrainRequest,
  RlTrainResult,
} from "@/rl/types";

/** One finished (or stopped) run, kept for the comparison. */
export interface RlRunRecord {
  key: string;
  request: RlTrainRequest;
  /** Every episode's return, unsmoothed. */
  returns: number[];
  result: RlTrainResult;
}

/** At most this many runs are kept; the oldest goes first. */
export const HISTORY_LIMIT = 12;

/** Two runs are the same configuration iff every field of the request matches. */
export function runKey(req: RlTrainRequest): string {
  return JSON.stringify(req);
}

export interface RlTrainingState {
  training: boolean;
  /** The current run's per-episode returns, or the last run's. */
  returns: number[];
  /** The newest progress line and render state. */
  progress: Omit<RlProgress, "returns"> | null;
  render: RlRenderState | null;
  result: RlTrainResult | null;
  error: string | null;
  history: RlRunRecord[];
  start: (req: RlTrainRequest, speed?: number | null) => void;
  stop: () => void;
  control: (control: RlControl) => void;
  clearHistory: () => void;
}

export function useRlTraining(): RlTrainingState {
  const workerRef = useRef<Worker | null>(null);
  const handleRef = useRef<RlTrainingHandle | null>(null);

  const [training, setTraining] = useState(false);
  const trainingRef = useRef(false);
  const [returns, setReturns] = useState<number[]>([]);
  const [progress, setProgress] = useState<Omit<RlProgress, "returns"> | null>(null);
  const [result, setResult] = useState<RlTrainResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<RlRunRecord[]>([]);

  // rAF-batched streaming. The worker already throttles to ~60 Hz, but React
  // should still see at most one commit per frame whatever arrives.
  const returnsRef = useRef<number[]>([]);
  const pendingProgress = useRef<Omit<RlProgress, "returns"> | null>(null);
  const flushScheduled = useRef(false);
  const scheduleFlush = useCallback(() => {
    if (flushScheduled.current) return;
    flushScheduled.current = true;
    requestAnimationFrame(() => {
      flushScheduled.current = false;
      setReturns(returnsRef.current.slice());
      if (pendingProgress.current) setProgress(pendingProgress.current);
    });
  }, []);

  useEffect(() => {
    return () => {
      handleRef.current?.cancel();
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  const start = useCallback(
    (req: RlTrainRequest, speed: number | null = null) => {
      if (trainingRef.current) return;
      trainingRef.current = true;
      workerRef.current ??= createRlWorker();

      returnsRef.current = [];
      pendingProgress.current = null;
      setReturns([]);
      setProgress(null);
      setResult(null);
      setError(null);
      setTraining(true);

      const handle = trainRlInWorker(
        workerRef.current,
        req,
        ({ returns: batch, ...rest }) => {
          for (const r of batch) returnsRef.current.push(r);
          pendingProgress.current = rest;
          scheduleFlush();
        },
        speed,
      );
      handleRef.current = handle;

      handle.promise
        .then((final) => {
          const all = returnsRef.current.slice();
          setReturns(all);
          if (pendingProgress.current) setProgress(pendingProgress.current);
          setResult(final);
          // Stopped before a single episode finished: nothing to compare.
          if (final.episodes === 0) return;
          const key = runKey(req);
          setHistory((h) =>
            [...h.filter((r) => r.key !== key), { key, request: req, returns: all, result: final }].slice(
              -HISTORY_LIMIT,
            ),
          );
        })
        .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => {
          trainingRef.current = false;
          handleRef.current = null;
          setTraining(false);
        });
    },
    [scheduleFlush],
  );

  const stop = useCallback(() => handleRef.current?.cancel(), []);
  const control = useCallback((c: RlControl) => handleRef.current?.control(c), []);
  const clearHistory = useCallback(() => setHistory([]), []);

  return {
    training,
    returns,
    progress,
    render: progress?.render ?? result?.render ?? null,
    result,
    error,
    history,
    start,
    stop,
    control,
    clearHistory,
  };
}
