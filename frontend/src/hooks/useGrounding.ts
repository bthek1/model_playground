// `/robotics` — an open-vocabulary detector and a depth model, loaded as one
// pair and run on one frame.
//
// Composed from the two hooks that already own each half —
// `useZeroShotDetector` (which pins `percentage: false` and the low model-side
// threshold; neither is this page's decision) and `useDepth`. Each goes through
// `useVisionPipeline`, so **each half is its own worker**. That shape decides
// three things here:
//
//  * **`ready` means both.** A box with no distance, or a distance with no box,
//    answers neither half of "which one do I reach for".
//  * **One bar, via `combineProgress`.** Two workers keep two progress tables;
//    showing either fills to 100% and then restarts — the "100% halfway through
//    a pair" failure `/pose` documents, reached from the other direction. The
//    denominator is the pair entry's measured combined size, a constant, so the
//    percent is monotonic by construction.
//  * **No `Promise.allSettled` on teardown — deliberately.** The obvious
//    precedent for "two models live at once" is `/pose`, which disposes its pair
//    with `allSettled`, and the plan for this page (§3.6) asked for the same.
//    That is right for `/pose` and dead code here: `/pose` loads both models
//    inside **one** worker, so one engine owns a combined teardown and can get it
//    wrong. Here each engine disposes its single model with `disposeQuietly`,
//    and `useModelWorker` terminates each worker from its own effect — a dispose
//    that throws in one worker is structurally unable to reach the other. That
//    is the property `allSettled` exists to buy, obtained by the split instead.
//    `/text-ranking` (`useRanking`) is the same shape and says the same thing;
//    `useGrounding.test.ts` asserts it rather than reasoning about it.

import { useCallback, useEffect, useMemo, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import { useDepth, type DepthResult } from "@/hooks/useDepth";
import { useZeroShotDetector } from "@/hooks/useZeroShotDetector";
import { pickBackend, type Backend } from "@/model/backend";
import { combineProgress, type LoadProgress } from "@/model/progress";
import type { ModelStatus } from "@/model/types";
import type { Detection } from "@/vision/draw";
import {
  DEFAULT_ROBOTICS_ENTRY,
  GROUNDING_ENTRIES,
  type GroundingEntry,
} from "@/vision/grounding";

/** One frame, both answers. Captured together so they can never disagree. */
export interface GroundingResult {
  /** Every detection above the model-side floor, in inference-frame pixels. */
  detections: Detection[];
  depth: DepthResult;
  /** Wall-clock per half — a pair on a live feed is this page's real cost. */
  detectMs: number;
  depthMs: number;
}

export interface UseGroundingResult {
  pair: GroundingEntry;
  /** Combined status: `ready` only when **both** halves are. */
  status: ModelStatus;
  idle: boolean;
  loading: boolean;
  ready: boolean;
  /** One aggregate bar across both workers. */
  loadProgress: LoadProgress | null;
  /** The slower half's load time — the pair was unusable until then. */
  loadedInMs: number | null;
  backend: string | null;
  /** Either half's load error, else the most recent run error. */
  error: string | null;
  running: boolean;
  result: GroundingResult | null;
  /** One press of LOAD starts both downloads. */
  load: () => void;
  /** Retries only the half that failed — the other is already in memory. */
  retry: (overrides?: Record<string, unknown>) => void;
  cancel: () => void;
  /**
   * Detect `queries` in `image`, then estimate its depth — in that order, one
   * after the other, on the **same** frame. `consume` hands the pixels to the
   * *last* call only: the first still needs them.
   */
  run: (
    image: RawImage,
    queries: readonly string[],
    opts?: { consume?: boolean },
  ) => Promise<GroundingResult>;
}

export function useGrounding(
  pairId: string = DEFAULT_ROBOTICS_ENTRY,
  autoLoad = false,
): UseGroundingResult {
  const pair = useMemo(
    () => GROUNDING_ENTRIES.find((p) => p.id === pairId) ?? GROUNDING_ENTRIES[0],
    [pairId],
  );

  const detector = useZeroShotDetector(pair.detector.id, autoLoad);
  const depth = useDepth(pair.depth.id, autoLoad);

  const [result, setResult] = useState<GroundingResult | null>(null);

  // Which backend's combined size is the bar's denominator. The two differ by
  // ~2x, so guessing would make the bar wrong for the first seconds of a load.
  const [probe, setProbe] = useState<Backend | null>(null);
  useEffect(() => {
    let live = true;
    void pickBackend().then((b) => {
      if (live) setProbe(b);
    });
    return () => {
      live = false;
    };
  }, []);

  const loadProgress = useMemo(
    () =>
      combineProgress(
        [detector.loadProgress, depth.loadProgress],
        pair.bytes[probe ?? "webgpu"] ?? 0,
        Math.max(
          detector.loadProgress?.elapsedMs ?? 0,
          depth.loadProgress?.elapsedMs ?? 0,
        ),
      ),
    [detector.loadProgress, depth.loadProgress, pair.bytes, probe],
  );

  const status: ModelStatus = useMemo(() => {
    if (detector.status === "error" || depth.status === "error") return "error";
    if (detector.ready && depth.ready) return "ready";
    if (detector.loading || depth.loading) return "loading";
    return "idle";
  }, [
    detector.status,
    detector.ready,
    detector.loading,
    depth.status,
    depth.ready,
    depth.loading,
  ]);

  const { load: loadDetector, retry: retryDetector, cancel: cancelDetector } =
    detector;
  const { load: loadDepth, retry: retryDepth, cancel: cancelDepth } = depth;
  const detectorFailed = detector.status === "error";
  const depthFailed = depth.status === "error";

  const load = useCallback(() => {
    loadDetector();
    loadDepth();
  }, [loadDetector, loadDepth]);

  const retry = useCallback(
    (overrides?: Record<string, unknown>) => {
      if (detectorFailed) retryDetector(overrides);
      if (depthFailed) retryDepth(overrides);
    },
    [detectorFailed, depthFailed, retryDetector, retryDepth],
  );

  const cancel = useCallback(() => {
    cancelDetector();
    cancelDepth();
  }, [cancelDetector, cancelDepth]);

  const { run: detect } = detector;
  const { run: estimate } = depth;
  const run = useCallback(
    async (
      image: RawImage,
      queries: readonly string[],
      opts?: { consume?: boolean },
    ): Promise<GroundingResult> => {
      // Sequential, not `Promise.all`: on one GPU the two would contend anyway,
      // and the detector's copy must be taken before the depth call is allowed
      // to consume the buffer.
      const t0 = performance.now();
      const detections = await detect(image, queries, { consume: false });
      const t1 = performance.now();
      const map = await estimate(image, { consume: opts?.consume ?? false });
      const t2 = performance.now();
      const out: GroundingResult = {
        detections,
        depth: map,
        detectMs: Math.round(t1 - t0),
        depthMs: Math.round(t2 - t1),
      };
      setResult(out);
      return out;
    },
    [detect, estimate],
  );

  return {
    pair,
    status,
    idle: status === "idle",
    loading: status === "loading",
    ready: status === "ready",
    loadProgress,
    loadedInMs:
      detector.loadedInMs != null && depth.loadedInMs != null
        ? Math.max(detector.loadedInMs, depth.loadedInMs)
        : null,
    backend: detector.backend ?? depth.backend,
    error:
      (detectorFailed ? detector.error : null) ??
      (depthFailed ? depth.error : null) ??
      detector.error ??
      depth.error,
    running: detector.running || depth.running,
    result,
    load,
    retry,
    cancel,
    run,
  };
}
