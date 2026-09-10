// Keypoint detection — the shared contract verbatim over `model/useModelWorker`.
//
// Nothing task-specific leaks into the contract: the fact that this is *two*
// models is entirely the worker's business, and the LOAD state covers both
// downloads with one aggregate bar because `model/progress.ts` already
// aggregates monotonically across files by bytes. (It aggregates across *repos*
// too now — it had to be keyed on repo + file for this route, since both
// checkpoints publish an `onnx/model_fp16.onnx` and the second was overwriting
// the first's entry.)
//
// The threshold and the people cap travel in the run payload rather than being
// re-derived on the main thread, unlike `/object-detection`'s slider. That is a
// real difference, not an inconsistency: filtering *after* the fact would mean
// running the pose model on people the user has already excluded, and the pose
// pass is the expensive half. The threshold here changes what is computed.

import { useCallback, useMemo, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import type { LoadProgress } from "@/model/progress";
import type { ModelProgress, ModelStatus } from "@/model/types";
import { useModelWorker } from "@/model/useModelWorker";
import { toPayload, transferablesOf } from "@/vision/image";
import { createPoseWorker } from "@/vision/pose/client";
import {
  DEFAULT_POSE_MODEL,
  POSE_MODELS,
  type PoseResult,
} from "@/vision/pose/types";

export interface UsePoseResult {
  status: ModelStatus;
  idle: boolean;
  loading: boolean;
  ready: boolean;
  progress: ModelProgress | null;
  loadProgress: LoadProgress | null;
  loadedInMs: number | null;
  backend: string | null;
  running: boolean;
  error: string | null;
  result: PoseResult | null;
  /** Find people in `image` and estimate each one's pose. */
  run: (
    image: RawImage,
    opts: { threshold: number; maxPeople: number; consume?: boolean },
  ) => Promise<PoseResult>;
  load: () => void;
  retry: (overrides?: Record<string, unknown>) => void;
  cancel: () => void;
}

export function usePose(
  model: string = DEFAULT_POSE_MODEL,
  autoLoad = false,
): UsePoseResult {
  const meta = useMemo(
    () => POSE_MODELS.find((m) => m.id === model) ?? POSE_MODELS[0],
    [model],
  );

  const loadMessage = useMemo(() => ({ model: meta.id }), [meta]);

  const worker = useModelWorker<PoseResult>({
    createWorker: createPoseWorker,
    key: `pose:${meta.id}`,
    loadMessage,
    autoLoad,
    notReadyMessage: "Pose worker not ready",
  });

  const { run: post } = worker;
  const [result, setResult] = useState<PoseResult | null>(null);

  const run = useCallback(
    async (
      image: RawImage,
      {
        threshold,
        maxPeople,
        consume = false,
      }: { threshold: number; maxPeople: number; consume?: boolean },
    ): Promise<PoseResult> => {
      const payload = toPayload(image, { copy: !consume });
      const res = await post(
        { image: payload, threshold, maxPeople },
        transferablesOf(payload),
      );
      setResult(res);
      return res;
    },
    [post],
  );

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
    error: worker.error,
    result,
    run,
    load: worker.load,
    retry: worker.retry,
    cancel: worker.cancel,
  };
}
