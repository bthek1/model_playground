// In-browser object detection. A thin wrapper over `useVisionPipeline`.
//
// **`percentage: false` is pinned here**, once, for every caller. Transformers.js
// returns 0–1 fractions by default and `drawBoxes` wants absolute pixels; the
// wrong choice puts every box in the top-left corner and is the most common bug
// on a first detection page. A unit test asserts the flag reaches the pipeline
// call — a regression test for a named bug, not box-ticking.
//
// The threshold sent to the *model* is deliberately low (`MODEL_THRESHOLD`). The
// page's slider re-filters the returned list, which is a pure derivation and
// needs no second inference — the same trick `/vad` uses for its threshold.

import { useCallback, useMemo, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import {
  useVisionPipeline,
  type UseVisionPipelineResult,
} from "@/hooks/useVisionPipeline";
import type { Detection } from "@/vision/draw";
import {
  DEFAULT_DETECTOR,
  DETECTOR_MODELS,
  MODEL_THRESHOLD,
} from "@/vision/detection";

export interface UseObjectDetectorResult
  extends Omit<UseVisionPipelineResult, "run"> {
  /** Every detection above `MODEL_THRESHOLD`, in source-image pixels. */
  result: Detection[] | null;
  run: (
    image: RawImage,
    opts?: { consume?: boolean },
  ) => Promise<Detection[]>;
}

export function useObjectDetector(
  model: string = DEFAULT_DETECTOR,
  autoLoad = false,
): UseObjectDetectorResult {
  const meta = useMemo(
    () => DETECTOR_MODELS.find((m) => m.id === model) ?? DETECTOR_MODELS[0],
    [model],
  );
  const pipe = useVisionPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<Detection[] | null>(null);

  const run = useCallback(
    async (
      image: RawImage,
      opts?: { consume?: boolean },
    ): Promise<Detection[]> => {
      const out = (await post(
        image,
        [{ threshold: MODEL_THRESHOLD, percentage: false }],
        opts,
      )) as Detection[];
      setResult(out);
      return out;
    },
    [post],
  );

  return { ...pipe, result, run };
}

/**
 * Filter detections by score. Pure, and on the main thread on purpose: dragging
 * the threshold slider re-derives the visible boxes without re-running the model.
 */
export function aboveThreshold(
  detections: readonly Detection[] | null,
  threshold: number,
): Detection[] {
  return (detections ?? []).filter((d) => d.score >= threshold);
}
