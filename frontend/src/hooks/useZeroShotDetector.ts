// In-browser zero-shot object detection. A thin wrapper over
// `useVisionPipeline`, and deliberately the same shape as `useObjectDetector` —
// the two tasks differ by one argument, and a reader who has read one should
// recognise the other immediately.
//
// **`percentage: false` is pinned here**, once, for every caller, for exactly
// the reason it is pinned in `useObjectDetector`: `drawBoxes` wants absolute
// pixels, and 0–1 fractions collapse every box into the top-left corner. The
// pipeline's own default already happens to be `false`, which is precisely what
// makes it worth pinning — a default that flips in a minor release would be a
// silent, plausible-looking regression. A unit test asserts the flag reaches
// the pipeline call.
//
// The threshold sent to the *model* is `MODEL_THRESHOLD`, well below anything
// the page displays. The slider re-filters the returned list on the main
// thread — the same pure-derivation trick `/object-detection` and `/vad` use.

import { useCallback, useMemo, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import {
  useVisionPipeline,
  type UseVisionPipelineResult,
} from "@/hooks/useVisionPipeline";
import type { Detection } from "@/vision/draw";
import {
  DEFAULT_ZERO_SHOT_DETECTOR,
  MODEL_THRESHOLD,
  ZERO_SHOT_DETECTOR_MODELS,
} from "@/vision/zeroShotDetection";

export interface UseZeroShotDetectorResult
  extends Omit<UseVisionPipelineResult, "run"> {
  /** Every detection above `MODEL_THRESHOLD`, in inference-frame pixels. */
  result: Detection[] | null;
  /**
   * Detect `queries` in `image`. The queries are sent **verbatim** — this
   * pipeline applies no prompt template, so the strings the user typed are the
   * strings the text tower sees.
   */
  run: (
    image: RawImage,
    queries: readonly string[],
    opts?: { consume?: boolean },
  ) => Promise<Detection[]>;
}

export function useZeroShotDetector(
  model: string = DEFAULT_ZERO_SHOT_DETECTOR,
  autoLoad = false,
): UseZeroShotDetectorResult {
  const meta = useMemo(
    () =>
      ZERO_SHOT_DETECTOR_MODELS.find((m) => m.id === model) ??
      ZERO_SHOT_DETECTOR_MODELS[0],
    [model],
  );
  const pipe = useVisionPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<Detection[] | null>(null);

  const run = useCallback(
    async (
      image: RawImage,
      queries: readonly string[],
      opts?: { consume?: boolean },
    ): Promise<Detection[]> => {
      const clean = queries.map((q) => q.trim()).filter(Boolean);
      if (clean.length === 0) {
        throw new Error("Add at least one query to detect");
      }
      const out = (await post(
        image,
        [clean, { threshold: MODEL_THRESHOLD, percentage: false }],
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
 * Group detections by the query that produced them, in the user's own query
 * order, keeping the empty groups.
 *
 * The empty ones are the point: "which of my phrases found nothing" is the
 * question this page exists to answer, and a list that silently omits the
 * misses answers it wrong. Pure, so it re-derives as the threshold moves.
 */
export function groupByQuery(
  detections: readonly Detection[],
  queries: readonly string[],
): { query: string; detections: Detection[] }[] {
  const groups = queries.map((query) => ({
    query,
    detections: [] as Detection[],
  }));
  const byLabel = new Map(groups.map((g) => [g.query, g]));
  const extra: { query: string; detections: Detection[] }[] = [];

  for (const d of detections) {
    const group = byLabel.get(d.label);
    if (group) {
      group.detections.push(d);
      continue;
    }
    // Grounding DINO answers with a fragment of the query rather than the query
    // itself, so a label we did not ask for is expected there — surfaced under
    // its own heading rather than dropped.
    let bucket = extra.find((g) => g.query === d.label);
    if (!bucket) {
      bucket = { query: d.label, detections: [] };
      extra.push(bucket);
    }
    bucket.detections.push(d);
  }

  return [...groups, ...extra];
}
