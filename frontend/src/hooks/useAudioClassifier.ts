import { useCallback, useMemo, useState } from "react";

import {
  CLASSIFIER_MODELS,
  DEFAULT_CLASSIFIER_MODEL,
} from "@/audio/classification";
import type { ClassLabel } from "@/audio/pipelineTypes";
import { usePipeline, type UsePipelineResult } from "@/hooks/usePipeline";

/** How many labels to return for fixed-label (non zero-shot) models. */
const TOP_K = 6;

export interface UseAudioClassifierResult extends UsePipelineResult {
  /** True when the selected model scores against free-text prompts (CLAP). */
  isZeroShot: boolean;
  /** Latest ranked predictions, highest score first. */
  result: ClassLabel[] | null;
  /**
   * Classify a mono 16 kHz clip. For zero-shot models, `labels` are the candidate
   * prompts to score against (required); ignored for fixed-label models.
   */
  classify: (audio: Float32Array, labels?: string[]) => Promise<void>;
}

/**
 * In-browser audio classification. Wraps the generic {@link usePipeline} worker,
 * selecting `audio-classification` or `zero-shot-audio-classification` from the
 * model catalogue, and shapes the pipeline args for each: `{ top_k }` for
 * fixed-label tagging, or the candidate-label list for zero-shot CLAP.
 */
export function useAudioClassifier(
  model: string = DEFAULT_CLASSIFIER_MODEL,
): UseAudioClassifierResult {
  const meta = useMemo(
    () => CLASSIFIER_MODELS.find((m) => m.id === model) ?? CLASSIFIER_MODELS[0],
    [model],
  );
  const isZeroShot = meta.task === "zero-shot-audio-classification";
  const pipe = usePipeline(meta.task, meta.id);
  const { run } = pipe;

  const [result, setResult] = useState<ClassLabel[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const classify = useCallback(
    async (audio: Float32Array, labels?: string[]) => {
      setError(null);
      try {
        let args: unknown[];
        if (isZeroShot) {
          const prompts = (labels ?? []).map((l) => l.trim()).filter(Boolean);
          if (prompts.length === 0) {
            throw new Error("Add at least one label to score against.");
          }
          args = [prompts];
        } else {
          args = [{ top_k: TOP_K }];
        }
        const out = (await run(audio, args)) as ClassLabel[];
        setResult(out);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [isZeroShot, run],
  );

  return {
    ...pipe,
    error: error ?? pipe.error,
    isZeroShot,
    result,
    classify,
  };
}
