// Zero-shot text classification — your own labels, no fine-tuning. A thin
// wrapper over the generic `useTextPipeline` worker: it picks the catalogue
// entry, shapes the run payload and types the result. Worker lifecycle, the
// pending table and the two state machines belong to `model/useModelWorker.ts`
// and are not re-derived here.
//
// The one thing worth knowing about the call underneath: the pipeline runs the
// model **once per label**, as an NLI premise–hypothesis pair. `run` is a
// single request from the page's point of view and N forward passes from the
// model's, which is why the route quotes the pass count beside GENERATE. See
// `text/zeroShot.ts`.

import { useCallback, useMemo, useState } from "react";

import {
  useTextPipeline,
  type UseTextPipelineResult,
} from "@/hooks/useTextPipeline";
import type { ClassLabel } from "@/model/types";
import {
  DEFAULT_ZERO_SHOT_TEXT,
  ZERO_SHOT_TEXT_MODELS,
} from "@/text/catalogue";
import { DEFAULT_HYPOTHESIS_TEMPLATE } from "@/text/zeroShot";

/** One press of GENERATE: the premise, the labels, and how to score them. */
export interface ZeroShotRun {
  text: string;
  labels: readonly string[];
  /**
   * Independent sigmoid-style scoring per label instead of a softmax across
   * them. **This changes the arithmetic, not the model** — but it cannot be
   * re-derived from a finished result, because the per-label entailment and
   * contradiction logits the independent branch needs are not in the numbers
   * the pipeline hands back. So flipping it runs nothing, and the next
   * GENERATE is a real second inference. The page says so before the click.
   */
  multiLabel?: boolean;
  /** The NLI hypothesis frame; `{}` is replaced by each label. */
  hypothesisTemplate?: string;
}

/** What the pipeline returns: labels and scores, already ranked. */
interface RawZeroShot {
  sequence: string;
  labels: string[];
  scores: number[];
}

export interface UseZeroShotTextResult
  extends Omit<UseTextPipelineResult, "run"> {
  /** Latest scores, highest first. */
  result: ClassLabel[] | null;
  run: (request: ZeroShotRun) => Promise<ClassLabel[]>;
}

export function useZeroShotText(
  model: string = DEFAULT_ZERO_SHOT_TEXT,
  autoLoad = false,
): UseZeroShotTextResult {
  const meta = useMemo(
    () =>
      ZERO_SHOT_TEXT_MODELS.find((m) => m.id === model) ??
      ZERO_SHOT_TEXT_MODELS[0],
    [model],
  );
  const pipe = useTextPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<ClassLabel[] | null>(null);

  const run = useCallback(
    async ({
      text,
      labels,
      multiLabel = false,
      hypothesisTemplate = DEFAULT_HYPOTHESIS_TEMPLATE,
    }: ZeroShotRun): Promise<ClassLabel[]> => {
      // The template is always sent, never left to default. The pipeline has a
      // default of its own ("This example is {}.") and a page that lets it
      // apply is comparing a prompt the user cannot see — the
      // `hypothesis_template` finding from /zero-shot-image-classification,
      // where the image pipeline silently prefixes "This is a photo of".
      const out = (await post(text, [
        [...labels],
        { multi_label: multiLabel, hypothesis_template: hypothesisTemplate },
      ])) as RawZeroShot;

      const scores = toScores(out);
      setResult(scores);
      return scores;
    },
    [post],
  );

  return { ...pipe, result, run };
}

/**
 * Zip the pipeline's parallel `labels` / `scores` arrays into the shared
 * `ClassLabel` shape every score list in the app renders.
 *
 * Guarded on length rather than trusting the pair: they come back sorted
 * together, and a mismatch would silently attach one label's score to another.
 */
function toScores(out: RawZeroShot | null | undefined): ClassLabel[] {
  const labels = out?.labels ?? [];
  const scores = out?.scores ?? [];
  const n = Math.min(labels.length, scores.length);
  return Array.from({ length: n }, (_, i) => ({
    label: labels[i],
    score: scores[i],
  }));
}
