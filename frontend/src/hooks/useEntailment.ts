// Does the article support this sentence? The faithfulness half of
// `/summarization`, and a second model rather than a second view.
//
// It downloads nothing new *for the app* — the checkpoint is
// `/zero-shot-classification`'s cheapest entry (27 MB on CPU), reused — but it
// is a second download and a second live model for this page, so the route
// gates it behind its own LOAD with the cost stated. `/text-classification`'s
// head-to-head is the precedent.
//
// **Scoring a sentence against a passage is exactly a one-label NLI call.** The
// zero-shot pipeline's `softmaxEach` is true when `labels.length === 1`, so for
// a single hypothesis it returns entailment scored against contradiction rather
// than softmaxed across siblings — which is the number wanted. Passing all the
// summary's sentences as one label list would instead softmax them against each
// other, making the scores sum to 1 and mean nothing: "which sentence is most
// entailed" is not the question. So it is **one call per sentence**, and the
// route states the pass count before the click, exactly as the zero-shot page
// does.
//
// The honest limitation, which the page also states: MNLI premises are single
// sentences, so a whole article as the premise is out of distribution and gets
// truncated at the encoder's 512 tokens. A low score is evidence worth looking
// at, not a verdict.

import { useCallback, useState } from "react";

import {
  useTextPipeline,
  type UseTextPipelineResult,
} from "@/hooks/useTextPipeline";
import { FAITHFULNESS_MODEL, ZERO_SHOT_TEXT_MODELS } from "@/text/catalogue";
import { BARE_HYPOTHESIS_TEMPLATE } from "@/text/zeroShot";

/** One sentence, and how well the passage supports it. */
export interface Entailment {
  sentence: string;
  /** Entailment probability against contradiction, in [0, 1]. */
  score: number;
}

interface RawZeroShot {
  labels?: string[];
  scores?: number[];
}

export interface UseEntailmentResult extends Omit<UseTextPipelineResult, "run"> {
  /** Model id, so the page can quote its size from the shared catalogue. */
  modelId: string;
  result: Entailment[] | null;
  /**
   * Score each sentence against the passage. **One forward pass per sentence**
   * — the count is `sentences.length`, and the page says so beforehand.
   */
  run: (passage: string, sentences: readonly string[]) => Promise<Entailment[]>;
}

export function useEntailment(autoLoad = false): UseEntailmentResult {
  const meta =
    ZERO_SHOT_TEXT_MODELS.find((m) => m.id === FAITHFULNESS_MODEL) ??
    ZERO_SHOT_TEXT_MODELS[0];
  const pipe = useTextPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<Entailment[] | null>(null);

  const run = useCallback(
    async (
      passage: string,
      sentences: readonly string[],
    ): Promise<Entailment[]> => {
      const out: Entailment[] = [];
      // Sequential, never `Promise.all`: two overlapping calls into one ONNX
      // session is not a guarantee worth relying on, and on WASM a fan-out
      // would queue anyway (the `useImageFeatures` rule).
      for (const sentence of sentences) {
        const raw = (await post(passage, [
          [sentence],
          // The sentence *is* the hypothesis. Without this the pipeline wraps
          // it in "This example is {}." and scores something nobody wrote —
          // the `hypothesis_template` trap, two pages running.
          { hypothesis_template: BARE_HYPOTHESIS_TEMPLATE },
        ])) as RawZeroShot | undefined;
        out.push({ sentence, score: Number(raw?.scores?.[0] ?? 0) });
      }
      setResult(out);
      return out;
    },
    [post],
  );

  return { ...pipe, modelId: meta.id, result, run };
}
