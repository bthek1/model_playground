// Summarization. A thin wrapper over the generic text worker; the interesting
// half of this page is not here.
//
// **`max_new_tokens` and `min_length` are run parameters, so they re-run.**
// They change the generation rather than a view of it — there is nothing in a
// finished summary from which a longer one could be derived — so the page lets
// them be edited for free and says that the next GENERATE is a real second
// inference. Same shape as `/video-text-to-text`'s reverse toggle and
// `/zero-shot-classification`'s `multi_label`.

import { useCallback, useMemo, useState } from "react";

import {
  useTextPipeline,
  type UseTextPipelineResult,
} from "@/hooks/useTextPipeline";
import {
  DEFAULT_SUMMARIZER,
  SUMMARIZER_MODELS,
  SUMMARY_MAX_TOKENS,
  SUMMARY_MIN_TOKENS,
  type SummarizerModel,
} from "@/text/catalogue";

/** One press of GENERATE. */
export interface SummarizeRun {
  text: string;
  maxNewTokens?: number;
  minLength?: number;
}

interface RawSummary {
  summary_text?: string;
}

export interface UseSummarizeResult extends Omit<UseTextPipelineResult, "run"> {
  meta: SummarizerModel;
  result: string | null;
  run: (request: SummarizeRun) => Promise<string>;
}

export function useSummarize(
  model: string = DEFAULT_SUMMARIZER,
  autoLoad = false,
): UseSummarizeResult {
  const meta = useMemo(
    () => SUMMARIZER_MODELS.find((m) => m.id === model) ?? SUMMARIZER_MODELS[0],
    [model],
  );
  const pipe = useTextPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<string | null>(null);

  const run = useCallback(
    async ({
      text,
      maxNewTokens = SUMMARY_MAX_TOKENS,
      minLength = SUMMARY_MIN_TOKENS,
    }: SummarizeRun): Promise<string> => {
      const raw = (await post(text, [
        { max_new_tokens: maxNewTokens, min_length: minLength },
      ])) as RawSummary[] | RawSummary | undefined;
      const first = Array.isArray(raw) ? raw[0] : raw;
      const out = String(first?.summary_text ?? "").trim();
      setResult(out);
      return out;
    },
    [post],
  );

  return { ...pipe, meta, result, run };
}
