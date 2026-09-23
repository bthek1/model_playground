// In-browser text classification. A thin wrapper over the generic
// `useTextPipeline` worker — it picks the catalogue entry, pins `top_k`, and
// types the result. Everything else (worker lifecycle, the pending table, the
// two state machines) belongs to `model/useModelWorker.ts` and is not
// re-derived here.
//
// The returned shape is the shared `ModelTask` contract, verbatim: `run`, not a
// `classify()` alias (docs/standards/model-page-pattern.md §3).

import { useCallback, useMemo, useState } from "react";

import type { ClassLabel } from "@/model/types";
import {
  useTextPipeline,
  type UseTextPipelineResult,
} from "@/hooks/useTextPipeline";
import {
  DEFAULT_TEXT_CLASSIFIER,
  TEXT_CLASSIFIER_MODELS,
  TOP_K,
} from "@/text/catalogue";

export interface UseTextClassifierResult
  extends Omit<UseTextPipelineResult, "run"> {
  /** Latest ranked predictions, highest score first. */
  result: ClassLabel[] | null;
  /** Classify one string. Resolves with the same list it stores in `result`. */
  run: (text: string) => Promise<ClassLabel[]>;
}

export function useTextClassifier(
  model: string = DEFAULT_TEXT_CLASSIFIER,
  autoLoad = false,
): UseTextClassifierResult {
  const meta = useMemo(
    () =>
      TEXT_CLASSIFIER_MODELS.find((m) => m.id === model) ??
      TEXT_CLASSIFIER_MODELS[0],
    [model],
  );
  const pipe = useTextPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<ClassLabel[] | null>(null);

  const run = useCallback(
    async (text: string): Promise<ClassLabel[]> => {
      // `top_k` rather than the default single label: these heads have two to
      // six classes and a 0.51/0.49 split is the case worth seeing.
      const out = (await post(text, [{ top_k: TOP_K }])) as ClassLabel[];
      const list = Array.isArray(out) ? out : [out as ClassLabel];
      setResult(list);
      return list;
    },
    [post],
  );

  return { ...pipe, result, run };
}
