// In-browser image classification. A thin wrapper over the generic
// `useVisionPipeline` worker — it picks the catalogue entry, pins `top_k`, and
// types the result. Everything else (worker lifecycle, the pending table, the
// two state machines) belongs to `model/useModelWorker.ts` and is not
// re-derived here.
//
// The returned shape is the shared `ModelTask` contract, verbatim: `run`, not a
// `classify()` alias (docs/standards/model-page-pattern.md §3).

import { useCallback, useMemo, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import type { ClassLabel } from "@/model/types";
import {
  useVisionPipeline,
  type UseVisionPipelineResult,
} from "@/hooks/useVisionPipeline";
import {
  DEFAULT_IMAGE_CLASSIFIER,
  IMAGE_CLASSIFIER_MODELS,
  TOP_K,
} from "@/vision/classification";

export interface UseImageClassifierResult
  extends Omit<UseVisionPipelineResult, "run"> {
  /** Latest ranked predictions, highest score first. */
  result: ClassLabel[] | null;
  /** Classify one image. Resolves with the same list it stores in `result`. */
  run: (image: RawImage) => Promise<ClassLabel[]>;
}

export function useImageClassifier(
  model: string = DEFAULT_IMAGE_CLASSIFIER,
  autoLoad = false,
): UseImageClassifierResult {
  const meta = useMemo(
    () =>
      IMAGE_CLASSIFIER_MODELS.find((m) => m.id === model) ??
      IMAGE_CLASSIFIER_MODELS[0],
    [model],
  );
  const pipe = useVisionPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<ClassLabel[] | null>(null);

  const run = useCallback(
    async (image: RawImage): Promise<ClassLabel[]> => {
      const out = (await post(image, [{ top_k: TOP_K }])) as ClassLabel[];
      setResult(out);
      return out;
    },
    [post],
  );

  return { ...pipe, result, run };
}
