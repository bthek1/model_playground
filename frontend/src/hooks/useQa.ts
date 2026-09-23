// Extractive question answering. The one NLP hook that does **not** wrap
// `useTextPipeline`, for the reason `text/qa/types.ts` sets out: the
// `question-answering` pipeline returns `{ answer, score }` and discards the
// token indices it chose, so it cannot tell a page *where* in the passage the
// answer is — which is this page's entire output. `text/qa/` drives the
// tokenizer and model directly and keeps them.
//
// The plumbing is still `model/useModelWorker.ts`. The worker lifecycle, the
// id-correlated pending table and the two state machines of
// docs/standards/model-page-pattern.md §2 are not re-derived here; the only
// thing this hook owns is the catalogue lookup and the run's typed shape.

import { useCallback, useMemo, useState } from "react";

import type { LoadProgress } from "@/model/progress";
import type { ModelProgress, ModelStatus } from "@/model/types";
import { useModelWorker } from "@/model/useModelWorker";
import { DEFAULT_QA_MODEL, QA_MODELS } from "@/text/catalogue";
import { createQaWorker } from "@/text/qa/client";
import type { QaAnswer } from "@/text/qa/types";

export interface UseQaResult {
  status: ModelStatus;
  /** True before the user has asked for the weights. */
  idle: boolean;
  loading: boolean;
  ready: boolean;
  progress: ModelProgress | null;
  /** Aggregate load progress across every file. Null outside `loading`. */
  loadProgress: LoadProgress | null;
  /** Duration of the load that produced `ready`, in ms. */
  loadedInMs: number | null;
  backend: string | null;
  running: boolean;
  error: string | null;
  result: QaAnswer | null;
  /** Ask `question` of `context`. Resolves with the span and its score. */
  run: (question: string, context: string) => Promise<QaAnswer>;
  load: () => void;
  retry: (overrides?: Record<string, unknown>) => void;
  cancel: () => void;
}

export function useQa(
  model: string = DEFAULT_QA_MODEL,
  autoLoad = false,
): UseQaResult {
  const meta = useMemo(
    () => QA_MODELS.find((m) => m.id === model) ?? QA_MODELS[0],
    [model],
  );

  const worker = useModelWorker<QaAnswer>({
    createWorker: createQaWorker,
    key: `question-answering:${meta.id}`,
    loadMessage: meta.dtypes
      ? { model: meta.id, dtypes: meta.dtypes }
      : { model: meta.id },
    autoLoad,
    notReadyMessage: "Question-answering worker not ready",
  });

  const { run: post } = worker;
  const [result, setResult] = useState<QaAnswer | null>(null);

  const run = useCallback(
    async (question: string, context: string): Promise<QaAnswer> => {
      const answer = (await post({ question, context })) as QaAnswer;
      setResult(answer);
      return answer;
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
