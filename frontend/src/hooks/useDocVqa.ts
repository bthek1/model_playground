// Document Question Answering — a photographed document and a question in, the
// answer extracted from it out.
//
// The shared contract verbatim over `model/useModelWorker`. No streaming: Donut
// answers with a short extracted span rather than prose, so there is no
// multi-second stretch of silence to narrate and no `partial` to surface.
//
// The question is held by the caller, not here. It is INPUT: editing it must
// cost nothing until GENERATE is pressed.

import { useCallback, useMemo, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import type { LoadProgress } from "@/model/progress";
import type { ModelProgress, ModelStatus } from "@/model/types";
import { useModelWorker } from "@/model/useModelWorker";
import { createDocVqaWorker } from "@/multimodal/docvqa/client";
import {
  DEFAULT_DOCVQA_MODEL,
  DOCVQA_MODELS,
  MAX_NEW_TOKENS,
  type DocVqaResult,
} from "@/multimodal/docvqa/types";
import { toPayload, transferablesOf } from "@/vision/image";

export interface UseDocVqaResult {
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
  result: DocVqaResult | null;
  /** Ask `question` of `document`. */
  run: (document: RawImage, question: string) => Promise<DocVqaResult>;
  load: () => void;
  retry: (overrides?: Record<string, unknown>) => void;
  cancel: () => void;
}

export function useDocVqa(
  model: string = DEFAULT_DOCVQA_MODEL,
  autoLoad = false,
): UseDocVqaResult {
  const meta = useMemo(
    () => DOCVQA_MODELS.find((m) => m.id === model) ?? DOCVQA_MODELS[0],
    [model],
  );

  const loadMessage = useMemo(
    () => ({
      model: meta.id,
      ...(meta.dtypes ? { dtypes: meta.dtypes } : {}),
    }),
    [meta],
  );

  const worker = useModelWorker<DocVqaResult>({
    createWorker: createDocVqaWorker,
    key: `docvqa:${meta.id}`,
    loadMessage,
    autoLoad,
    notReadyMessage: "Document QA worker not ready",
  });

  const [result, setResult] = useState<DocVqaResult | null>(null);

  const { run: post } = worker;
  const run = useCallback(
    async (document: RawImage, question: string): Promise<DocVqaResult> => {
      const asked = question.trim();
      if (!asked) throw new Error("Ask a question about the document");
      // `copy: false` — the payload is built for this post and the page keeps
      // its own `RawImage` for the preview.
      const payload = toPayload(document, { copy: false });
      const res = await post(
        { image: payload, question: asked, maxNewTokens: MAX_NEW_TOKENS },
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
