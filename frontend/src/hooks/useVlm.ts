// Image-Text-to-Text — a picture and a question in, an answer in prose out.
//
// The shared contract verbatim over `model/useModelWorker`, plus the one field
// this family adds: `partial`, the in-run progress that makes the pre-token
// pause legible. It does **not** wrap `useVisionPipeline` — there is no
// `image-text-to-text` pipeline in 4.2.0 to wrap (see `multimodal/types.ts`).
//
// The prompt is held by the caller, not here. It is INPUT: editing it must cost
// nothing until GENERATE is pressed.

import { useCallback, useMemo, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import type { LoadProgress } from "@/model/progress";
import type { ModelProgress, ModelStatus } from "@/model/types";
import { useModelWorker } from "@/model/useModelWorker";
import { createVlmWorker } from "@/multimodal/client";
import {
  DEFAULT_MAX_NEW_TOKENS,
  DEFAULT_VLM_MODEL,
  VLM_MODELS,
  type VlmPartial,
  type VlmResult,
} from "@/multimodal/types";
import { toPayload, transferablesOf } from "@/vision/image";

export interface UseVlmResult {
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
  result: VlmResult | null;
  /**
   * In-run progress: `encoding` while the image is being encoded, then
   * `generating` with the text so far. Null when no run is in flight, so a page
   * renders `partial` while it exists and `result` afterwards, never both.
   */
  partial: VlmPartial | null;
  /** Answer `prompt` about `image`. */
  run: (
    image: RawImage,
    prompt: string,
    maxNewTokens?: number,
  ) => Promise<VlmResult>;
  load: () => void;
  retry: (overrides?: Record<string, unknown>) => void;
  cancel: () => void;
}

export function useVlm(
  model: string = DEFAULT_VLM_MODEL,
  autoLoad = false,
): UseVlmResult {
  const meta = useMemo(
    () => VLM_MODELS.find((m) => m.id === model) ?? VLM_MODELS[0],
    [model],
  );

  const loadMessage = useMemo(
    () => ({
      model: meta.id,
      family: meta.family,
      ...(meta.dtypes ? { dtypes: meta.dtypes } : {}),
    }),
    [meta],
  );

  const worker = useModelWorker<VlmResult, VlmPartial>({
    createWorker: createVlmWorker,
    key: `vlm:${meta.id}`,
    loadMessage,
    autoLoad,
    notReadyMessage: "Vision-language worker not ready",
  });

  const [result, setResult] = useState<VlmResult | null>(null);

  const { run: post } = worker;
  const run = useCallback(
    async (
      image: RawImage,
      prompt: string,
      maxNewTokens: number = DEFAULT_MAX_NEW_TOKENS,
    ): Promise<VlmResult> => {
      const question = prompt.trim();
      if (!question) throw new Error("Ask a question about the image");
      // `copy: false` — the payload is built for this post and the page keeps
      // its own `RawImage` for the preview.
      const payload = toPayload(image, { copy: false });
      const res = await post(
        { image: payload, prompt: question, maxNewTokens },
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
    partial: worker.partial,
    run,
    load: worker.load,
    retry: worker.retry,
    cancel: worker.cancel,
  };
}
