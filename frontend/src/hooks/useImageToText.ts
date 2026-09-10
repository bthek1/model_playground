// Image-to-text — captioning, OCR and grounding, from one download.
//
// The shared contract verbatim, over `model/useModelWorker`. It does **not**
// wrap `useVisionPipeline`, and the reason is in `vision/caption/types.ts`: the
// `image-to-text` pipeline cannot load Florence-2 at all (its model type is
// registered for image-text-to-text, not vision2seq) and has nowhere to put a
// task token even for the models it can load.
//
// The one piece of logic that lives here rather than in the route: **the mode is
// validated against the selected model's declared capabilities.** A task token a
// model has never seen does not error — it produces a confident, fluent,
// unrelated sentence — so an unsupported mode has to be caught before it is
// sent, not diagnosed afterwards from the output.

import { useCallback, useEffect, useMemo, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import type { LoadProgress } from "@/model/progress";
import type { ModelProgress, ModelStatus } from "@/model/types";
import { useModelWorker } from "@/model/useModelWorker";
import { createCaptionWorker } from "@/vision/caption/client";
import {
  CAPTION_MODELS,
  DEFAULT_CAPTION_MODEL,
  MAX_NEW_TOKENS,
  type CaptionMode,
  type CaptionResult,
} from "@/vision/caption/types";
import { toPayload, transferablesOf } from "@/vision/image";

export interface UseImageToTextResult {
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
  result: CaptionResult | null;
  /** The modes the selected checkpoint declares it can answer. */
  modes: readonly CaptionMode[];
  /** The current mode, always one of `modes`. */
  mode: CaptionMode;
  /** Choose a mode. A mode the model does not declare is ignored. */
  setMode: (mode: CaptionMode) => void;
  /** Describe `image` in the current mode. */
  run: (image: RawImage) => Promise<CaptionResult>;
  load: () => void;
  retry: (overrides?: Record<string, unknown>) => void;
  cancel: () => void;
}

export function useImageToText(
  model: string = DEFAULT_CAPTION_MODEL,
  autoLoad = false,
): UseImageToTextResult {
  const meta = useMemo(
    () => CAPTION_MODELS.find((m) => m.id === model) ?? CAPTION_MODELS[0],
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

  const worker = useModelWorker<CaptionResult>({
    createWorker: createCaptionWorker,
    key: `caption:${meta.id}`,
    loadMessage,
    autoLoad,
    notReadyMessage: "Caption worker not ready",
  });

  const [mode, setModeState] = useState<CaptionMode>(meta.modes[0]);
  const [result, setResult] = useState<CaptionResult | null>(null);

  // Switching to a model that cannot answer the current mode resets to that
  // model's first mode rather than sending a token it has never seen.
  useEffect(() => {
    setModeState((current) =>
      meta.modes.includes(current) ? current : meta.modes[0],
    );
  }, [meta]);

  const setMode = useCallback(
    (next: CaptionMode) => {
      if (!meta.modes.includes(next)) return;
      setModeState(next);
    },
    [meta],
  );

  const { run: post } = worker;
  const run = useCallback(
    async (image: RawImage): Promise<CaptionResult> => {
      if (!meta.modes.includes(mode)) {
        throw new Error(`${meta.label} cannot answer this mode`);
      }
      // `copy: false` — the payload is built for this post and the page keeps
      // its own `RawImage` for the preview and the overlay.
      const payload = toPayload(image, { copy: false });
      const res = await post(
        { image: payload, mode, maxNewTokens: MAX_NEW_TOKENS[mode] },
        transferablesOf(payload),
      );
      setResult(res);
      return res;
    },
    [post, mode, meta],
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
    modes: meta.modes,
    mode,
    setMode,
    run,
    load: worker.load,
    retry: worker.retry,
    cancel: worker.cancel,
  };
}
