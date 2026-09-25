// Text generation — the standard hook contract plus `partial`, and nothing
// bespoke. That is the claim #30 made when it put the `partial` variant in the
// **shared** `ModelResponse` envelope rather than in a private VLM protocol
// ("NLP text-generation will need exactly the same thing"), and this is the page
// that tests it. Streaming needed nothing the envelope did not already have.
//
// Everything else belongs to `model/useModelWorker.ts`: the worker lifecycle,
// the id-correlated pending table, `running` as an inflight count, and the rule
// that **a partial whose request has already settled is dropped** so a late
// chunk cannot repaint a finished answer.
//
// The prompt and the decoding parameters are held by the caller. They are
// INPUT: editing any of them must cost nothing until GENERATE is pressed.

import { useCallback, useMemo, useState } from "react";

import type { LoadProgress } from "@/model/progress";
import type { ModelProgress, ModelStatus } from "@/model/types";
import { useModelWorker } from "@/model/useModelWorker";
import { createTextGenWorker } from "@/text/client";
import {
  DEFAULT_TEXTGEN_MODEL,
  TEXTGEN_MODELS,
} from "@/text/catalogue";
import type {
  Decoding,
  TextGenModel,
  TextGenPartial,
  TextGenResult,
} from "@/text/textgenTypes";

export interface UseTextGenResult {
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
  /** The selected entry — `instruct` and the size line come from it. */
  meta: TextGenModel;
  result: TextGenResult | null;
  /**
   * The text so far, while a run is in flight. Null otherwise, so a page
   * renders `partial` while it exists and `result` afterwards — never both, and
   * never a half-finished answer beside a finished one.
   */
  partial: TextGenPartial | null;
  run: (prompt: string, decoding: Decoding) => Promise<TextGenResult>;
  load: () => void;
  retry: (overrides?: Record<string, unknown>) => void;
  cancel: () => void;
}

export function useTextGen(
  model: string = DEFAULT_TEXTGEN_MODEL,
  autoLoad = false,
): UseTextGenResult {
  const meta = useMemo(
    () => TEXTGEN_MODELS.find((m) => m.id === model) ?? TEXTGEN_MODELS[0],
    [model],
  );

  const loadMessage = useMemo(
    () => ({
      model: meta.id,
      ...(meta.dtypes ? { dtypes: meta.dtypes } : {}),
      ...(meta.modelFile ? { modelFile: meta.modelFile } : {}),
      ...(meta.requireShaderF16 ? { requireShaderF16: true } : {}),
    }),
    [meta],
  );

  const worker = useModelWorker<TextGenResult, TextGenPartial>({
    createWorker: createTextGenWorker,
    key: `text-generation:${meta.id}`,
    loadMessage,
    autoLoad,
    notReadyMessage: "Text generation worker not ready",
  });

  const { run: post } = worker;
  const [result, setResult] = useState<TextGenResult | null>(null);

  const run = useCallback(
    async (prompt: string, decoding: Decoding): Promise<TextGenResult> => {
      const out = await post({
        prompt,
        decoding,
        // A bare LM continues text; an instruct model is asked a question.
        // Sending a raw prompt to an instruct model produces a fluent
        // non-answer with nothing failing, so which mode applies is a fact
        // about the checkpoint rather than a page setting.
        chat: meta.instruct,
      });
      setResult(out);
      return out;
    },
    [post, meta.instruct],
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
    meta,
    result,
    partial: worker.partial,
    run,
    load: worker.load,
    retry: worker.retry,
    cancel: worker.cancel,
  };
}
