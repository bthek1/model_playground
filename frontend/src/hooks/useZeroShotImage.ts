// Zero-shot image classification — the user's own labels, scored by CLIP/SigLIP.
//
// Unlike every other vision hook this one does **not** wrap `useVisionPipeline`,
// and the reason is the feature the page exists to demonstrate. The
// `zero-shot-image-classification` pipeline re-encodes the labels on every call;
// the label embeddings do not depend on the image, so on a live feed that is the
// text tower re-run per frame to produce identical numbers. `vision/zeroshot/`
// drives the two towers separately and keeps the text side
// (docs/roadmaps/vision.md §5, "encode once, decode many"). The plumbing is
// still `model/useModelWorker.ts` — the state machines and the pending table are
// not re-derived here.
//
// Two things this hook owns that no other vision hook does:
//
//  1. **The prompt template.** The towers are handed prompts, not labels, so
//     what comes back is scored against "a photo of a cat". The user typed
//     "cat", and that is what the page must show, so the mapping back is done
//     here — by position, because the towers answer in the order they were asked
//     and a template can collapse two labels to the same string.
//  2. **Scoring more than one template per run.** The page's whole thesis is
//     that wording moves the numbers, and a claim like that is only legible with
//     both columns on screen. Each template is its own softmax; concatenating
//     them into one call would make the wordings compete and mean nothing.
//
// The runs are sequential, not concurrent: two overlapping calls into one ONNX
// session is not a guarantee worth relying on. The second template is also a
// *cache miss on purpose* — different prompts, different embeddings — which is
// why the cache holds several sets at once (`TEXT_CACHE_LIMIT`).

import { useCallback, useMemo, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import type { LoadProgress } from "@/model/progress";
import type { ClassLabel, ModelProgress, ModelStatus } from "@/model/types";
import { useModelWorker } from "@/model/useModelWorker";
import { toPayload, transferablesOf } from "@/vision/image";
import {
  buildPrompts,
  DEFAULT_ZERO_SHOT_MODEL,
  scoringSpec,
  ZERO_SHOT_MODELS,
} from "@/vision/zeroShot";
import { createZeroShotWorker } from "@/vision/zeroshot/client";
import type { ZeroShotResult } from "@/vision/zeroshot/types";

/** One template's scores, in the user's own words. */
export interface TemplateScores {
  template: string;
  /** `label` is what the user typed, never the prompt that was sent. */
  scores: ClassLabel[];
  /** True when the label embeddings were reused rather than recomputed. */
  textCached: boolean;
  /** Text-tower milliseconds. Zero on a cache hit — the saving, measured. */
  textMs: number;
  /** Vision-tower milliseconds. Paid every run, cache or not. */
  imageMs: number;
}

export interface UseZeroShotImageResult {
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
  result: TemplateScores[] | null;
  /** Score `image` against `labels`, once per template, in template order. */
  run: (
    image: RawImage,
    labels: readonly string[],
    templates: readonly string[],
  ) => Promise<TemplateScores[]>;
  load: () => void;
  retry: (overrides?: Record<string, unknown>) => void;
  cancel: () => void;
}

export function useZeroShotImage(
  model: string = DEFAULT_ZERO_SHOT_MODEL,
  autoLoad = false,
): UseZeroShotImageResult {
  const meta = useMemo(
    () => ZERO_SHOT_MODELS.find((m) => m.id === model) ?? ZERO_SHOT_MODELS[0],
    [model],
  );

  const loadMessage = useMemo(
    () => ({
      model: meta.id,
      family: meta.family,
      scoring: scoringSpec(meta),
      ...(meta.dtypes ? { dtypes: meta.dtypes } : {}),
    }),
    [meta],
  );

  const worker = useModelWorker<ZeroShotResult>({
    createWorker: createZeroShotWorker,
    key: `zero-shot:${meta.id}`,
    loadMessage,
    autoLoad,
    notReadyMessage: "Zero-shot worker not ready",
  });

  const { run: post } = worker;
  const [result, setResult] = useState<TemplateScores[] | null>(null);

  const run = useCallback(
    async (
      image: RawImage,
      labels: readonly string[],
      templates: readonly string[],
    ): Promise<TemplateScores[]> => {
      const clean = labels.map((l) => l.trim()).filter(Boolean);
      if (clean.length === 0) throw new Error("Add at least one label to score");

      const out: TemplateScores[] = [];
      for (const template of templates) {
        const prompts = buildPrompts(template, clean);
        // `copy`, not transfer: the same image is scored again by the next
        // template, and a transferred buffer would arrive detached.
        const payload = toPayload(image);
        const res = await post({ image: payload, prompts }, transferablesOf(payload));
        out.push({
          template,
          scores: res.scores.map((score, i) => ({ label: clean[i], score })),
          textCached: res.textCached,
          textMs: res.textMs,
          imageMs: res.imageMs,
        });
      }
      setResult(out);
      return out;
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
