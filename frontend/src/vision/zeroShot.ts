// Zero-shot image classification: the user types the labels and CLIP scores the
// picture against them. The most satisfying page in the category, and the
// cheapest live demo — the text side is encoded once and reused for every frame.
//
// **The prompt template is the point of this page.** `"a photo of a {}"` beats a
// bare `"{}"` by several points on the same model and the same picture, which is
// the single most instructive result in the vision roadmap and something very
// few demos anywhere show. The route puts both templates' scores side by side
// rather than making the user remember a before/after.
//
// The other thing the UI has to say plainly: **a zero-shot score is relative to
// the labels given.** CLIP softmaxes over exactly the list it was handed, so
// "cat 0.98" against ["cat", "dog"] means "more cat than dog", not "a cat is
// present". SigLIP is the interesting contrast — it was trained with a sigmoid
// loss, so its scores are independent per label and do not sum to 1.
//
// Sizes read off the Hub's blob listing:
//
//   CLIP ViT-B/32   fp16 289 MB · q8 147 MB
//   SigLIP base     fp16 388 MB · q8 201 MB
//   SigLIP 2 base   fp16 716 MB · q8 360 MB

import type { VisionModel } from "./types";

export interface ZeroShotModel extends VisionModel {
  task: "zero-shot-image-classification";
  /** Which pair of tower classes and which tokenizer padding this needs. */
  family: "clip" | "siglip";
  /** How the scores are produced — and therefore how to read them. */
  scoring: "softmax" | "sigmoid";
  /**
   * `exp(logit_scale)` from the checkpoint's own weights.
   *
   * Splitting the towers to cache the text side means we lose the tail of the
   * full graph and have to reapply this ourselves (`zeroshot/scoring.ts`). These
   * are **read out of the published weights**, not guessed: CLIP's raw
   * `logit_scale` is 4.605170, and `exp(4.605170) = 100.000006` — the ln(100)
   * clamp OpenAI trains against. SigLIP's come from `model.safetensors` on
   * `google/siglip-base-patch16-224` and `google/siglip2-base-patch16-224`.
   *
   * A wrong value keeps every score in [0, 1] and keeps the ranking intact, so
   * only comparing numbers against the full-graph pipeline catches it — which is
   * what the parity spec in `zero-shot-parity.spec.ts` does.
   */
  logitScale: number;
  /** `logit_bias` from the checkpoint. Sigmoid models only. */
  logitBias?: number;
}

export const ZERO_SHOT_MODELS: ZeroShotModel[] = [
  {
    id: "Xenova/clip-vit-base-patch32",
    label: "CLIP ViT-B/32",
    hint: "The reference model. Scores are a softmax over your labels — they always sum to 1.",
    params: 151,
    task: "zero-shot-image-classification",
    family: "clip",
    scoring: "softmax",
    logitScale: 100.000006, // exp(4.605170), read from pytorch_model.bin

    bytes: { webgpu: 303_515_168, wasm: 153_695_702 },
  },
  {
    id: "Xenova/siglip-base-patch16-224",
    label: "SigLIP base/16",
    hint: "Sigmoid loss: each label scored independently, so the numbers mean more.",
    params: 203,
    task: "zero-shot-image-classification",
    family: "siglip",
    scoring: "sigmoid",
    logitScale: 117.330795, // exp(4.764997)
    logitBias: -12.932437,

    bytes: { webgpu: 407_013_090, wasm: 210_977_441 },
  },
  {
    id: "onnx-community/siglip2-base-patch16-224-ONNX",
    label: "SigLIP 2 base/16",
    hint: "The current generation, and the best of the three. 360 MB even quantized.",
    params: 375,
    task: "zero-shot-image-classification",
    family: "siglip",
    scoring: "sigmoid",
    logitScale: 112.668907, // exp(4.724453)
    logitBias: -16.771725,

    bytes: { webgpu: 750_910_198, wasm: 378_000_135 },
  },
];

export const DEFAULT_ZERO_SHOT_MODEL = ZERO_SHOT_MODELS[0].id;

/**
 * Prompt templates, and the experiment this page exists to run. `{}` is
 * substituted with each label.
 */
export const TEMPLATES = {
  bare: "{}",
  photo: "a photo of a {}",
} as const;

export type TemplateKey = keyof typeof TEMPLATES;

/** Fill a template with one label. Exported because it is what the test pins. */
export function applyTemplate(template: string, label: string): string {
  return template.includes("{}") ? template.split("{}").join(label) : label;
}

/** Apply a template across a label list, in order. */
export function buildPrompts(
  template: string,
  labels: readonly string[],
): string[] {
  return labels.map((label) => applyTemplate(template, label));
}

/**
 * A starting set that says something interesting about the cats sample.
 *
 * **Bare nouns, no article** — the template supplies it. Phrases like `"a cat"`
 * compose with `"a photo of a {}"` into `"a photo of a a cat"`, which is both
 * visibly broken in the UI and a worse prompt than either template alone.
 */
export const DEFAULT_LABELS = ["cat", "dog", "empty sofa"];

/** Longest side fed to the model — resolution is the throttle, not the model. */
export const MAX_INFERENCE_SIDE = 640;

/**
 * How many distinct prompt sets keep their text embeddings live at once.
 *
 * More than one, and that is the point: the page scores two templates side by
 * side, so a single-entry cache would be evicted on every alternation and buy
 * nothing on exactly the screen this feature exists for. Eight is far more than
 * the UI can produce, and an entry is only `labels x dim` floats — five labels
 * at 768 dims is 15 KB.
 */
export const TEXT_CACHE_LIMIT = 8;

/** The scoring parameters, in the shape `zeroshot/scoring.ts` wants. */
export function scoringSpec(model: ZeroShotModel): {
  kind: "softmax" | "sigmoid";
  scale: number;
  bias?: number;
} {
  return {
    kind: model.scoring,
    scale: model.logitScale,
    ...(model.logitBias == null ? {} : { bias: model.logitBias }),
  };
}
