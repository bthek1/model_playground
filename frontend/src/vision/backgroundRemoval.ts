// Background-removal (image matting) catalogue.
//
// One forward pass produces a per-pixel **alpha matte**: how much of each pixel
// belongs to the subject rather than the background. Transformers.js wraps this
// as its own `background-removal` pipeline, which is the segmentation pipeline
// plus one step — it puts the matte into the source image's alpha channel and
// hands back an RGBA `RawImage`. So the model's answer and the cut-out arrive
// together, and `matte.ts` composites from there.
//
// ---
//
// **Licence, and it is the reason the default is what it is.**
//
// This repo is MIT. `briaai/RMBG-1.4` is not: it ships under `bria-rmbg-1.4`,
// a Creative Commons licence for **non-commercial use only**, with commercial
// use gated behind a paid agreement with BRIA. It is also the better matte —
// noticeably so on hair and fur — so the answer is not to pretend it doesn't
// exist. It is offered, and it is labelled, and it is never the default.
//
// MODNet is Apache-2.0, is Transformers.js's own default for this pipeline, and
// is 6.6 MB on WASM. That is the model someone gets by clicking Load without
// reading anything, which is the only place a licence constraint can actually
// bite. Nothing here redistributes weights either way — the browser fetches them
// from the Hub — but a default nobody downstream may legally use is still a
// trap, and `licence.commercial` is what puts it on screen.
//
// ---
//
// **The soft matte survives, and that is not guaranteed by the pipeline.**
//
// `ImageSegmentationPipeline` picks its post-processing by looking for a
// `post_process_*_segmentation` method on the processor. Find one and it takes
// the semantic branch, which is an argmax: every pixel becomes 0 or 255 and the
// soft edge — the entire point of a matting model — is gone, silently. Both
// entries below publish a plain `ImageFeatureExtractor` (no such method), so the
// pipeline falls through to the `!subtask` branch: sigmoid if the logits need
// it, scale to 0–255, resize to the source. A genuine soft matte.
//
// A matting checkpoint whose repo happened to ship a `SegformerImageProcessor`
// config would be hard-thresholded instead, and would still look plausible. If a
// third entry is ever added, check its `preprocessor_config.json` first.
//
// Sizes are read off the Hub's blob listing, not estimated:
//
//   MODNet      fp16 12.4 MB · q8  6.3 MB
//   RMBG-1.4    fp16 84.1 MB · q8 42.3 MB

import type { VisionModel } from "./types";

/** What a model's licence permits, in the terms a user has to act on. */
export interface ModelLicence {
  /** Short name, as the Hub states it. */
  name: string;
  /** True when the licence permits commercial use without a separate agreement. */
  commercial: boolean;
  /** The licence text or the page that gates it. */
  url: string;
  /** One sentence, shown when `commercial` is false. */
  note?: string;
}

export interface MatteModel extends VisionModel {
  task: "background-removal";
  licence: ModelLicence;
}

export const MATTE_MODELS: MatteModel[] = [
  {
    id: "Xenova/modnet",
    label: "MODNet",
    hint: "6.5M params, portrait matting, seconds on a CPU. Permissive — the default.",
    params: 6.5,
    task: "background-removal",
    bytes: { webgpu: 12_984_781, wasm: 6_632_188 },
    licence: {
      name: "Apache-2.0",
      commercial: true,
      url: "https://huggingface.co/Xenova/modnet",
    },
  },
  {
    id: "briaai/RMBG-1.4",
    label: "RMBG-1.4",
    hint: "44M params, general-purpose, a visibly better edge on hair and fur.",
    params: 44.1,
    task: "background-removal",
    bytes: { webgpu: 88_217_533, wasm: 44_403_226 },
    licence: {
      name: "bria-rmbg-1.4",
      commercial: false,
      url: "https://bria.ai/bria-huggingface-model-license-agreement/",
      note: "Released under a Creative Commons licence for non-commercial use. Commercial use needs a separate paid agreement with BRIA.",
    },
  },
];

export const DEFAULT_MATTE_MODEL = MATTE_MODELS[0].id;

/**
 * Longest side fed to the model.
 *
 * Higher than the 640 the detection routes use, and deliberately: a matte is
 * judged on its *edge*, and 640 px of a portrait leaves too few pixels across a
 * strand of hair for the difference between the two models here to be visible
 * at all. The pipeline resizes to the model's own input anyway (512 shortest
 * edge for MODNet, 1024x1024 for RMBG) and then resizes the matte back up, so
 * this caps what we carry across the worker boundary and what we composite —
 * both of which are per-pixel costs on the main thread.
 */
export const MAX_INFERENCE_SIDE = 1024;
