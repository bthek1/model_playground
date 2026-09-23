// Depth-estimation model catalogue.
//
// One forward pass produces a per-pixel depth map — the best-looking output in
// the category, and structurally the simplest after classification.
//
// **What comes back is *relative* depth, not metres.** The values are an
// inverse-depth map on an arbitrary, per-image scale: comparing two frames
// without aligning them first is meaningless. That is a fact about the models,
// so it belongs in the UI copy on `/depth`, not only in this comment.
//
// Sizes are read off the Hub's blob listing, not estimated:
//
//   Depth Anything V2 Small   fp16 47.3 MB · q8 26.0 MB
//   Depth Anything Small      fp16 47.6 MB · q8 26.2 MB
//
// **Depth Pro used to be the third entry and was cut for size.** It is the only
// metric-depth export that exists for the browser, and it is 1009 MB at q8
// (the fp16 build is 1.8 GB) — twenty times either model above, on a page whose
// point is that one forward pass is interactive. `metric` stays on the
// interface because the legend direction is read off the catalogue rather than
// hard-coded, and a metric model is what would set it again.

import { isHeavyDownload } from "@/model/size";

import type { VisionModel } from "./types";

export interface DepthModel extends VisionModel {
  task: "depth-estimation";
  /** True when the output is in metres. Only Depth Pro is. */
  metric?: boolean;
}

export const DEPTH_MODELS: DepthModel[] = [
  {
    id: "onnx-community/depth-anything-v2-small",
    label: "Depth Anything V2 Small",
    hint: "25M params, one pass, interactive on a GPU. Relative depth — the default.",
    params: 24.8,
    task: "depth-estimation",
    bytes: { webgpu: 49_642_442, wasm: 27_258_801 },
  },
  {
    id: "Xenova/depth-anything-small-hf",
    label: "Depth Anything (v1)",
    hint: "The older mirror of the same architecture. Equivalent, and still maintained.",
    params: 24.8,
    task: "depth-estimation",
    bytes: { webgpu: 49_861_035, wasm: 27_524_771 },
  },
];

export const DEFAULT_DEPTH_MODEL = DEPTH_MODELS[0].id;

/**
 * Models heavy enough that the page states the cost and downloads nothing until
 * the user opts in a second time.
 *
 * **No shipped entry on this page trips it today**, and that is the intended
 * state: Depth Pro did, and was cut for it. The predicate is a size test rather
 * than a model id precisely so it survived that cut — an id check would have
 * had to be deleted with its subject, taking the gate with it, and the next
 * heavy entry would have arrived ungated. It survived twice over: the threshold
 * and the test now live in `model/size.ts`, where
 * `/zero-shot-classification` picked them up for BART-large-MNLI rather than
 * writing a second copy.
 */
export function isHeavy(model: DepthModel): boolean {
  return isHeavyDownload(model.bytes);
}

/**
 * Longest side fed to the model. Depth transformers resize to their own fixed
 * input anyway, so this only caps the cost of the pixels we hand across the
 * worker boundary — and on a live camera that cost is real.
 */
export const MAX_INFERENCE_SIDE = 640;
