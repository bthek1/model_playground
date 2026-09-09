// Depth-estimation model catalogue.
//
// One forward pass produces a per-pixel depth map — the best-looking output in
// the category, and structurally the simplest after classification.
//
// **What comes back is *relative* depth, not metres**, for everything here
// except Depth Pro. The values are an inverse-depth map on an arbitrary,
// per-image scale: comparing two frames without aligning them first is
// meaningless. That is a fact about the models, so it belongs in the UI copy on
// `/depth`, not only in this comment.
//
// Sizes are read off the Hub's blob listing, not estimated:
//
//   Depth Anything V2 Small   fp16 47.3 MB · q8 26.0 MB
//   Depth Anything Small      fp16 47.6 MB · q8 26.2 MB
//   Depth Pro                 q8  962 MB  (fp16 is 1.8 GB — see below)

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
  {
    id: "onnx-community/DepthPro-ONNX",
    label: "Depth Pro (metric)",
    hint: "Real metres, plus a focal-length estimate — and roughly 1 GB to download.",
    params: 952,
    task: "depth-estimation",
    metric: true,
    // WebGPU only, and **quantized on both**: the fp16 export is 1.8 GB, which
    // is past what a tab will reliably hold alongside a WebGPU context. q8 is
    // still ~1 GB, which is why this entry is gated behind an explicit opt-in
    // the way `/text-to-audio` gates MusicGen rather than merely warned about.
    backends: ["webgpu"],
    dtypes: { webgpu: "q8", wasm: "q8" },
    bytes: { webgpu: 1_009_098_757, wasm: 1_009_098_757 },
  },
];

export const DEFAULT_DEPTH_MODEL = DEPTH_MODELS[0].id;

/**
 * Models heavy enough that the page states the cost and downloads nothing until
 * the user opts in a second time. One entry today; the predicate is the contract.
 */
export function isHeavy(model: DepthModel): boolean {
  return model.id === "onnx-community/DepthPro-ONNX";
}

/**
 * Longest side fed to the model. Depth transformers resize to their own fixed
 * input anyway, so this only caps the cost of the pixels we hand across the
 * worker boundary — and on a live camera that cost is real.
 */
export const MAX_INFERENCE_SIDE = 640;
