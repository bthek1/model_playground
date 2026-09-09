// Object-detection model catalogue — the flagship live demo of the category.
//
// **`percentage: false` is pinned in `useObjectDetector`, not left to a caller.**
// Transformers.js returns 0–1 fractions by default; `drawBoxes` wants absolute
// pixels. Getting it backwards collapses every box into the top-left corner, it
// is the single most common bug on a first detection page, and a unit test
// asserts the flag rather than trusting the comment.
//
// Sizes read off the Hub's blob listing:
//
//   D-FINE nano   fp16  7.5 MB · q8  4.3 MB
//   D-FINE small  fp16 20.1 MB · q8 10.7 MB
//   RT-DETR r50   fp16 83.9 MB · q8 43.4 MB
//   YOLOS small   fp16 58.9 MB · q8 54.3 MB
//   DETR r50      fp16 79.9 MB · q8 41.1 MB
//
// `Roboflow/rf-detr-*` is deliberately absent: no ONNX export exists, so it is a
// table row in docs/roadmaps/vision.md and nothing more.

import type { VisionModel } from "./types";

export interface DetectorModel extends VisionModel {
  task: "object-detection";
  /** Fast enough to drive a webcam at interactive rates. */
  live?: boolean;
}

export const DETECTOR_MODELS: DetectorModel[] = [
  {
    id: "onnx-community/dfine_n_coco-ONNX",
    label: "D-FINE nano",
    hint: "4 MB and real-time — the one to point a webcam at. The default.",
    params: 3.8,
    task: "object-detection",
    live: true,
    bytes: { webgpu: 7_867_551, wasm: 4_477_982 },
  },
  {
    id: "onnx-community/dfine_s_coco-ONNX",
    label: "D-FINE small",
    hint: "The localisation specialist: tighter boxes, still quick enough to be live.",
    params: 10.3,
    task: "object-detection",
    live: true,
    bytes: { webgpu: 21_041_872, wasm: 11_185_079 },
  },
  {
    id: "Xenova/yolos-small",
    label: "YOLOS small",
    hint: "A plain ViT that detects — no detection-specific machinery at all.",
    params: 30.7,
    task: "object-detection",
    bytes: { webgpu: 61_746_058, wasm: 56_963_687 },
  },
  {
    id: "onnx-community/rtdetr_r50vd",
    label: "RT-DETR r50",
    hint: "Real-time DETR at ResNet-50 scale. WebGPU only; the r18 export doesn't exist.",
    params: 42.9,
    task: "object-detection",
    backends: ["webgpu"],
    bytes: { webgpu: 87_976_913, wasm: 45_463_461 },
  },
  {
    id: "Xenova/detr-resnet-50",
    label: "DETR r50",
    hint: "The canonical set-prediction original. Slow, and worth running once.",
    params: 41.6,
    task: "object-detection",
    bytes: { webgpu: 83_812_437, wasm: 43_102_531 },
  },
];

export const DEFAULT_DETECTOR = DETECTOR_MODELS[0].id;

/**
 * Score below which a detection is dropped *by the model*.
 *
 * Deliberately low. The page's own threshold slider re-filters the list the
 * model already returned — a pure derivation, no second inference — so the
 * model is asked for everything plausible once and the user explores it for
 * free. Asking the model to filter would make every slider drag a re-run.
 */
export const MODEL_THRESHOLD = 0.05;

/** Where the page's slider starts. Above the noise, below the interesting cases. */
export const DEFAULT_THRESHOLD = 0.4;

/** Longest side fed to the model — resolution is the throttle, not the model. */
export const MAX_INFERENCE_SIDE = 640;
