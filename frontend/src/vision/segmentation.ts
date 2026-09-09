// Image-segmentation model catalogue. **Semantic segmentation only** — one class
// label per pixel. Instance and panoptic segmentation have almost no browser
// path (Mask2Former, OneFormer and EoMT publish no ONNX export at all), and the
// page says which of the three it is doing rather than leaving the user to guess.
//
// The output shape drives the whole route: a pipeline returns
// `[{ label, score, mask: RawImage }, …]` — **one single-channel mask per class
// present**, not one indexed label map. They composite into a single canvas with
// a per-label colour; rendering 150 separate images is the failure mode
// `drawMasks` exists to prevent.
//
// Each entry states its **class space**, because a segmenter can only ever say
// what its training set contained: asking the face parser about a street scene
// gets you a confident answer made entirely of face parts.
//
// Sizes read off the Hub's blob listing:
//
//   SegFormer-B0 ADE   fp16  7.6 MB · q8  4.2 MB
//   Face parsing       fp16 164 MB  · q8 85 MB
//   Clothes (B2)       fp32 105 MB, both backends — see the note below
//   DETR panoptic      fp16 82.6 MB · q8 42.4 MB

import type { VisionModel } from "./types";

export interface SegmenterModel extends VisionModel {
  task: "image-segmentation";
  /** What this model can possibly say — its label space, in words. */
  classes: string;
  /** Semantic (a label per pixel) or panoptic (things and stuff, with instances). */
  kind: "semantic" | "panoptic";
}

export const SEGMENTER_MODELS: SegmenterModel[] = [
  {
    id: "Xenova/segformer-b0-finetuned-ade-512-512",
    label: "SegFormer-B0 (ADE20K)",
    hint: "4 MB, 150 everyday scene classes. The general-purpose default.",
    params: 3.8,
    task: "image-segmentation",
    kind: "semantic",
    classes: "150 ADE20K scene classes — sky, building, road, person, tree…",
    bytes: { webgpu: 7_939_696, wasm: 4_418_863 },
  },
  {
    id: "Xenova/face-parsing",
    label: "Face parsing",
    hint: "Splits a face into its parts. Point it at a portrait, not a street.",
    params: 84.7,
    task: "image-segmentation",
    kind: "semantic",
    classes: "19 face parts — skin, eyes, brows, nose, lips, hair, neck, clothing",
    bytes: { webgpu: 171_716_570, wasm: 89_439_678 },
  },
  {
    id: "mattmdjaga/segformer_b2_clothes",
    label: "Clothes parsing (B2)",
    hint: "Garment-by-garment on a person. 105 MB — the repo ships fp32 only.",
    params: 27.4,
    task: "image-segmentation",
    kind: "semantic",
    classes: "18 garment and body classes — hat, upper clothes, skirt, pants, shoes…",
    // **fp32 on both backends, and this is a property of the repo, not caution.**
    // It publishes a single `onnx/model.onnx` and no fp16 or quantized export,
    // so `loadOpts()` — fp16 on WebGPU, q8 on WASM — cannot resolve a file and
    // the load 404s. Same situation as `Xenova/mobilevitv2-1.0-imagenet1k-256`,
    // which is absent from the classification catalogue for exactly this reason;
    // this one earns its place because pinning fp32 makes it work, at a 105 MB
    // download that the params estimate would have quoted as 55 MB.
    dtypes: { webgpu: "fp32", wasm: "fp32" },
    bytes: { webgpu: 110_039_290, wasm: 110_039_290 },
  },
  {
    id: "Xenova/detr-resnet-50-panoptic",
    label: "DETR panoptic",
    hint: "Things and stuff together — the only panoptic head with a browser export.",
    params: 42.9,
    task: "image-segmentation",
    kind: "panoptic",
    classes: "COCO things (person, car, dog…) plus stuff (sky, grass, road…)",
    bytes: { webgpu: 86_559_030, wasm: 44_483_407 },
  },
];

export const DEFAULT_SEGMENTER = SEGMENTER_MODELS[0].id;

/** Starting overlay opacity. Enough to read the classes, not enough to hide the photo. */
export const DEFAULT_MASK_ALPHA = 0.55;

/** Longest side fed to the model — resolution is the throttle, not the model. */
export const MAX_INFERENCE_SIDE = 640;
