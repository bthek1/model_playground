// Image-classification model catalogue — the first Computer Vision route, and
// the simplest instance of the four-slot pattern in the app: one image in, a
// ranked label list out, no decode step at all.
//
// All three are ImageNet-1k (1000 classes) and run through the generic vision
// worker. Sizes below were read off the Hub's own blob listing rather than
// estimated, and the params figures are set so `sizeEstimate` lands on them:
//
//   ViT-base      fp16 173.5 MB · q8 88.3 MB
//   ResNet-50     fp16  51.1 MB · q8 25.8 MB
//   MobileNetV4   fp16   7.6 MB · q8  3.9 MB
//
// NOTE: `Xenova/mobilevitv2-1.0-imagenet1k-256` is deliberately absent. The repo
// publishes a single fp32 `model.onnx` and no fp16 or quantized export, so
// `loadOpts()` — which asks for fp16 on WebGPU and q8 on WASM — cannot resolve a
// file there. Re-check if that repo gains the other dtypes.

import type { VisionModel } from "./types";

export interface ClassifierModel extends VisionModel {
  task: "image-classification";
}

export const IMAGE_CLASSIFIER_MODELS: ClassifierModel[] = [
  {
    id: "Xenova/vit-base-patch16-224",
    label: "ViT-Base/16",
    hint: "The transformer baseline — most accurate of the three, and the slowest.",
    params: 86,
    task: "image-classification",
  },
  {
    id: "Xenova/resnet-50",
    label: "ResNet-50",
    hint: "The CNN baseline. A third of the download, and close behind on accuracy.",
    params: 26,
    task: "image-classification",
  },
  {
    id: "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k",
    label: "MobileNetV4 Small",
    hint: "Loads almost instantly, and shows you the honest accuracy floor.",
    params: 3.8,
    task: "image-classification",
    // **Not q8 on WASM, and this is measured, not cautious.** The quantized
    // export labels the bundled tiger photo "sidewinder, horned rattlesnake"
    // at 44% — it loads, it runs, and it is wrong, which is the failure mode
    // this repo has been bitten by before. At fp32 the same weights give
    // "tiger 62% / tiger cat 30%". Depthwise-separable convolutions are the
    // classic casualty of per-tensor int8 quantization, so this is a property
    // of the MobileNet family rather than a bad upload. Same shape of exception
    // as `asrLoadOpts()` keeping the Whisper decoder at fp32.
    dtypes: { wasm: "fp32" },
    // fp32 on WASM is 15.1 MB, not the 3.8 MB a q8 estimate implies. Measured,
    // because a quoted size that undersells the download by 4x is worse than
    // no quote at all.
    bytes: { webgpu: 7_563_434, wasm: 15_086_122 },
  },
];

export const DEFAULT_IMAGE_CLASSIFIER = IMAGE_CLASSIFIER_MODELS[0].id;

/**
 * How many labels to show.
 *
 * Five, not one, and this is a design decision rather than a default: the
 * interesting case is a 0.31 / 0.29 near-tie, and a single confident-looking
 * label hides exactly the failure the user most needs to see.
 */
export const TOP_K = 5;
