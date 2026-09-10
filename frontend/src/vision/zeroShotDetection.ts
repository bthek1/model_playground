// Zero-shot object detection: the user types phrases, the model draws boxes.
// #15 without the fixed class list, #13 with localisation — and the page where
// the two halves of the category finally meet.
//
// **The prompt is the model here, and more literally than on /zero-shot-image-
// classification.** That page applies a template (`"a photo of a {}"`) around a
// bare noun, because CLIP was trained on captions and the wording moves the
// numbers. This pipeline applies **no template at all**: `candidate_labels` are
// tokenized verbatim and handed to the text tower, so what the user types is
// exactly what the model is asked. That is why the defaults below carry their
// articles (`"a person"`, not `"person"`) — an OWL-ViT query reads as a noun
// phrase, and stripping the article measurably weakens it.
//
// Sizes are the ONNX blob totals read off the Hub, not a params estimate: OWLv2
// carries a text tower and a vision tower in one graph and the estimate would be
// badly wrong. All three pass `LARGE_MODEL_BYTES` on WebGPU, which is correct —
// this is the most expensive route in the category after SigLIP 2, and the
// picker says so before anything downloads.
//
//   OWLv2 base/16 ensemble   fp16 308 MB · q8 155 MB
//   OWL-ViT base/32          fp16 307 MB · q8 155 MB
//   Grounding DINO tiny      fp16 360 MB · q8 204 MB
//
// `iSEE-Laboratory/llmdet_tiny` is deliberately absent: no ONNX export exists,
// so it stays a table row in docs/roadmaps/vision.md.

import type { VisionModel } from "./types";

export interface ZeroShotDetectorModel extends VisionModel {
  task: "zero-shot-object-detection";
  /**
   * How the checkpoint reads its queries — and therefore what the user should
   * type. `label` models score each phrase as a candidate class; `phrase`
   * models (Grounding DINO) ground free text against the image and will happily
   * return a fragment of the query as the label.
   */
  queries: "label" | "phrase";
}

export const ZERO_SHOT_DETECTOR_MODELS: ZeroShotDetectorModel[] = [
  {
    id: "Xenova/owlv2-base-patch16-ensemble",
    label: "OWLv2 base/16",
    hint: "Self-trained on far more pseudo-labelled data than OWL-ViT. The default, and the one to beat.",
    params: 154,
    task: "zero-shot-object-detection",
    queries: "label",
    bytes: { webgpu: 307_904_711, wasm: 155_312_754 },
  },
  {
    id: "Xenova/owlvit-base-patch32",
    label: "OWL-ViT base/32",
    hint: "The original. Coarser patches, so it is quicker and misses small objects OWLv2 finds.",
    params: 153,
    task: "zero-shot-object-detection",
    queries: "label",
    bytes: { webgpu: 306_775_986, wasm: 155_431_700 },
  },
  {
    id: "onnx-community/grounding-dino-tiny-ONNX",
    label: "Grounding DINO tiny",
    hint: "Detection as phrase grounding: give it a sentence, not a class name. The interesting contrast.",
    params: 172,
    task: "zero-shot-object-detection",
    queries: "phrase",
    bytes: { webgpu: 360_393_267, wasm: 203_824_675 },
  },
];

export const DEFAULT_ZERO_SHOT_DETECTOR = ZERO_SHOT_DETECTOR_MODELS[0].id;

/**
 * Score below which a detection is dropped *by the model*.
 *
 * Deliberately far below anything the page shows, for the same reason as
 * `detection.ts`: the model is asked once for everything plausible and the
 * page's slider re-filters that list, which is a pure derivation and needs no
 * second inference. A slider that re-runs a 300 MB model is a slider nobody
 * drags.
 *
 * Grounding DINO takes this value as its `text_threshold` as well as its
 * `box_threshold`, so at 0.02 its labels can come back as ragged fragments of
 * the query. That is the honest low-confidence tail, not a bug — drag the
 * slider up and it disappears.
 */
export const MODEL_THRESHOLD = 0.02;

/**
 * Where the page's slider starts.
 *
 * **Much lower than a closed-vocabulary detector's 0.4.** An open-vocabulary
 * model spreads its probability mass over an unbounded label space, so OWLv2's
 * confident hits land around 0.1–0.3 where D-FINE's land around 0.7. Starting
 * at 0.4 here shows an empty canvas on a picture full of correctly-found
 * objects, which reads as a broken page rather than a badly-chosen default.
 */
export const DEFAULT_THRESHOLD = 0.1;

/** The slider's range. Above 0.5 an open-vocabulary detector finds nothing. */
export const THRESHOLD_RANGE = { min: 0.02, max: 0.5, step: 0.01 } as const;

/**
 * Starting queries that say something about the bundled city-street sample.
 *
 * **Articles included, unlike `DEFAULT_LABELS` in `zeroShot.ts`.** No template
 * is applied on this route — see the header.
 */
export const DEFAULT_QUERIES = ["a person", "a car", "a traffic light"];

/** Longest side fed to the model — resolution is the throttle, not the model. */
export const MAX_INFERENCE_SIDE = 640;
