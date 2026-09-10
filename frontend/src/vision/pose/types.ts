// Catalogue and worker protocol for Keypoint Detection (pose).
//
// **This is the single deliberate exception to "one model live at a time"**
// (docs/roadmaps/vision.md §5). Top-down pose is two networks — a detector finds
// people, then the pose model runs on each person's crop — and neither half is
// useful alone. So a catalogue entry here names *both* checkpoints and quotes
// their **combined** download, because a size guardrail that quotes half the
// bytes is worse than none.
//
// The memory budget is stated rather than hoped for: nano D-FINE plus ViTPose
// base is ~180 MB of fp16 weights, which fits a tab comfortably. Two base-sized
// models would not, which is why the alternative detector below is marked
// still-image only.

import type { Backend, DtypeSpec, LoadOpts } from "@/model/backend";
import type { ModelRequest, ModelResponse } from "@/model/types";
import type { MeasuredBytes } from "@/model/size";

import type { ImagePayload } from "../image";
import type { Person } from "./skeleton";

/** One half of a pair: a repo plus the graphs and precision it needs. */
export interface PoseStage {
  id: string;
  /** ONNX graph base names, for the Hub-id spec. */
  graphs: readonly string[];
  bytes: MeasuredBytes;
  dtypes?: Partial<Record<Backend, DtypeSpec>>;
}

export interface PosePairEntry {
  /** Composite id — the pair is what the user picks, not either half. */
  id: string;
  label: string;
  hint: string;
  /** Combined parameter count in millions, for the shared `ModelPicker`. */
  params: number;
  /** **Combined** download. The guardrail must fire on the sum. */
  bytes: MeasuredBytes;
  detector: PoseStage;
  pose: PoseStage;
  backends?: readonly Backend[];
  /** Fast enough for a webcam. Two forward passes per person, so: only the nano. */
  live?: boolean;
}

/**
 * The catalogue. Sizes are ONNX blob totals read off the Hub.
 *
 *   D-FINE nano   fp16   7.9 MB · q8   4.5 MB
 *   RT-DETR r50   fp16  88.0 MB · q8  45.5 MB   (WebGPU only)
 *   ViTPose base  fp16 172.1 MB · q8  87.5 MB
 *
 * `facebook/sapiens2-pose-0.4b` (308 keypoints) and the SuperPoint /
 * LightGlue interest-point line have no export and stay server-side.
 */
export const POSE_MODELS: PosePairEntry[] = [
  {
    id: "dfine-n+vitpose-base",
    label: "D-FINE nano + ViTPose base",
    hint: "The pair that fits a tab: a 4 MB detector and the standard 17-keypoint pose model.",
    params: 90,
    // 7_867_551 + 172_069_880 fp16 · 4_477_982 + 87_479_939 q8
    bytes: { webgpu: 179_937_431, wasm: 91_957_921 },
    live: true,
    detector: {
      id: "onnx-community/dfine_n_coco-ONNX",
      graphs: ["model"],
      bytes: { webgpu: 7_867_551, wasm: 4_477_982 },
    },
    pose: {
      id: "onnx-community/vitpose-base-simple",
      graphs: ["model"],
      bytes: { webgpu: 172_069_880, wasm: 87_479_939 },
    },
  },
  {
    id: "rtdetr-r50+vitpose-base",
    label: "RT-DETR r50 + ViTPose base",
    hint: "A much stronger detector, and 260 MB of weights live at once. Still images only.",
    params: 133,
    // 87_976_913 + 172_069_880 fp16 · 45_463_461 + 87_479_939 q8
    bytes: { webgpu: 260_046_793, wasm: 132_943_400 },
    // RT-DETR r50's q8 export is the reason `/object-detection` pins it to
    // WebGPU; the pair inherits that.
    backends: ["webgpu"],
    detector: {
      id: "onnx-community/rtdetr_r50vd",
      graphs: ["model"],
      bytes: { webgpu: 87_976_913, wasm: 45_463_461 },
    },
    pose: {
      id: "onnx-community/vitpose-base-simple",
      graphs: ["model"],
      bytes: { webgpu: 172_069_880, wasm: 87_479_939 },
    },
  },
];

export const DEFAULT_POSE_MODEL = POSE_MODELS[0].id;

/** Ask the detector for everything plausible; the page's slider filters. */
export const DETECTOR_THRESHOLD = 0.05;

/** Where the person-confidence slider starts. */
export const DEFAULT_PERSON_THRESHOLD = 0.4;

/** Default cap on people. Each one is another pose forward pass. */
export const DEFAULT_MAX_PEOPLE = 5;

/** Hard cap, so a crowd scene cannot wedge the tab. */
export const MAX_PEOPLE_LIMIT = 12;

/** Longest side fed to the detector — resolution is the throttle. */
export const MAX_INFERENCE_SIDE = 640;

export interface PoseLoad {
  /** The pair's composite id; the worker resolves both halves from it. */
  model: string;
  opts?: LoadOpts;
}

export interface PoseRun {
  image: ImagePayload;
  threshold: number;
  maxPeople: number;
}

export interface PoseResult {
  people: Person[];
  /** People the detector found above the threshold, before the cap. */
  detected: number;
  /** Milliseconds in the detector, and in the pose model, separately —
   *  this page's whole shape is two models, so it says what each cost. */
  detectMs: number;
  poseMs: number;
}

export type PoseRequest = ModelRequest<PoseLoad, PoseRun>;
export type PoseResponse = ModelResponse<PoseResult>;
