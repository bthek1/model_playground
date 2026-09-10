// Task-agnostic worker protocol for in-browser Transformers.js **vision**
// pipelines. One generic worker (`vision.worker.ts`) serves every discriminative
// vision task — the pipeline `task` travels in the `load` message rather than
// being baked into a per-task worker file, exactly as `audio/pipelineTypes.ts`
// does for audio.
//
// The envelope is the shared one from `model/types.ts`; only the load/run
// payloads below are vision-specific. The run payload carries an `ImagePayload`
// rather than a `RawImage` because a class instance does not survive
// `postMessage` — see `image.ts` → **Worker transport**.

import type { Backend, DtypeSpec, LoadOpts } from "@/model/backend";
import type { ModelProgress, ModelRequest, ModelResponse } from "@/model/types";
import type { MeasuredBytes } from "@/model/size";

import type { ImagePayload } from "./image";

/** Transformers.js pipeline task strings this generic worker supports. */
export type VisionTask =
  | "image-classification"
  | "zero-shot-image-classification"
  | "object-detection"
  | "zero-shot-object-detection"
  | "image-segmentation"
  | "depth-estimation"
  | "image-feature-extraction"
  | "image-to-text"
  // Two tasks that arrived with Wave 3. Both are plain `pipeline()` calls
  // returning a `RawImage`, so neither earns an engine of its own
  // (docs/guides/adding-a-model.md §10) — they ride this worker unchanged.
  //
  // `background-removal` is a Transformers.js invention rather than a Hub task:
  // it subclasses the segmentation pipeline and puts the mask into the source's
  // alpha channel. `image-to-image` is the super-resolution slug.
  | "background-removal"
  | "image-to-image";

/** Load/warm-up progress. Alias of the shared `ModelProgress`. */
export type VisionProgress = ModelProgress;

/**
 * The shape every vision model catalogue entry shares. Structurally a
 * `PickableModel` (so `ModelPicker` takes it as-is) plus the task it runs and
 * the backends it is known to work on.
 */
export interface VisionModel {
  id: string;
  label: string;
  hint: string;
  /** Parameter count in millions — drives the size-before-load estimate. */
  params: number;
  /** Measured download bytes, where the params estimate would mislead. */
  bytes?: MeasuredBytes;
  task: VisionTask;
  /**
   * Backends this model is known to run on. Most vision models manage both;
   * list one when the other is a known failure rather than merely slower, so
   * the picker can gate instead of the load failing.
   */
  backends?: readonly Backend[];
  /**
   * The ONNX graph base names this entry actually downloads, relative to
   * `onnx/`. Defaults to `["model"]`.
   *
   * Not every repo publishes one `model.onnx`: CLIP as a feature extractor
   * loads `vision_model.onnx` alone (Transformers.js sets `model_file_name`
   * per class), and SAM ships `vision_encoder` plus
   * `prompt_encoder_mask_decoder`. `model-ids.spec.ts` uses this to check that
   * the dtype a backend asks for is actually published — a repo with only an
   * fp32 export resolves fine on the API and 404s at load time, and a check
   * hard-coded to `model.onnx` would look at the wrong file and pass.
   */
  graphs?: readonly string[];
  /**
   * Override the weight precision `loadOpts()` would pick, per backend. The
   * escape hatch for a model whose quantized export is broken — the vision
   * counterpart of `asrLoadOpts()`, and used for the same reason. Leave it
   * unset unless a real measurement says otherwise.
   */
  dtypes?: Partial<Record<Backend, DtypeSpec>>;
}

// --- Worker message protocol -------------------------------------------------

/** Main thread → worker. `args` are spread as positional args to the pipeline. */
export type VisionRequest = ModelRequest<
  {
    task: VisionTask;
    model: string;
    opts?: LoadOpts;
    /** Per-backend precision override; see `VisionModel.dtypes`. */
    dtypes?: Partial<Record<Backend, DtypeSpec>>;
  },
  { image: ImagePayload; args?: unknown[] }
>;

/** Worker → main thread. */
export type VisionResponse = ModelResponse<unknown>;
