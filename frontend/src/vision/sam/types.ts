// Worker protocol and catalogue for Mask Generation (SAM).
//
// This is the third task in the app that owns an engine rather than riding a
// generic pipeline worker, and the criterion is the documented one: it is not a
// plain `pipeline()` call. SAM ships as **two graphs** — a vision encoder and a
// prompt-encoder/mask-decoder — and the split is the entire point. The encoder
// runs once per image; every click after that decodes a mask from the cached
// embedding in milliseconds. A pipeline call would re-encode the image on every
// click, which is the whole cost.
//
// **`run` carries a discriminated payload rather than one shape.** Encoding and
// decoding are genuinely different requests with different costs, and making the
// encode its own id-correlated request buys two things a `{ status: "encoding" }`
// progress event (the shape #17 sketched) cannot:
//
//   1. **A failure path.** A progress event has none. An encode that fails —
//      a 40-megapixel image, a lost GPU device — must land in Machine B, where
//      the model stays loaded and the next image still works. Routed through a
//      progress event it would either be swallowed or would push Machine A to
//      `error` and force a full reload of weights that are perfectly fine.
//   2. **Ordering.** The id correlates the answer with the request, so a decode
//      cannot resolve against an encode that was superseded.
//
// The envelope is still the shared `ModelRequest`/`ModelResponse` from
// `model/types.ts`; only the payloads below are specific to this task.

import type { Backend, DtypeSpec, LoadOpts } from "@/model/backend";
import type { ModelRequest, ModelResponse } from "@/model/types";
import type { MeasuredBytes } from "@/model/size";

import type { ImagePayload } from "../image";

export interface SamModelEntry {
  id: string;
  label: string;
  hint: string;
  /** Parameter count in millions — the shared `ModelPicker` wants it. */
  params: number;
  /** Measured download bytes: both graphs, summed. */
  bytes: MeasuredBytes;
  /** Backends this checkpoint is known to run on. */
  backends?: readonly Backend[];
  /** Per-backend precision override; unset means `loadOpts()` decides. */
  dtypes?: Partial<Record<Backend, DtypeSpec>>;
  /**
   * The two ONNX graph base names, so `model-ids.spec.ts` checks the files that
   * are actually downloaded. There is no `model.onnx` in either repo.
   */
  graphs: readonly string[];
}

/**
 * The catalogue.
 *
 * Sizes are the sum of **both** graphs' blobs (`.onnx` plus any `.onnx_data`
 * external-weights sidecar), read off the Hub. Quoting only the decoder — the
 * small half — would understate SlimSAM by 60% and SAM 2.1 by 85%.
 *
 *   SlimSAM-77   vision_encoder 12.2 MB + decoder  8.6 MB fp16 · 8.9 + 4.9 MB q8
 *   SAM 2.1 tiny vision_encoder 67.0 MB + decoder 10.5 MB fp16 · 52.6 + 8.7 MB q8
 *
 * `syscv-community/sam-hq-vit-base` and `facebook/sam3` have no export and stay
 * server-side.
 */
export const SAM_MODELS: SamModelEntry[] = [
  {
    id: "Xenova/slimsam-77-uniform",
    label: "SlimSAM-77",
    hint: "SAM pruned to a fifth of its size and still interactive. 21 MB, and the default.",
    params: 10,
    bytes: { webgpu: 20_720_775, wasm: 13_785_975 },
    graphs: ["vision_encoder", "prompt_encoder_mask_decoder"],
  },
  {
    id: "onnx-community/sam2.1-hiera-tiny-ONNX",
    label: "SAM 2.1 tiny",
    hint: "The 2.1 line — a hierarchical encoder, better on small objects, four times the download.",
    params: 39,
    bytes: { webgpu: 78_004_244, wasm: 61_966_687 },
    graphs: ["vision_encoder", "prompt_encoder_mask_decoder"],
  },
];

export const DEFAULT_SAM_MODEL = SAM_MODELS[0].id;

/**
 * Longest side the image is capped to before encoding.
 *
 * Not only a speed throttle here. The decoder's masks come back **at the source
 * resolution**, three of them per click, one byte per pixel — so a 12-megapixel
 * photo would post 36 MB across the worker boundary on every click. At 640 the
 * three masks are under a megabyte and a click still feels instant.
 */
export const MAX_INFERENCE_SIDE = 640;

/** A click, in source-image pixels. */
export interface SamPoint {
  x: number;
  y: number;
  /** True for "this is the object", false for "this is not" (alt-click). */
  positive: boolean;
}

/** One candidate mask, flattened for `postMessage`. */
export interface SamMask {
  /** One byte per pixel, 0 or 1, row-major at the encoded image's size. */
  data: Uint8Array;
  width: number;
  height: number;
  /** The decoder's own IoU estimate for this candidate. */
  score: number;
}

export interface SamLoad {
  model: string;
  opts?: LoadOpts;
  dtypes?: Partial<Record<Backend, DtypeSpec>>;
}

export type SamRun =
  | {
      kind: "encode";
      /** A caller-supplied identity for this image; re-encoding is skipped
       *  when it matches what is already encoded. */
      token: string;
      image: ImagePayload;
    }
  | { kind: "decode"; points: SamPoint[] };

export type SamResult =
  | {
      kind: "encode";
      /** True when the embedding was reused rather than recomputed. */
      cached: boolean;
      ms: number;
      width: number;
      height: number;
    }
  | {
      kind: "decode";
      /** SAM's three candidates, best-scoring first. */
      masks: SamMask[];
      ms: number;
    };

export type SamRequest = ModelRequest<SamLoad, SamRun>;
export type SamResponse = ModelResponse<SamResult>;
