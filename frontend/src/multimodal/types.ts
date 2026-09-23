// Image-Text-to-Text: a picture and a question in, an answer in prose out.
// Catalogue, capability flags and worker protocol — the Multimodal category's
// first route (roadmap §3.1).
//
// **There is no `image-text-to-text` pipeline to ride.** The roadmap's §1 says
// `@huggingface/transformers` 4.2.0 "carries" one; it does not. Its
// `SUPPORTED_TASKS` has 25 entries — `image-to-text` and
// `document-question-answering` among them — and `image-text-to-text` is not
// one. So this engine drives `AutoModelForImageTextToText` + `AutoProcessor`
// directly, which is what the roadmap's own §2 code sketch does anyway.
//
// That is the **third** time this repo has planned a page around a pipeline that
// could not carry it: MusicGen needed `MusicgenForConditionalGeneration` over
// the `text-to-audio` pipeline, and Florence-2 needed
// `Florence2ForConditionalGeneration` because `image-to-text` resolves through
// `AutoModelForVision2Seq`, whose registry does not map `florence2`. The rule
// earned three times over: **check `SUPPORTED_TASKS` before planning around a
// pipeline**, not after.
//
// Both of those routes have since been cut for size — `/text-to-audio` (599 MB
// minimum) and `/image-to-text` (482 MB minimum) had no lighter checkpoint to
// fall back to — but the rule they paid for outlives them, which is why it is
// recorded here rather than in either deleted file.

import type { Backend, DtypeSpec, LoadOpts } from "@/model/backend";
import type { ModelRequest, ModelResponse } from "@/model/types";
import type { MeasuredBytes } from "@/model/size";

import type { ImagePayload } from "@/vision/image";

export interface VlmModelEntry {
  id: string;
  label: string;
  hint: string;
  params: number;
  bytes: MeasuredBytes;
  /**
   * How the worker drives it. One family today; the field exists because the
   * catalogue's next entry (Qwen3-VL, roadmap §3.1) is `qwen3_vl` and loads
   * through the same auto-class but processes images differently. The deleted
   * captioning catalogue learned the same lesson the hard way: a second family
   * arrives sooner than expected.
   */
  family: "idefics3";
  backends?: readonly Backend[];
  dtypes?: Partial<Record<Backend, DtypeSpec>>;
  /** ONNX graph base names, for the Hub-id spec. Neither repo has `model.onnx`. */
  graphs: readonly string[];
}

/**
 * The catalogue: two rungs of one family.
 *
 * Both are `idefics3` (`Idefics3ForConditionalGeneration`), which 4.2.0
 * registers, and both publish a `chat_template`. Sizes are the sum of all three
 * graphs at `q4f16`, read off the Hub:
 *
 *   SmolVLM-256M   embed_tokens 56.8 + vision_encoder 55.0 + decoder 77.0  =  189 MB
 *   SmolVLM-500M                                                           =  358 MB
 *
 * **Measured, not estimated, and the gap is not rounding.** `estimateBytes`
 * assumes one precision across the model; at `q4f16` that is false in a way that
 * gets worse the smaller the model is. SmolVLM-256M's `embed_tokens_q4f16.onnx`
 * is 56.8 MB — the same size as its fp16 build, because the embedding table is
 * not 4-bit quantized at all. That is 30% of the download a params estimate
 * would silently halve.
 *
 * **Why the 2B is absent.** The roadmap §3.1 wants `Qwen3-VL-2B` here as the
 * heavy end of a head-to-head. Measured, it is **1373 MB** at q4f16 — past the
 * ~1 GB ceiling `docs/guides/adding-a-task-page.md` §0 sets, and past the budget
 * `model/size.test.ts` asserts over every shipped catalogue. It gets its own
 * plan and its own decision rather than arriving as a side effect of this one.
 * Two things to carry into that plan: Qwen3-VL keeps its weights in external
 * `.onnx_data` files (summing only the `.onnx` stubs measures it at 1.2 **MB**),
 * and the roadmap's alternative `Qwen2-VL-2B` is **2668 MB** at q4f16 rather
 * than the ~1.1 GB it quotes, so Qwen3-VL is the better heavy model by a wide
 * margin and not a toss-up.
 */
export const VLM_MODELS: VlmModelEntry[] = [
  {
    id: "HuggingFaceTB/SmolVLM-256M-Instruct",
    label: "SmolVLM 256M",
    hint: "The smallest VLM worth running. Seconds to load, and visibly the weaker of the two. WebGPU only.",
    params: 256,
    family: "idefics3",
    // Not a preference. An autoregressive decoder on WASM is seconds per token,
    // so the picker disables the row rather than offering a page that looks
    // broken.
    backends: ["webgpu"],
    graphs: ["embed_tokens", "vision_encoder", "decoder_model_merged"],
    bytes: { webgpu: 188_843_109, wasm: 263_889_326 },
  },
  {
    id: "HuggingFaceTB/SmolVLM-500M-Instruct",
    label: "SmolVLM 500M",
    hint: "Twice the download and noticeably better answers. The comparison worth making on this page.",
    params: 500,
    family: "idefics3",
    backends: ["webgpu"],
    graphs: ["embed_tokens", "vision_encoder", "decoder_model_merged"],
    bytes: { webgpu: 357_638_049, wasm: 485_088_765 },
  },
];

export const DEFAULT_VLM_MODEL = VLM_MODELS[0].id;

/**
 * Longest side fed to the processor. **512, and this one is not a round number
 * picked for latency** — it is SmolVLM's own tile size.
 *
 * `preprocessor_config.json` declares `size.longest_edge: 2048`,
 * `max_image_size.longest_edge: 512` and `do_image_splitting: true`: the
 * processor resizes up to 2048 and then **cuts the picture into 512px tiles**,
 * each of which is encoded separately and costs its own image tokens, plus a
 * global view on top. A 2048px input is up to a 4x4 grid — seventeen encodes
 * for one question.
 *
 * Handing it an image already at 512 produces **one** tile. So this is the
 * difference between ~64 image tokens and over a thousand before the user's
 * question is even appended, which is the roadmap §2 budget made concrete.
 *
 * This is a cap on the **source**, not preprocessing: `AutoProcessor` still
 * reads the model's own config and does the real resize and normalisation.
 * Never resize *for* the model.
 *
 * A document-QA page would have to reverse this — a document needs pixels
 * before its text is legible to the encoder — so any future page sets its own
 * number rather than inheriting this one. Noted here because inheriting a
 * sibling page's hyperparameter is a mistake this repo has already made once,
 * on `/link-prediction`.
 */
export const MAX_INFERENCE_SIDE = 512;

/** Generation cap. An answer is a sentence or two, not an essay. */
export const DEFAULT_MAX_NEW_TOKENS = 128;

/** The bounds the page's slider offers. */
export const MIN_NEW_TOKENS = 16;
export const MAX_NEW_TOKENS = 512;

export interface VlmLoad {
  model: string;
  family: VlmModelEntry["family"];
  opts?: LoadOpts;
  dtypes?: Partial<Record<Backend, DtypeSpec>>;
}

export interface VlmRun {
  image: ImagePayload;
  prompt: string;
  maxNewTokens: number;
}

/**
 * Progress **inside** one run, posted against the request id.
 *
 * A VLM encodes the image before a single token exists, and on a 500M model that
 * pause is seconds long. An unlabelled pause is indistinguishable from a hang —
 * roadmap §2 — so the two halves are named and the generating half carries the
 * text so far.
 *
 * This is not a `ModelStatus`. Machine A stays `ready` throughout and `running`
 * stays an inflight count; see the `partial` variant in `model/types.ts`.
 */
export type VlmPartial =
  | { stage: "encoding" }
  | { stage: "generating"; text: string };

export interface VlmResult {
  text: string;
  /** Total wall time for the run, in ms. */
  ms: number;
  /** How long the image encode took, in ms — the part the user waits through blind. */
  encodeMs: number;
  /** Tokens generated, so the page can report a real tokens/sec. */
  tokens: number;
}

export type VlmRequest = ModelRequest<VlmLoad, VlmRun>;
export type VlmResponse = ModelResponse<VlmResult, VlmPartial>;
