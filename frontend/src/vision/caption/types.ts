// Image-to-text: captioning, OCR and grounding. Catalogue, capability flags and
// worker protocol.
//
// **This task does not ride the generic vision worker, and the reason is a
// measurement rather than a preference.** #19 planned it as
// `pipeline("image-to-text")` with a task token in the run payload. That
// pipeline (`ImageToTextPipeline`) does exactly two things — run the processor
// for `pixel_values` and call `model.generate({ inputs })` — so there is nowhere
// to put a task token, and it resolves its model through
// `AutoModelForVision2Seq`, whose registry maps only `vision-encoder-decoder`,
// `idefics3` and `smolvlm`. Florence-2's model type is `florence2`, which lives
// in the *image-text-to-text* mapping instead: the pipeline cannot load it at
// all. Same shape of finding as MusicGen needing
// `MusicgenForConditionalGeneration` rather than the `text-to-audio` pipeline.
//
// So the engine takes one `Captioner` interface and the worker supplies two
// implementations behind it — the precedent being `tts.worker.ts`, which owns
// Kokoro, MMS and MusicGen behind one `TtsSynthesizer` because they all fit.
//
// `HuggingFaceTB/SmolVLM-256M-Instruct` is deliberately absent. It needs a chat
// template with image tokens to be prompted at all, and "ask a question about
// this picture" is the **Image-Text-to-Text** taxonomy row, not this one — a
// model that can only be driven by a question does not belong on a page whose
// four modes are fixed task tokens.

import type { Backend, DtypeSpec, LoadOpts } from "@/model/backend";
import type { ModelRequest, ModelResponse } from "@/model/types";
import type { MeasuredBytes } from "@/model/size";

import type { Detection } from "../draw";
import type { ImagePayload } from "../image";

/**
 * A mode the page can offer. The value is Florence-2's own task token, because
 * that is what the model is actually sent; the models that do not speak task
 * tokens simply declare fewer modes.
 */
export type CaptionMode =
  | "<CAPTION>"
  | "<DETAILED_CAPTION>"
  | "<OCR>"
  | "<OD>";

export const MODE_LABELS: Record<CaptionMode, string> = {
  "<CAPTION>": "Caption",
  "<DETAILED_CAPTION>": "Detailed caption",
  "<OCR>": "OCR",
  "<OD>": "Grounding",
};

export const MODE_HINTS: Record<CaptionMode, string> = {
  "<CAPTION>": "One sentence. The cheap, fast answer.",
  "<DETAILED_CAPTION>": "A paragraph. Several times the tokens, so several times the wait.",
  "<OCR>": "Every piece of printed text the model can find, as plain text.",
  "<OD>": "Objects, named and boxed — captioning and detection from one download.",
};

/** Modes whose answer is boxes rather than prose. */
export function isBoxMode(mode: CaptionMode): boolean {
  return mode === "<OD>";
}

export interface CaptionModelEntry {
  id: string;
  label: string;
  hint: string;
  params: number;
  bytes: MeasuredBytes;
  /**
   * The modes this checkpoint can actually answer. The UI must not offer OCR on
   * a model that only captions — a task token a model has never seen produces
   * a confident, fluent, unrelated sentence rather than an error.
   */
  modes: readonly CaptionMode[];
  /** How the worker drives it. See `caption.worker.ts`. */
  family: "florence2" | "vision-encoder-decoder";
  backends?: readonly Backend[];
  dtypes?: Partial<Record<Backend, DtypeSpec>>;
  /** ONNX graph base names, for the Hub-id spec. Neither repo has `model.onnx`. */
  graphs: readonly string[];
}

/**
 * The catalogue.
 *
 * Sizes are the sum of every graph the model loads, read off the Hub. Both are
 * large — this is the first vision route with a **generative decoder**, and an
 * encoder-decoder pair is simply bigger than a classifier.
 *
 *   Florence-2 base-ft  embed_tokens + vision_encoder + encoder + decoder_merged
 *                       544 MB fp16 · 275 MB q8
 *   ViT-GPT2            encoder + decoder_merged
 *                       482 MB fp16 · 246 MB q8
 */
export const CAPTION_MODELS: CaptionModelEntry[] = [
  {
    id: "onnx-community/Florence-2-base-ft",
    label: "Florence-2 base",
    hint: "One 0.23B download that captions, reads text and detects — the task token picks. WebGPU only.",
    params: 230,
    family: "florence2",
    modes: ["<CAPTION>", "<DETAILED_CAPTION>", "<OCR>", "<OD>"],
    // Not a preference. Four graphs and an autoregressive decoder on WASM is
    // tens of seconds per caption; the picker gates it rather than letting the
    // page look broken. See `useBackendProbe`.
    backends: ["webgpu"],
    graphs: [
      "embed_tokens",
      "vision_encoder",
      "encoder_model",
      "decoder_model_merged",
    ],
    bytes: { webgpu: 544_004_842, wasm: 274_966_163 },
  },
  {
    id: "Xenova/vit-gpt2-image-captioning",
    label: "ViT-GPT2",
    hint: "A plain caption, and the only one of the two that runs on a CPU. No OCR, no boxes.",
    params: 239,
    family: "vision-encoder-decoder",
    modes: ["<CAPTION>"],
    graphs: ["encoder_model", "decoder_model_merged"],
    bytes: { webgpu: 481_936_436, wasm: 246_053_209 },
  },
];

export const DEFAULT_CAPTION_MODEL = CAPTION_MODELS[0].id;

/**
 * Longest side fed to the model.
 *
 * Larger than the other vision routes' 640: Florence-2's processor resizes to
 * 768 anyway, and OCR is the one task where throwing away source resolution
 * before the processor sees it costs accuracy directly — small print stops
 * being legible.
 */
export const MAX_INFERENCE_SIDE = 1024;

/** Generation cap. A caption is a sentence; a detailed one is a paragraph. */
export const MAX_NEW_TOKENS: Record<CaptionMode, number> = {
  "<CAPTION>": 64,
  "<DETAILED_CAPTION>": 256,
  "<OCR>": 512,
  "<OD>": 512,
};

export interface CaptionLoad {
  model: string;
  family: CaptionModelEntry["family"];
  opts?: LoadOpts;
  dtypes?: Partial<Record<Backend, DtypeSpec>>;
}

export interface CaptionRun {
  image: ImagePayload;
  mode: CaptionMode;
  maxNewTokens: number;
}

export type CaptionResult =
  | { kind: "text"; mode: CaptionMode; text: string; ms: number }
  | { kind: "boxes"; mode: CaptionMode; detections: Detection[]; ms: number };

export type CaptionRequest = ModelRequest<CaptionLoad, CaptionRun>;
export type CaptionResponse = ModelResponse<CaptionResult>;
