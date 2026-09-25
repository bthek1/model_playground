// The streaming worker's protocol, and the decoding-parameter set that is this
// page's whole subject.
//
// `/text-generation` is the category's only streaming page, and it is the payoff
// for a decision already taken: #30 put the `partial` variant in the **shared**
// `ModelResponse` envelope rather than in a private VLM protocol, with the
// stated justification that NLP text generation would need exactly the same
// thing. It did, and it needed nothing the envelope does not have — `partial`
// carries the text so far against the request id, Machine A stays `ready`, and
// `running` stays an inflight count.
//
// It gets its own worker rather than riding `pipeline.worker.ts` for the ASR
// reason: the task that owns a loop owns its worker.

import type { Backend, DtypeSpec } from "@/model/backend";
import type { ModelProgress, ModelRequest, ModelResponse } from "@/model/types";
import type { MeasuredBytes } from "@/model/size";

/**
 * The decoding strategy, in full.
 *
 * **None of these can re-derive from a finished generation** — the model has to
 * run again — so every one of them is INPUT that spends nothing until GENERATE,
 * and the page says the next press is a real inference. Same shape as
 * `/video-text-to-text`'s reverse toggle and `/zero-shot-classification`'s
 * `multi_label`.
 */
export interface Decoding {
  /** Greedy when false: always the highest-probability token, reproducible. */
  doSample: boolean;
  /** Flattens (>1) or sharpens (<1) the distribution. Ignored when greedy. */
  temperature: number;
  /** Nucleus sampling: keep the smallest set whose mass reaches this. */
  topP: number;
  /** Keep only this many candidates. 0 means "no cap". */
  topK: number;
  /** >1 penalises tokens already produced — the anti-loop dial. */
  repetitionPenalty: number;
  maxNewTokens: number;
}

/** The repo's defaults: greedy, so the first thing a visitor sees is reproducible. */
export const DEFAULT_DECODING: Decoding = {
  doSample: false,
  temperature: 0.7,
  topP: 0.9,
  topK: 50,
  repetitionPenalty: 1.0,
  maxNewTokens: 96,
};

/**
 * In-run progress: the text so far.
 *
 * Deliberately not a stage union like the VLM's. A text-only decoder has no
 * encode phase to announce — the first token arrives in about a second — so
 * there is nothing a stage would tell the user that the arriving text does not.
 */
export interface TextGenPartial {
  text: string;
  /** Tokens streamed so far, for the tokens/sec readout. */
  tokens: number;
}

/** What one finished generation produced. */
export interface TextGenResult {
  /** The continuation only — never prompt + continuation. */
  text: string;
  tokens: number;
  /** Wall-clock of the generation, so the page can show tokens/sec. */
  ms: number;
}

/** A text-generation checkpoint. */
export interface TextGenModel {
  id: string;
  label: string;
  hint: string;
  params: number;
  bytes: MeasuredBytes;
  /** Where the weights came from / what it was tuned for. */
  domain: string;
  backends?: readonly Backend[];
  graphs?: readonly string[];
  dtypes?: Partial<Record<Backend, DtypeSpec>>;
  /**
   * An ONNX graph base name other than `model`.
   *
   * `MODEL_SESSION_CONFIG[DecoderOnly]` is `{ model: options.model_file_name ??
   * 'model' }`, so a repo whose only quantized build lives under a legacy
   * `decoder_model_merged_*` name is reachable **only** by naming it. Measured:
   * the option does pass through `pipeline()` and the file is found. See the
   * GPT-2 note in the catalogue for why that turned out not to be enough.
   */
  modelFile?: string;
  /**
   * True when the entry loads `f16` weights and therefore needs the adapter's
   * `shader-f16` feature — not merely "a GPU".
   *
   * #30's finding, and the worst failure available: without the feature the
   * adapter loads the weights, reports `ready`, and fails on the **first
   * operator** of every run, after the user has paid for the download.
   */
  requireShaderF16?: boolean;
  /** Is this an instruction-tuned chat model, or a bare LM that continues text? */
  instruct: boolean;
}

/** Load/warm-up progress. Alias of the shared shape. */
export type TextGenProgress = ModelProgress;

/** Main thread → worker. */
export type TextGenRequest = ModelRequest<
  {
    model: string;
    dtypes?: Partial<Record<Backend, DtypeSpec>>;
    modelFile?: string;
    requireShaderF16?: boolean;
  },
  {
    prompt: string;
    decoding: Decoding;
    /** Wrap the prompt in the model's chat template. Instruct models only. */
    chat: boolean;
  }
>;

/** Worker → main thread, with the streaming arm inhabited. */
export type TextGenResponse = ModelResponse<TextGenResult, TextGenPartial>;
