// Task-agnostic worker protocol for in-browser Transformers.js **text**
// pipelines — the NLP counterpart of `vision/types.ts`, and deliberately the
// same shape. One generic worker (`pipeline.worker.ts`) serves every
// discriminative text task; the pipeline `task` travels in the `load` message
// rather than being baked into a per-task worker file.
//
// The envelope is the shared one from `model/types.ts`; only the load/run
// payloads below are text-specific. Unlike audio and vision there is **no
// decode step and no transport problem**: the input is already a string, so
// nothing here corresponds to `audio/io.ts` or `vision/image.ts`, and a run
// payload survives `postMessage` as-is. That absence is the whole reason
// `/text-classification` is the category's first page.
//
// Text **generation** is the one task that will not ride this worker: it
// streams, so it gets its own (the `useAsr` precedent — the task that owns a
// loop owns its worker).

import type { Backend, DtypeSpec, LoadOpts } from "@/model/backend";
import type { ModelProgress, ModelRequest, ModelResponse } from "@/model/types";
import type { MeasuredBytes } from "@/model/size";

/**
 * Transformers.js pipeline task strings this generic worker supports.
 *
 * Every NLP page adds its own arm as it ships, rather than the union being
 * written out in advance: an arm with no caller is an untested branch, and the
 * engine's per-task warm-up and result handling have to be written against a
 * real pipeline's output anyway.
 */
export type TextTask =
  | "text-classification"
  | "token-classification"
  | "zero-shot-classification"
  | "fill-mask"
  | "feature-extraction"
  | "translation"
  | "summarization";

/**
 * Every task in the NLP category, whether or not it rides the generic worker.
 *
 * `question-answering` is the first that does not, for the repo's standing
 * reason: it is not a plain `pipeline()` call. `QuestionAnsweringPipeline`
 * returns `{ answer, score }` and discards the token indices it chose, so a
 * page that marks the answer **in the passage** cannot use it — `text/qa/`
 * drives `AutoTokenizer` + `AutoModelForQuestionAnswering` directly and keeps
 * them. The same criterion put `vision/zeroshot/`, `vision/sam/` and
 * `audio/vad/` in modules of their own.
 *
 * It is deliberately *not* folded into `TextTask` above: that union is the
 * contract of `pipeline.worker.ts`, and widening it would add an arm to the
 * engine's per-task switches that no caller could ever reach.
 */
export type TextCategoryTask = TextTask | "question-answering";

/** Load/warm-up progress. Alias of the shared `ModelProgress`. */
export type TextProgress = ModelProgress;

/**
 * The shape every text model catalogue entry shares. Structurally a
 * `PickableModel` (so `ModelPicker` takes it as-is) plus the task it runs and
 * the backends it is known to work on.
 */
export interface TextModel {
  id: string;
  label: string;
  hint: string;
  /** Parameter count in millions — drives the size-before-load estimate. */
  params: number;
  /**
   * Measured download bytes. **Every entry in this category carries one**, which
   * is a stricter rule than vision's "where the estimate would mislead", and it
   * is a finding rather than a preference: #5's size tables were all q8 figures,
   * while `loadOpts()` asks for fp16 on WebGPU — the backend any machine with an
   * adapter gets. Roughly double, all the way down the category, and enough to
   * move a page across the feasibility bar.
   */
  bytes: MeasuredBytes;
  task: TextCategoryTask;
  /**
   * Backends this model is known to run on. Omitted means both. List one when
   * the other is a known failure rather than merely slower, so `ModelPicker`
   * can gate the row instead of the load failing after the download.
   */
  backends?: readonly Backend[];
  /**
   * The ONNX graph base names this entry downloads, relative to `onnx/`.
   * Defaults to `["model"]`. A seq2seq entry publishes `encoder_model` and
   * `decoder_model_merged` instead — and `model-ids.spec.ts` checks *these*
   * files for the dtype each backend asks for, which a check hard-coded to
   * `model.onnx` would get wrong.
   */
  graphs?: readonly string[];
  /**
   * Override the weight precision `loadOpts()` would pick, per backend.
   *
   * There is deliberately **no `textLoadOpts()`**. The `asrLoadOpts`/`vlmLoadOpts`
   * precedent is for a decision that holds across a whole family, and the
   * measurements do not support one here: q8-on-WebGPU is right for a seq2seq
   * summarizer and wrong by default for a 22 MB embedder. Per the repo's
   * standing rule, each pin's comment says whether it is a **measurement or a
   * precaution** — only one of those is evidence.
   */
  dtypes?: Partial<Record<Backend, DtypeSpec>>;
}

/**
 * The first positional argument of a pipeline call.
 *
 * A plain string for most tasks; a `{ text, text_pair }` object for a
 * cross-encoder, which scores a *pair* rather than a concatenation; an array
 * where a pipeline batches.
 */
export type TextInput = string | string[] | { text: string; text_pair: string };

// --- Worker message protocol -------------------------------------------------

/** Main thread → worker. `args` are spread as positional args after `input`. */
export type TextRequest = ModelRequest<
  {
    task: TextTask;
    model: string;
    opts?: LoadOpts;
    /** Per-backend precision override; see `TextModel.dtypes`. */
    dtypes?: Partial<Record<Backend, DtypeSpec>>;
  },
  {
    input: TextInput;
    args?: unknown[];
    /**
     * The mask literal the caller put in `input`, for `fill-mask` only.
     *
     * It rides here rather than inside `args` because it is **not a pipeline
     * option** — the pipeline never sees it. The engine rewrites it to the
     * *loaded tokenizer's* own `mask_token` before the call, so a catalogue
     * entry that has drifted from its repo cannot turn into a failed run: the
     * page shows the declared token, the tokenizer decides what is sent. See
     * `text/mask.ts`.
     */
    mask?: string;
  }
>;

/** One candidate filling, as `FillMaskPipeline` returns it. */
export interface RawFilling {
  /** The predicted token. Byte-level BPE leaves its leading space on: ` Paris`. */
  token_str: string;
  score: number;
  token?: number;
  /**
   * The decoded sentence with the token substituted.
   *
   * Present, and deliberately unused by the page: it is `tokenizer.decode(…)`
   * output, so an uncased model hands back "the capital of france is paris."
   * The route splices the *user's own string* instead (`text/mask.ts`).
   */
  sequence?: string;
}

/**
 * What the engine returns for a `fill-mask` run.
 *
 * The resolved mask travels with the result because it is a fact about the
 * **loaded tokenizer**, and the page has no other way to learn it — the
 * catalogue's declaration is what it shows before a model exists, and the two
 * disagreeing is something the page states rather than hides.
 */
export interface FillMaskResult {
  mask: string | null;
  /** One ranked list per prompt: nested when the input was an array. */
  fills: RawFilling[] | RawFilling[][];
}

/** Worker → main thread. */
export type TextResponse = ModelResponse<unknown>;
