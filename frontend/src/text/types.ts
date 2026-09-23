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
export type TextTask = "text-classification";

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
  task: TextTask;
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
  { input: TextInput; args?: unknown[] }
>;

/** Worker → main thread. */
export type TextResponse = ModelResponse<unknown>;
