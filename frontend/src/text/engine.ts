// The generic text worker's message-handling core, factored out of
// `pipeline.worker.ts` so it can be unit-tested with a fake pipeline factory —
// no model download, no real Worker, and no `@huggingface/transformers` import.
// `vision/engine.ts` and `audio/pipelineEngine.ts` are the counterparts, and
// this deliberately mirrors them: same three obligations, same disposal order,
// same warm-up rule.
//
// It is the simplest of the three, and for one reason: there is nothing to
// convert. A vision engine speaks `ImagePayload` because a `RawImage` does not
// survive `postMessage`; an audio engine hands over a detached `Float32Array`.
// A string crosses the wire as itself.

import { loadOpts, pickBackend, type DtypeSpec } from "@/model/backend";

import type {
  FillMaskResult,
  TextInput,
  TextProgress,
  TextRequest,
  TextResponse,
  TextTask,
} from "./types";

/** A loaded pipeline: callable with positional args, with an optional dispose. */
export type CallableTextPipeline = ((
  input: TextInput,
  ...args: unknown[]
) => Promise<unknown>) & {
  dispose?: () => Promise<void>;
  /**
   * The pipeline's tokenizer. Every Transformers.js text pipeline carries one;
   * only the masked-LM path reads it, and only for `mask_token` — which is a
   * **tokenizer** fact, not a model one, and the reason `/fill-mask` can ship
   * three tokenizer families behind one catalogue.
   */
  tokenizer?: { mask_token?: string | null };
};

export interface TextPipelineOpts {
  device: string;
  dtype: DtypeSpec;
  progress_callback?: (p: TextProgress) => void;
}

/** Builds a pipeline for a task + model id — the real one wraps Transformers.js. */
export type TextPipelineFactory = (
  task: string,
  model: string,
  opts: TextPipelineOpts,
) => Promise<CallableTextPipeline>;

/**
 * The throwaway input the model is run on once at load, so the first real
 * request does not pay to compile the WebGPU shaders. Short on purpose: an
 * encoder's cost is quadratic in sequence length and the point is to touch
 * every kernel, not to be representative.
 */
const WARMUP_TEXT = "Hello world.";

/** Minimal positional args that make a warm-up call valid for each task. */
function warmupArgs(task: TextTask): unknown[] {
  switch (task) {
    case "text-classification":
      return [{ top_k: 1 }];
    case "token-classification":
      return [];
    // Zero-shot needs candidate labels or the call throws, and **two** rather
    // than one: the pipeline takes a different normalisation branch for a
    // single label (`softmaxEach = multi_label || labels.length === 1`), so a
    // one-label warm-up would compile a path the first real run does not use.
    // Two labels is two forward passes on a three-word premise — still cheap.
    case "zero-shot-classification":
      return [["yes", "no"]];
    case "fill-mask":
      return [{ top_k: 1 }];
  }
}

/**
 * The string the warm-up runs on.
 *
 * `fill-mask` needs its own, because `FillMaskPipeline` raises "Mask token
 * (…) not found in text." on an input with no mask — *after* the forward pass,
 * so the shaders do compile and the swallowed error looks harmless. Relying on
 * that is relying on the order of two statements in a dependency: give the
 * warm-up a real mask instead, taken from the tokenizer that was just loaded.
 */
function warmupInput(task: TextTask, pipe: CallableTextPipeline): TextInput {
  const mask = task === "fill-mask" ? maskToken(pipe) : null;
  return mask ? `Hello ${mask}.` : WARMUP_TEXT;
}

/** The mask literal the loaded tokenizer uses, or null if it has none. */
function maskToken(pipe: CallableTextPipeline): string | null {
  const token = pipe.tokenizer?.mask_token;
  return typeof token === "string" && token.length > 0 ? token : null;
}

/**
 * Swap the caller's mask literal for the loaded tokenizer's own.
 *
 * **This is what makes "never hard-code the mask token" true rather than
 * merely intended.** The page inserts the literal its catalogue entry declares,
 * so it can show the token before a model exists; the tokenizer is what decides
 * which characters are actually sent. If the two agree — the normal case — this
 * is a no-op. If they have drifted, the run is still correct and the page says
 * so, instead of the user meeting `Mask token (<mask>) not found in text.`
 */
function retargetInput(
  input: TextInput,
  from: string | undefined,
  to: string | null,
): TextInput {
  if (!from || !to || from === to) return input;
  const swap = (s: string) => s.split(from).join(to);
  if (typeof input === "string") return swap(input);
  if (Array.isArray(input)) return input.map(swap);
  return input;
}

/**
 * Options this task cannot run correctly without, merged into the caller's
 * own options object.
 *
 * It lives here rather than in the hook because forgetting one is not an error
 * — it is a **rendering bug**. Without `aggregation_strategy: "simple"` the
 * token-classification pipeline returns one result per *subword token*, so
 * "Wellington" comes back as `Well` / `##ing` / `##ton`, each with its own
 * offsets, and the page paints three highlights across one word. That looks
 * like a broken overlay, not a missing option, and it is one forgetful call
 * site away at every future caller. Pinning it in the engine means there is
 * only one call site.
 */
function pinnedArgs(task: TextTask): Record<string, unknown> | null {
  switch (task) {
    case "token-classification":
      return { aggregation_strategy: "simple" };
    default:
      return null;
  }
}

/**
 * Merge the task's mandatory options into the first options argument.
 *
 * This assumes the task takes its options **first**, which is true of every
 * task that currently pins anything. It is not true of zero-shot
 * classification, whose first positional argument is the candidate-label
 * *array* — pinning an option there would shift the labels to position 1 and
 * the pipeline would classify against `undefined`. That task pins nothing (its
 * one interesting option, `hypothesis_template`, is the user's to set), so the
 * case never arises; a future task that needs both must widen this rather than
 * relying on the array check below.
 */
function withPinned(task: TextTask, args: unknown[]): unknown[] {
  const pinned = pinnedArgs(task);
  if (!pinned) return args;
  const [first, ...rest] = args;
  const options =
    first != null && typeof first === "object" && !Array.isArray(first)
      ? (first as Record<string, unknown>)
      : null;
  return options
    ? [{ ...options, ...pinned }, ...rest]
    : [pinned, ...(first === undefined ? [] : [first, ...rest])];
}

/**
 * Create the async message handler for the generic text worker. `post` sends
 * responses to the main thread; `factory` loads a pipeline.
 *
 * The three obligations every engine in this app owes:
 *
 *  1. **One model live at a time.** Null the reference *first*, then dispose, so
 *     a teardown that throws can never leave a stale model live.
 *  2. **Warm up before `ready`.** One throwaway inference, announced as
 *     `{ status: "warmup" }`, and a failure there never fails the load.
 *  3. **Never block the main thread** — which is what the worker is for.
 */
export function createTextHandler(
  post: (message: TextResponse, transfer?: Transferable[]) => void,
  factory: TextPipelineFactory,
  { warmup = true }: { warmup?: boolean } = {},
) {
  let pipe: CallableTextPipeline | null = null;
  let task: TextTask | null = null;

  return async function handle(msg: TextRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        const previous = pipe;
        pipe = null;
        task = null;
        await disposeQuietly(previous);

        const opts = msg.opts ?? loadOpts(await pickBackend());
        // A catalogue entry may pin the precision for the backend we landed on:
        // some exports are only correct — or only present — at one of them.
        const dtype = msg.dtypes?.[opts.device] ?? opts.dtype;
        pipe = await factory(msg.task, msg.model, {
          device: opts.device,
          dtype,
          progress_callback: (progress) => post({ type: "progress", progress }),
        });
        if (warmup) {
          post({ type: "progress", progress: { status: "warmup" } });
          try {
            await pipe(warmupInput(msg.task, pipe), ...warmupArgs(msg.task));
          } catch {
            /* ignore — the first real run just pays the compile cost instead */
          }
        }
        task = msg.task;
        post({ type: "ready", model: msg.model, backend: opts.device });
      } catch (error) {
        post({ type: "error", error: errMessage(error) });
      }
      return;
    }

    // msg.type === "run"
    try {
      if (!pipe || !task) throw new Error("No model loaded");
      // `fill-mask` is the one task whose *input* depends on the loaded
      // tokenizer, and whose result carries a fact about it back. Everything
      // else passes straight through.
      const mask = task === "fill-mask" ? maskToken(pipe) : null;
      const input =
        task === "fill-mask"
          ? retargetInput(msg.input, msg.mask, mask)
          : msg.input;
      const output = await pipe(input, ...withPinned(task, msg.args ?? []));
      const result =
        task === "fill-mask"
          ? ({ mask, fills: output } as FillMaskResult)
          : output;
      post({ type: "result", id: msg.id, result });
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

/** Free a pipeline, tolerating a backend that fails or has no `dispose`. */
async function disposeQuietly(pipe: CallableTextPipeline | null): Promise<void> {
  try {
    await pipe?.dispose?.();
  } catch {
    /* the reference is already dropped; GC + backend teardown reclaim it */
  }
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
