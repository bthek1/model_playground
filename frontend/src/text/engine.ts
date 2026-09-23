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
  }
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

  return async function handle(msg: TextRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        const previous = pipe;
        pipe = null;
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
            await pipe(WARMUP_TEXT, ...warmupArgs(msg.task));
          } catch {
            /* ignore — the first real run just pays the compile cost instead */
          }
        }
        post({ type: "ready", model: msg.model, backend: opts.device });
      } catch (error) {
        post({ type: "error", error: errMessage(error) });
      }
      return;
    }

    // msg.type === "run"
    try {
      if (!pipe) throw new Error("No model loaded");
      const result = await pipe(msg.input, ...(msg.args ?? []));
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
