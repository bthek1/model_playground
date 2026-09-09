// The generic vision worker's message-handling core, factored out of
// `vision.worker.ts` so it can be unit-tested with a fake pipeline factory — no
// model download, no real Worker, and no `@huggingface/transformers` import.
// The audio counterpart is `audio/pipelineEngine.ts`, and this deliberately
// mirrors it: same three obligations, same disposal order, same warm-up rule.
//
// It speaks `ImagePayload`, never `RawImage`. Converting a payload back into the
// class the pipeline wants is the *factory's* job (`vision.worker.ts`), which is
// what keeps this file free of the runtime and therefore cheap to test.

import { loadOpts, pickBackend, type DtypeSpec } from "@/model/backend";

import type { ImagePayload } from "./image";
import type {
  VisionProgress,
  VisionRequest,
  VisionResponse,
  VisionTask,
} from "./types";

/** A loaded pipeline: callable with positional args, with an optional dispose. */
export type CallableVisionPipeline = ((
  image: ImagePayload,
  ...args: unknown[]
) => Promise<unknown>) & {
  dispose?: () => Promise<void>;
};

export interface VisionPipelineOpts {
  device: string;
  dtype: DtypeSpec;
  progress_callback?: (p: VisionProgress) => void;
}

/** Builds a pipeline for a task + model id — the real one wraps Transformers.js. */
export type VisionPipelineFactory = (
  task: string,
  model: string,
  opts: VisionPipelineOpts,
) => Promise<CallableVisionPipeline>;

/**
 * A tiny grey image, run through the model once on load so the first real
 * request doesn't pay to compile the WebGPU shaders. On a detector that is two
 * to four seconds the user should not be charged for. 64x64 is large enough that
 * every processor's resize path is exercised and small enough to be free.
 */
const WARMUP_SIDE = 64;

function warmupImage(): ImagePayload {
  const data = new Uint8ClampedArray(WARMUP_SIDE * WARMUP_SIDE * 3);
  data.fill(128);
  return { data, width: WARMUP_SIDE, height: WARMUP_SIDE, channels: 3 };
}

/** Minimal positional args that make a warm-up call valid for each task. */
function warmupArgs(task: VisionTask): unknown[] {
  switch (task) {
    // Both zero-shot tasks need at least one candidate label to score against.
    case "zero-shot-image-classification":
    case "zero-shot-object-detection":
      return [["a photo"]];
    case "image-classification":
      return [{ top_k: 1 }];
    default:
      return [];
  }
}

/**
 * Create the async message handler for the generic vision worker. `post` sends
 * responses to the main thread; `factory` loads a pipeline.
 *
 * The three obligations every engine in this app owes (docs/roadmaps/audio.md
 * §2, and they are not audio-specific):
 *
 *  1. **One model live at a time.** Null the reference *first*, then dispose, so
 *     a teardown that throws can never leave a stale model live.
 *  2. **Warm up before `ready`.** One throwaway inference, announced as
 *     `{ status: "warmup" }`, and a failure there never fails the load.
 *  3. **Never block the main thread** — which is what the worker is for.
 */
export function createVisionHandler(
  post: (message: VisionResponse, transfer?: Transferable[]) => void,
  factory: VisionPipelineFactory,
  { warmup = true }: { warmup?: boolean } = {},
) {
  let pipe: CallableVisionPipeline | null = null;

  return async function handle(msg: VisionRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        const previous = pipe;
        pipe = null;
        await disposeQuietly(previous);

        const opts = msg.opts ?? loadOpts(await pickBackend());
        // A catalogue entry may pin the precision for the backend we landed on:
        // some exports are only correct at one of them (see `VisionModel.dtypes`).
        const dtype = msg.dtypes?.[opts.device] ?? opts.dtype;
        pipe = await factory(msg.task, msg.model, {
          device: opts.device,
          dtype,
          progress_callback: (progress) => post({ type: "progress", progress }),
        });
        if (warmup) {
          post({ type: "progress", progress: { status: "warmup" } });
          try {
            await pipe(warmupImage(), ...warmupArgs(msg.task));
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
      const result = await pipe(msg.image, ...(msg.args ?? []));
      post({ type: "result", id: msg.id, result });
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

/** Free a pipeline, tolerating a backend that fails or has no `dispose`. */
async function disposeQuietly(
  pipe: CallableVisionPipeline | null,
): Promise<void> {
  try {
    await pipe?.dispose?.();
  } catch {
    /* the reference is already dropped; GC + backend teardown reclaim it */
  }
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
