// The generic pipeline worker's message-handling core, factored out of
// `pipeline.worker.ts` so it can be unit-tested with a fake pipeline factory (no
// model download, no real Worker). Mirrors `asrEngine.ts` but is task-agnostic:
// the `run` handler spreads `args` positionally, so it serves both
// `audio-classification` (`pipe(audio, { top_k })`) and
// `zero-shot-audio-classification` (`pipe(audio, labels, opts)`).

import { loadOpts, pickBackend, type DtypeSpec } from "@/model/backend";
import type {
  PipelineProgress,
  PipelineRequest,
  PipelineResponse,
  PipelineTask,
} from "./pipelineTypes";

/** A loaded pipeline: callable with positional args, with an optional dispose. */
export type CallablePipeline = ((
  input: Float32Array,
  ...args: unknown[]
) => Promise<unknown>) & {
  dispose?: () => Promise<void>;
};

export interface PipelineOpts {
  device: string;
  dtype: DtypeSpec;
  progress_callback?: (p: PipelineProgress) => void;
}

/** Builds a pipeline for a task + model id — the real one wraps Transformers.js. */
export type PipelineFactory = (
  task: string,
  model: string,
  opts: PipelineOpts,
) => Promise<CallablePipeline>;

/**
 * A short buffer of silence, run through the model once on load so the first
 * real request doesn't pay to compile the WebGPU shaders / JIT the WASM module.
 */
const WARMUP_AUDIO_SAMPLES = 4000; // 0.25 s @ 16 kHz

/** Minimal positional args that make a warm-up call valid for each task. */
function warmupArgs(task: PipelineTask): unknown[] {
  return task === "zero-shot-audio-classification"
    ? [["speech"]] // zero-shot needs at least one candidate label
    : [{ top_k: 1 }];
}

/**
 * Create the async message handler for the generic pipeline worker. `post` sends
 * responses to the main thread; `factory` loads a pipeline. Holds one model live
 * at a time, disposing the previous one first.
 */
export function createPipelineHandler(
  post: (message: PipelineResponse, transfer?: Transferable[]) => void,
  factory: PipelineFactory,
  { warmup = true }: { warmup?: boolean } = {},
) {
  let pipe: CallablePipeline | null = null;

  return async function handle(msg: PipelineRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        // One model live at a time — free the previous before loading the next.
        // Null it first so a failed dispose can never leave a stale model live.
        const previous = pipe;
        pipe = null;
        await disposeQuietly(previous);

        const opts = msg.opts ?? loadOpts(await pickBackend());
        pipe = await factory(msg.task, msg.model, {
          device: opts.device,
          dtype: opts.dtype,
          progress_callback: (progress) => post({ type: "progress", progress }),
        });
        if (warmup) {
          post({ type: "progress", progress: { status: "warmup" } });
          // Never fail a load over the warm-up — the model is still usable.
          try {
            await pipe(
              new Float32Array(WARMUP_AUDIO_SAMPLES),
              ...warmupArgs(msg.task),
            );
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
async function disposeQuietly(pipe: CallablePipeline | null): Promise<void> {
  try {
    await pipe?.dispose?.();
  } catch {
    /* the reference is already dropped; GC + backend teardown reclaim it */
  }
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
