// The generic pipeline worker's message-handling core, factored out of
// `pipeline.worker.ts` so it can be unit-tested with a fake pipeline factory (no
// model download, no real Worker). Mirrors `asrEngine.ts` but is task-agnostic:
// the `run` handler spreads `args` positionally, so it serves both
// `audio-classification` (`pipe(audio, { top_k })`) and
// `zero-shot-audio-classification` (`pipe(audio, labels, opts)`).

import { loadOpts, pickBackend } from "./backend";
import type {
  PipelineProgress,
  PipelineRequest,
  PipelineResponse,
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
  dtype: string;
  progress_callback?: (p: PipelineProgress) => void;
}

/** Builds a pipeline for a task + model id — the real one wraps Transformers.js. */
export type PipelineFactory = (
  task: string,
  model: string,
  opts: PipelineOpts,
) => Promise<CallablePipeline>;

/**
 * Create the async message handler for the generic pipeline worker. `post` sends
 * responses to the main thread; `factory` loads a pipeline. Holds one model live
 * at a time, disposing the previous one first.
 */
export function createPipelineHandler(
  post: (message: PipelineResponse, transfer?: Transferable[]) => void,
  factory: PipelineFactory,
) {
  let pipe: CallablePipeline | null = null;

  return async function handle(msg: PipelineRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        // One model live at a time — free the previous before loading the next.
        if (pipe?.dispose) await pipe.dispose();
        pipe = null;

        const opts = msg.opts ?? loadOpts(await pickBackend());
        pipe = await factory(msg.task, msg.model, {
          device: opts.device,
          dtype: opts.dtype,
          progress_callback: (progress) => post({ type: "progress", progress }),
        });
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

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
