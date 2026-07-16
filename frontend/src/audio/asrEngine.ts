// The ASR worker's message-handling core, factored out of `asr.worker.ts` so it
// can be unit-tested with a fake pipeline factory (no model download, no real
// Worker). The worker file is a thin wrapper that wires this to `self`.

import { loadOpts, pickBackend } from "./backend";
import type {
  AsrProgress,
  AsrRequest,
  AsrResponse,
  AsrResult,
  AsrRunArgs,
} from "./types";

/** A loaded ASR pipeline: callable, with an optional `dispose`. */
export type AsrPipeline = ((
  audio: Float32Array,
  args?: AsrRunArgs,
) => Promise<AsrResult | AsrResult[]>) & {
  dispose?: () => Promise<void>;
};

export interface AsrPipelineOpts {
  device: string;
  dtype: string;
  progress_callback?: (p: AsrProgress) => void;
}

/** Builds a pipeline for a model id — the real one wraps Transformers.js. */
export type AsrPipelineFactory = (
  model: string,
  opts: AsrPipelineOpts,
) => Promise<AsrPipeline>;

/** Default run args mirror the notebook: long-form chunking + timestamps. */
const DEFAULT_RUN_ARGS: AsrRunArgs = {
  return_timestamps: true,
  chunk_length_s: 30,
  task: "transcribe",
};

/**
 * Create the async message handler for the ASR worker. `post` sends responses
 * back to the main thread (with optional transfer list); `factory` loads a
 * pipeline. Holds one model live at a time, disposing the previous one first.
 */
export function createAsrHandler(
  post: (message: AsrResponse, transfer?: Transferable[]) => void,
  factory: AsrPipelineFactory,
) {
  let pipe: AsrPipeline | null = null;

  return async function handle(msg: AsrRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        // One model live at a time — free the previous before loading the next.
        if (pipe?.dispose) await pipe.dispose();
        pipe = null;

        const opts = msg.opts ?? loadOpts(await pickBackend());
        pipe = await factory(msg.model, {
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
      if (!pipe) throw new Error("No ASR model loaded");
      const out = await pipe(msg.audio, { ...DEFAULT_RUN_ARGS, ...msg.args });
      const result = Array.isArray(out) ? out[0] : out;
      post({ type: "result", id: msg.id, result });
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
