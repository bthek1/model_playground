// The ASR worker's message-handling core, factored out of `asr.worker.ts` so it
// can be unit-tested with a fake pipeline factory (no model download, no real
// Worker). The worker file is a thin wrapper that wires this to `self`.

import { asrLoadOpts, pickBackend, type DtypeSpec } from "./backend";
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
  dtype: DtypeSpec;
  progress_callback?: (p: AsrProgress) => void;
}

/** Builds a pipeline for a model id — the real one wraps Transformers.js. */
export type AsrPipelineFactory = (
  model: string,
  opts: AsrPipelineOpts,
) => Promise<AsrPipeline>;

/**
 * A short buffer of silence, run through the model once on load. The first real
 * inference otherwise pays to compile the WebGPU shaders / JIT the WASM module;
 * doing it here moves that cost into the load bar the user is already watching.
 * Whisper pads to 30 s regardless, so the length only matters for Moonshine.
 */
const WARMUP_AUDIO_SAMPLES = 4000; // 0.25 s @ 16 kHz

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
  { warmup = true }: { warmup?: boolean } = {},
) {
  let pipe: AsrPipeline | null = null;

  return async function handle(msg: AsrRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        // One model live at a time — free the previous before loading the next.
        // Null it first so a failed dispose can never leave a stale model live.
        const previous = pipe;
        pipe = null;
        await disposeQuietly(previous);

        const opts = msg.opts ?? asrLoadOpts(await pickBackend());
        pipe = await factory(msg.model, {
          device: opts.device,
          dtype: opts.dtype,
          progress_callback: (progress) => post({ type: "progress", progress }),
        });
        if (warmup) {
          post({ type: "progress", progress: { status: "warmup" } });
          // Never fail a load over the warm-up — the model is still usable.
          try {
            await pipe(new Float32Array(WARMUP_AUDIO_SAMPLES), {
              ...DEFAULT_RUN_ARGS,
              return_timestamps: false,
            });
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
      if (!pipe) throw new Error("No ASR model loaded");
      const out = await pipe(msg.audio, { ...DEFAULT_RUN_ARGS, ...msg.args });
      const result = Array.isArray(out) ? out[0] : out;
      post({ type: "result", id: msg.id, result });
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

/** Free a pipeline, tolerating a backend that fails or has no `dispose`. */
async function disposeQuietly(pipe: AsrPipeline | null): Promise<void> {
  try {
    await pipe?.dispose?.();
  } catch {
    /* the reference is already dropped; GC + backend teardown reclaim it */
  }
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
