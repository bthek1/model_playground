// The TTS worker's message-handling core, factored out of `tts.worker.ts` so it
// can be unit-tested with a fake synthesizer factory (no model download, no real
// Worker). Same shape as `pipelineEngine.ts`, but text in → audio out, and the
// result's sample buffer is transferred back to the main thread (zero-copy).

import { loadOpts, pickBackend, type DtypeSpec } from "@/model/backend";
import type { PipelineProgress } from "./pipelineTypes";
import type { TtsAudio, TtsRequest, TtsResponse, TtsRunOpts } from "./tts";

/** A loaded synthesizer: text + opts → audio, with an optional dispose. */
export type TtsSynthesizer = ((
  text: string,
  opts?: TtsRunOpts,
) => Promise<TtsAudio>) & {
  dispose?: () => Promise<void>;
};

export interface TtsFactoryOpts {
  device: string;
  dtype: DtypeSpec;
  progress_callback?: (p: PipelineProgress) => void;
}

/** Builds a synthesizer for a model id — the real one wraps kokoro-js / pipeline. */
export type TtsFactory = (
  model: string,
  opts: TtsFactoryOpts,
) => Promise<TtsSynthesizer>;

/**
 * Synthesised once on load so the first real request doesn't pay to compile the
 * WebGPU shaders / JIT the WASM module. Short on purpose — synthesis time scales
 * with the text, and this only needs to touch every kernel once.
 */
const WARMUP_TEXT = "Hi.";

/**
 * Token budget for the warm-up pass. Only MusicGen reads it (the speech models
 * ignore `maxNewTokens`), but for MusicGen it is the difference between a ~2 s
 * warm-up and a ~37 s one: without it the run falls back to the 256-token
 * default and generates five seconds of music nobody hears.
 */
const WARMUP_TOKENS = 16;

/**
 * Create the async message handler for the TTS worker. `post` sends responses to
 * the main thread (with optional transfer list); `factory` loads a synthesizer.
 * Holds one model live at a time, disposing the previous one first.
 */
export function createTtsHandler(
  post: (message: TtsResponse, transfer?: Transferable[]) => void,
  factory: TtsFactory,
  { warmup = true }: { warmup?: boolean } = {},
) {
  let synth: TtsSynthesizer | null = null;

  return async function handle(msg: TtsRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        // One model live at a time — free the previous before loading the next.
        // Null it first so a failed dispose can never leave a stale model live.
        const previous = synth;
        synth = null;
        await disposeQuietly(previous);

        const opts = msg.opts ?? loadOpts(await pickBackend());
        synth = await factory(msg.model, {
          device: opts.device,
          dtype: opts.dtype,
          progress_callback: (progress) => post({ type: "progress", progress }),
        });
        if (warmup) {
          post({ type: "progress", progress: { status: "warmup" } });
          // Never fail a load over the warm-up — the model is still usable.
          try {
            await synth(WARMUP_TEXT, { maxNewTokens: WARMUP_TOKENS });
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
      if (!synth) throw new Error("No TTS model loaded");
      const result = await synth(msg.text, msg.opts);
      // Transfer the sample buffer — the worker no longer needs it.
      post({ type: "result", id: msg.id, result }, [result.audio.buffer]);
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

/** Free a synthesizer, tolerating one that fails or has no `dispose`. */
async function disposeQuietly(synth: TtsSynthesizer | null): Promise<void> {
  try {
    await synth?.dispose?.();
  } catch {
    /* the reference is already dropped; GC + backend teardown reclaim it */
  }
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
