// The TTS worker's message-handling core, factored out of `tts.worker.ts` so it
// can be unit-tested with a fake synthesizer factory (no model download, no real
// Worker). Same shape as `pipelineEngine.ts`, but text in → audio out, and the
// result's sample buffer is transferred back to the main thread (zero-copy).

import { loadOpts, pickBackend } from "./backend";
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
  dtype: string;
  progress_callback?: (p: PipelineProgress) => void;
}

/** Builds a synthesizer for a model id — the real one wraps kokoro-js / pipeline. */
export type TtsFactory = (
  model: string,
  opts: TtsFactoryOpts,
) => Promise<TtsSynthesizer>;

/**
 * Create the async message handler for the TTS worker. `post` sends responses to
 * the main thread (with optional transfer list); `factory` loads a synthesizer.
 * Holds one model live at a time, disposing the previous one first.
 */
export function createTtsHandler(
  post: (message: TtsResponse, transfer?: Transferable[]) => void,
  factory: TtsFactory,
) {
  let synth: TtsSynthesizer | null = null;

  return async function handle(msg: TtsRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        // One model live at a time — free the previous before loading the next.
        if (synth?.dispose) await synth.dispose();
        synth = null;

        const opts = msg.opts ?? loadOpts(await pickBackend());
        synth = await factory(msg.model, {
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
      if (!synth) throw new Error("No TTS model loaded");
      const result = await synth(msg.text, msg.opts);
      // Transfer the sample buffer — the worker no longer needs it.
      post({ type: "result", id: msg.id, result }, [result.audio.buffer]);
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
