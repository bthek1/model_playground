// The speech-enhancement worker's message-handling core, factored out of
// `enhance.worker.ts` so it can be unit-tested with a fake session — no model
// download, no ONNX Runtime, no real Worker. Mirrors `asrEngine` /
// `pipelineEngine`, and owes the same three duties every engine owes:
//
//   1. one model live at a time — null the reference *before* disposing, so a
//      failed teardown can't leave a stale session running;
//   2. warm up on load — one throwaway inference so the first real request does
//      not pay for shader compilation, and never fail the load if it throws;
//   3. never block the main thread — this file only ever runs inside a Worker.

import type { DeepFilterSession } from "./session";
import { enhanceAudio, SAMPLE_RATE } from "./deepFilterNet";
import type { EnhanceRequest, EnhanceResponse, EnhanceProgress } from "./types";
import type { Backend } from "@/model/backend";

/** Opens a session for a repo. The real one wraps `loadDeepFilterNet`. */
export type SessionFactory = (
  repo: string,
  onProgress: (p: EnhanceProgress) => void,
  backend?: Backend,
) => Promise<DeepFilterSession>;

/** Half a second of silence — long enough to exercise every stage once. */
const WARMUP_SAMPLES = SAMPLE_RATE / 2;

export function createEnhanceHandler(
  post: (message: EnhanceResponse, transfer?: Transferable[]) => void,
  factory: SessionFactory,
  { warmup = true }: { warmup?: boolean } = {},
) {
  let session: DeepFilterSession | null = null;

  return async function handle(msg: EnhanceRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        const previous = session;
        session = null;
        await disposeQuietly(previous);

        session = await factory(
          msg.model,
          (progress) => post({ type: "progress", progress }),
          msg.backend,
        );

        if (warmup) {
          post({ type: "progress", progress: { status: "warmup" } });
          try {
            await enhanceAudio(
              new Float32Array(WARMUP_SAMPLES),
              session.aux,
              session.infer,
            );
          } catch {
            /* the first real run just pays the compile cost instead */
          }
        }
        post({ type: "ready", model: msg.model, backend: session.backend });
      } catch (error) {
        post({ type: "error", error: errMessage(error) });
      }
      return;
    }

    // msg.type === "run"
    try {
      if (!session) throw new Error("No model loaded");
      const audio = await enhanceAudio(msg.audio, session.aux, session.infer);
      // The result is the only large payload; hand the buffer over rather than
      // copying a minute of 48 kHz float samples across the boundary.
      post(
        { type: "result", id: msg.id, result: { audio, sampleRate: SAMPLE_RATE } },
        [audio.buffer],
      );
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

async function disposeQuietly(session: DeepFilterSession | null): Promise<void> {
  try {
    await session?.dispose();
  } catch {
    /* the reference is already dropped; GC and the backend reclaim the rest */
  }
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
