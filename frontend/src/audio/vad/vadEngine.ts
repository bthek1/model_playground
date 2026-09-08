// The VAD worker's message-handling core, factored out of `vad.worker.ts` so it
// can be unit-tested with a fake session — no download, no ONNX Runtime, no real
// Worker. Mirrors `enhanceEngine` and owes the same three duties:
//
//   1. one model live at a time — null the reference *before* disposing;
//   2. warm up on load — one throwaway inference so the first real clip doesn't
//      pay the ORT graph-init cost, and never fail the load if it throws;
//   3. never block the main thread — this file only runs inside a Worker.

import type { Backend } from "../backend";
import { FRAME_SAMPLES, SAMPLE_RATE } from "./types";
import type { VadProgress, VadRequest, VadResponse } from "./types";
import type { VadSession } from "./session";

/** Opens a session for a model id. The real one wraps `loadVad`. */
export type SessionFactory = (
  model: string,
  onProgress: (p: VadProgress) => void,
  backend?: Backend,
) => Promise<VadSession>;

/** A tenth of a second of silence — a few frames, enough to init the graph. */
const WARMUP_SAMPLES = SAMPLE_RATE / 10;

export function createVadHandler(
  post: (message: VadResponse, transfer?: Transferable[]) => void,
  factory: SessionFactory,
  { warmup = true }: { warmup?: boolean } = {},
) {
  let session: VadSession | null = null;

  return async function handle(msg: VadRequest): Promise<void> {
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
            await session.probabilities(new Float32Array(WARMUP_SAMPLES));
          } catch {
            /* the first real clip just pays the init cost instead */
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
      const samples = msg.audio.length;
      const probabilities = await session.probabilities(msg.audio);
      // Small next to the audio itself (one float per 32 ms), but transferring
      // costs nothing and keeps the protocol uniform with the other routes.
      post(
        {
          type: "result",
          id: msg.id,
          result: {
            probabilities,
            frameSamples: FRAME_SAMPLES,
            sampleRate: SAMPLE_RATE,
            samples,
          },
        },
        [probabilities.buffer],
      );
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

async function disposeQuietly(session: VadSession | null): Promise<void> {
  try {
    await session?.dispose();
  } catch {
    /* the reference is already dropped; GC and the backend reclaim the rest */
  }
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
