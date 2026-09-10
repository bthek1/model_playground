// The SAM worker's message-handling core, factored out of `sam.worker.ts` so it
// can be unit-tested with fake towers — no download, no real Worker, and no
// `@huggingface/transformers` import. Same shape as `vision/engine.ts` and
// `vision/zeroshot/engine.ts`, and it owes the same three behaviours:
//
//  1. **One model live at a time.** Null the reference *first*, then dispose, so
//     a teardown that throws can never leave a stale model live. The encode
//     cache is cleared with it — an embedding from another checkpoint is not
//     decodable by this one.
//  2. **Warm up before `ready`.** One throwaway encode *and* decode, because
//     both graphs compile shaders and the first real click should pay for
//     neither. A failure there never fails the load.
//  3. **Never block the main thread** — which is what the worker is for.
//
// The one thing it adds over the other two engines: `run` is a discriminated
// union, so an encode and a decode are separate id-correlated requests. See
// `types.ts` for why that beats a `{ status: "encoding" }` progress event.

import { loadOpts, pickBackend, type DtypeSpec } from "@/model/backend";
import type { ModelProgress } from "@/model/types";

import type { ImagePayload } from "../image";
import { SamSession, type SamTowers } from "./sam";
import type { SamRequest, SamResponse } from "./types";

export interface SamTowerOpts {
  device: string;
  dtype: DtypeSpec;
  progress_callback?: (p: ModelProgress) => void;
}

export type SamTowerFactory = (
  model: string,
  opts: SamTowerOpts,
) => Promise<SamTowers>;

const WARMUP_SIDE = 64;

function warmupImage(): ImagePayload {
  const data = new Uint8ClampedArray(WARMUP_SIDE * WARMUP_SIDE * 3);
  data.fill(128);
  return { data, width: WARMUP_SIDE, height: WARMUP_SIDE, channels: 3 };
}

export function createSamHandler(
  post: (message: SamResponse, transfer?: Transferable[]) => void,
  factory: SamTowerFactory,
  { warmup = true }: { warmup?: boolean } = {},
) {
  let session: SamSession | null = null;
  let towers: SamTowers | null = null;

  return async function handle(msg: SamRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        const previous = towers;
        towers = null;
        session = null;
        await disposeQuietly(previous);

        const opts = msg.opts ?? loadOpts(await pickBackend());
        const dtype = msg.dtypes?.[opts.device] ?? opts.dtype;
        const next = await factory(msg.model, {
          device: opts.device,
          dtype,
          progress_callback: (progress) => post({ type: "progress", progress }),
        });
        towers = next;
        session = new SamSession(next);

        if (warmup) {
          post({ type: "progress", progress: { status: "warmup" } });
          try {
            // Both graphs: the encoder and the decoder compile separately, and
            // a first click that pays for the decoder's compile is the one the
            // user is most likely to interpret as "this is slow".
            await session.encode("warmup", warmupImage());
            await session.decode([
              { x: WARMUP_SIDE / 2, y: WARMUP_SIDE / 2, positive: true },
            ]);
          } catch {
            /* the first real click pays the compile cost instead */
          } finally {
            session.clear();
          }
        }
        post({ type: "ready", model: msg.model, backend: opts.device });
      } catch (error) {
        post({ type: "error", error: errMessage(error) });
      }
      return;
    }

    try {
      if (!session) throw new Error("No model loaded");

      if (msg.kind === "encode") {
        const outcome = await session.encode(msg.token, msg.image);
        post({ type: "result", id: msg.id, result: { kind: "encode", ...outcome } });
        return;
      }

      const { masks, ms } = await session.decode(msg.points);
      // The mask buffers are ours — the decoder allocated them for this reply
      // and nothing in the worker reads them again — so they are transferred
      // rather than copied. Three 640x480 masks is 921 KB per click, and a copy
      // on every click is exactly the cost this route claims not to pay.
      post(
        { type: "result", id: msg.id, result: { kind: "decode", masks, ms } },
        masks.map((m) => m.data.buffer as ArrayBuffer),
      );
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

async function disposeQuietly(t: SamTowers | null): Promise<void> {
  try {
    await t?.dispose?.();
  } catch {
    /* the reference is already dropped; GC and backend teardown reclaim it */
  }
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
