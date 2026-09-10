// The image-to-text worker's message-handling core, factored out of
// `caption.worker.ts` so it can be unit-tested with a fake captioner — no
// download, no real Worker, and no `@huggingface/transformers` import.
//
// It owes the same three behaviours as every engine in this app
// (docs/roadmaps/audio.md §2): one model live at a time with the reference
// nulled *before* the dispose, one warm-up inference before `ready` that never
// fails the load, and never blocking the main thread.
//
// One thing is specific to this route. **The warm-up costs a real generation.**
// Every other engine warms up with a single forward pass; this one has an
// autoregressive decoder, so a warm-up that generated a full caption would add
// seconds to a load that is already the longest in the category. It is capped
// at a handful of tokens instead — enough to compile both graphs' shaders,
// which is the entire point, and not enough to be worth waiting for.

import { loadOpts, pickBackend, type DtypeSpec } from "@/model/backend";
import type { ModelProgress } from "@/model/types";

import type { Detection } from "../draw";
import type { ImagePayload } from "../image";
import type {
  CaptionMode,
  CaptionModelEntry,
  CaptionRequest,
  CaptionResponse,
} from "./types";

/** What one generation produced: prose, or boxes, depending on the mode. */
export type CaptionOutput =
  | { kind: "text"; text: string }
  | { kind: "boxes"; detections: Detection[] };

/** A loaded model, however it is driven underneath. */
export interface Captioner {
  generate: (
    image: ImagePayload,
    mode: CaptionMode,
    maxNewTokens: number,
  ) => Promise<CaptionOutput>;
  dispose?: () => Promise<void>;
}

export interface CaptionerOpts {
  device: string;
  dtype: DtypeSpec;
  family: CaptionModelEntry["family"];
  progress_callback?: (p: ModelProgress) => void;
}

export type CaptionerFactory = (
  model: string,
  opts: CaptionerOpts,
) => Promise<Captioner>;

const WARMUP_SIDE = 64;
/** Enough to compile the decoder's shaders, not enough to wait for. */
const WARMUP_TOKENS = 2;

function warmupImage(): ImagePayload {
  const data = new Uint8ClampedArray(WARMUP_SIDE * WARMUP_SIDE * 3);
  data.fill(128);
  return { data, width: WARMUP_SIDE, height: WARMUP_SIDE, channels: 3 };
}

export function createCaptionHandler(
  post: (message: CaptionResponse, transfer?: Transferable[]) => void,
  factory: CaptionerFactory,
  { warmup = true }: { warmup?: boolean } = {},
) {
  let captioner: Captioner | null = null;

  return async function handle(msg: CaptionRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        const previous = captioner;
        captioner = null;
        await disposeQuietly(previous);

        const opts = msg.opts ?? loadOpts(await pickBackend());
        const dtype = msg.dtypes?.[opts.device] ?? opts.dtype;
        captioner = await factory(msg.model, {
          device: opts.device,
          dtype,
          family: msg.family,
          progress_callback: (progress) => post({ type: "progress", progress }),
        });

        if (warmup) {
          post({ type: "progress", progress: { status: "warmup" } });
          try {
            await captioner.generate(warmupImage(), "<CAPTION>", WARMUP_TOKENS);
          } catch {
            /* the first real run pays the compile cost instead */
          }
        }
        post({ type: "ready", model: msg.model, backend: opts.device });
      } catch (error) {
        post({ type: "error", error: errMessage(error) });
      }
      return;
    }

    try {
      if (!captioner) throw new Error("No model loaded");
      const started = now();
      const out = await captioner.generate(msg.image, msg.mode, msg.maxNewTokens);
      const ms = now() - started;
      post({
        type: "result",
        id: msg.id,
        result:
          out.kind === "text"
            ? { kind: "text", mode: msg.mode, text: out.text, ms }
            : { kind: "boxes", mode: msg.mode, detections: out.detections, ms },
      });
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

async function disposeQuietly(c: Captioner | null): Promise<void> {
  try {
    await c?.dispose?.();
  } catch {
    /* the reference is already dropped; GC and backend teardown reclaim it */
  }
}

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
