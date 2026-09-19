// The VLM worker's message-handling core, factored out of `vlm.worker.ts` so it
// can be unit-tested with a fake model — no download, no real Worker, and no
// `@huggingface/transformers` import.
//
// It owes the same three behaviours as every engine in this app
// (docs/roadmaps/audio.md §2): one model live at a time with the reference
// nulled *before* the dispose, one warm-up inference before `ready` that never
// fails the load, and never blocking the main thread. "One model live at a time"
// is not a style rule here — these are the largest downloads in the app, and a
// leaked session ends the tab.
//
// Two things are specific to this route.
//
// **The warm-up costs a real generation**, as it does for `vision/caption/`: an
// autoregressive decoder warmed up to completion would add seconds to the
// longest load in the app. Capped at a couple of tokens — enough to compile the
// shaders of all three graphs, which is the entire point.
//
// **A run reports its own progress.** The image encode happens before a single
// token exists, so the engine posts `{ stage: "encoding" }`, then
// `{ stage: "generating", text }` as tokens arrive. These are `partial`
// messages correlated to the request id: Machine A stays `ready` and `running`
// stays an inflight count throughout (see `model/types.ts`).

import { pickBackend, vlmLoadOpts, type DtypeSpec } from "@/model/backend";
import type { ModelProgress } from "@/model/types";

import type { ImagePayload } from "@/vision/image";
import type {
  VlmModelEntry,
  VlmPartial,
  VlmRequest,
  VlmResponse,
} from "./types";

/** What one generation produced, plus the timings the page reports. */
export interface VlmOutput {
  text: string;
  encodeMs: number;
  tokens: number;
}

/** A loaded vision-language model, however it is driven underneath. */
export interface Vlm {
  generate: (
    image: ImagePayload,
    prompt: string,
    maxNewTokens: number,
    onPartial: (partial: VlmPartial) => void,
  ) => Promise<VlmOutput>;
  dispose?: () => Promise<void>;
}

export interface VlmOpts {
  device: string;
  dtype: DtypeSpec;
  family: VlmModelEntry["family"];
  progress_callback?: (p: ModelProgress) => void;
}

export type VlmFactory = (model: string, opts: VlmOpts) => Promise<Vlm>;

const WARMUP_SIDE = 64;
/** Enough to compile the decoder's shaders, not enough to wait for. */
const WARMUP_TOKENS = 2;
const WARMUP_PROMPT = "Hi";

function warmupImage(): ImagePayload {
  const data = new Uint8ClampedArray(WARMUP_SIDE * WARMUP_SIDE * 3);
  data.fill(128);
  return { data, width: WARMUP_SIDE, height: WARMUP_SIDE, channels: 3 };
}

export function createVlmHandler(
  post: (message: VlmResponse, transfer?: Transferable[]) => void,
  factory: VlmFactory,
  { warmup = true }: { warmup?: boolean } = {},
) {
  let vlm: Vlm | null = null;

  return async function handle(msg: VlmRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        // Null the reference *first*: a dispose that throws must not leave a
        // stale model live, which on this family is hundreds of MB of GPU.
        const previous = vlm;
        vlm = null;
        await disposeQuietly(previous);

        const opts = msg.opts ?? vlmLoadOpts(await pickBackend());
        const dtype = msg.dtypes?.[opts.device] ?? opts.dtype;
        vlm = await factory(msg.model, {
          device: opts.device,
          dtype,
          family: msg.family,
          progress_callback: (progress) => post({ type: "progress", progress }),
        });

        if (warmup) {
          post({ type: "progress", progress: { status: "warmup" } });
          try {
            await vlm.generate(
              warmupImage(),
              WARMUP_PROMPT,
              WARMUP_TOKENS,
              () => {
                /* a warm-up has no audience */
              },
            );
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
      if (!vlm) throw new Error("No model loaded");
      const started = now();
      const out = await vlm.generate(
        msg.image,
        msg.prompt,
        msg.maxNewTokens,
        (partial) => post({ type: "partial", id: msg.id, partial }),
      );
      post({
        type: "result",
        id: msg.id,
        result: {
          text: out.text,
          ms: now() - started,
          encodeMs: out.encodeMs,
          tokens: out.tokens,
        },
      });
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

async function disposeQuietly(v: Vlm | null): Promise<void> {
  try {
    await v?.dispose?.();
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
