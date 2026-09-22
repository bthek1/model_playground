// The document-QA worker's message-handling core, factored out of
// `docvqa.worker.ts` so it can be unit-tested with a fake answerer — no
// download, no real Worker, and no `@huggingface/transformers` import.
//
// It owes the same three behaviours as every engine in this app
// (docs/roadmaps/audio.md §2): one model live at a time with the reference
// nulled *before* the dispose, one warm-up inference before `ready` that never
// fails the load, and never blocking the main thread.
//
// One thing is specific to this route, and it is shared with the other
// generative pages: **the warm-up costs a real generation**, so it is capped at
// a couple of tokens. Enough to compile both graphs' shaders, which is the whole
// point, and not enough to add to a load that is already ~219–411 MB.

import { loadOpts, pickBackend, type DtypeSpec } from "@/model/backend";
import type { ModelProgress } from "@/model/types";

import type { ImagePayload } from "@/vision/image";
import type { DocVqaRequest, DocVqaResponse } from "./types";

/** A loaded document-QA model, however it is driven underneath. */
export interface DocAnswerer {
  answer: (
    image: ImagePayload,
    question: string,
    maxNewTokens: number,
  ) => Promise<string | null>;
  dispose?: () => Promise<void>;
}

export interface DocAnswererOpts {
  device: string;
  dtype: DtypeSpec;
  progress_callback?: (p: ModelProgress) => void;
}

export type DocAnswererFactory = (
  model: string,
  opts: DocAnswererOpts,
) => Promise<DocAnswerer>;

const WARMUP_SIDE = 64;
/** Enough to compile the decoder's shaders, not enough to wait for. */
const WARMUP_TOKENS = 2;
const WARMUP_QUESTION = "What is this?";

function warmupImage(): ImagePayload {
  const data = new Uint8ClampedArray(WARMUP_SIDE * WARMUP_SIDE * 3);
  data.fill(255);
  return { data, width: WARMUP_SIDE, height: WARMUP_SIDE, channels: 3 };
}

export function createDocVqaHandler(
  post: (message: DocVqaResponse, transfer?: Transferable[]) => void,
  factory: DocAnswererFactory,
  { warmup = true }: { warmup?: boolean } = {},
) {
  let answerer: DocAnswerer | null = null;

  return async function handle(msg: DocVqaRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        // Null the reference *first*: a dispose that throws must not leave a
        // stale model live.
        const previous = answerer;
        answerer = null;
        await disposeQuietly(previous);

        const opts = msg.opts ?? loadOpts(await pickBackend());
        const dtype = msg.dtypes?.[opts.device] ?? opts.dtype;
        answerer = await factory(msg.model, {
          device: opts.device,
          dtype,
          progress_callback: (progress) => post({ type: "progress", progress }),
        });

        if (warmup) {
          post({ type: "progress", progress: { status: "warmup" } });
          try {
            await answerer.answer(warmupImage(), WARMUP_QUESTION, WARMUP_TOKENS);
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
      if (!answerer) throw new Error("No model loaded");
      const started = now();
      const answer = await answerer.answer(
        msg.image,
        msg.question,
        msg.maxNewTokens,
      );
      // `answer` may legitimately be null — the pipeline returns that when its
      // `<s_answer>` regex misses. It travels as null rather than being turned
      // into an error or an empty string, because "found nothing" and "found an
      // empty span" are different things to say on screen.
      post({
        type: "result",
        id: msg.id,
        result: { answer, question: msg.question, ms: now() - started },
      });
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

async function disposeQuietly(a: DocAnswerer | null): Promise<void> {
  try {
    await a?.dispose?.();
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
