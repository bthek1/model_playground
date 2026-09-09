// The split-tower zero-shot engine, and the one place in the app that holds an
// inference cache.
//
// **Why this exists.** The `zero-shot-image-classification` pipeline re-encodes
// the labels on every call. The label embeddings do not depend on the image, so
// on a live camera feed that is the text tower run thirty times a second to
// produce the same numbers — roughly 40% of the work, thrown away and redone
// (docs/roadmaps/vision.md §5, "encode once, decode many"). Driving
// `CLIPTextModelWithProjection` and `CLIPVisionModelWithProjection` separately
// lets the text side be computed once and kept.
//
// **What it costs.** The full `model.onnx` ends with normalise → matmul →
// scale → softmax. Running the towers alone means owning those steps, which is
// why they live in `scoring.ts` as pure arithmetic pinned by a parity spec
// against the pipeline. This is the same trade `audio/enhance/` makes, and it
// carries the same risk: wrong scoring arithmetic produces confident, plausible,
// wrongly-scaled numbers with the ranking intact.
//
// Factored out of the worker so it is testable with fake towers — no download,
// no real Worker, no `@huggingface/transformers` import.

import { loadOpts, pickBackend, type DtypeSpec } from "@/model/backend";
import type { ModelProgress } from "@/model/types";

import type { ImagePayload } from "../image";
import { TEXT_CACHE_LIMIT } from "../zeroShot";
import { normalizedRows, scoreImage, type ScoringSpec } from "./scoring";
import type { ZeroShotRequest, ZeroShotResponse } from "./types";

/** A tower's output: a flat `[rows x dim]` matrix. */
export interface Embeddings {
  data: ArrayLike<number>;
  rows: number;
  dim: number;
}

/** The two towers plus their pre-processing, as the worker supplies them. */
export interface ZeroShotTowers {
  encodeText: (prompts: string[]) => Promise<Embeddings>;
  encodeImage: (image: ImagePayload) => Promise<Embeddings>;
  dispose?: () => Promise<void>;
}

export interface TowerOpts {
  device: string;
  dtype: DtypeSpec;
  family: "clip" | "siglip";
  progress_callback?: (p: ModelProgress) => void;
}

export type ZeroShotTowerFactory = (
  model: string,
  opts: TowerOpts,
) => Promise<ZeroShotTowers>;

/** Cache key for a prompt set. Order matters — it is the answer's order. */
export function promptsKey(prompts: readonly string[]): string {
  return JSON.stringify(prompts);
}

export function createZeroShotHandler(
  post: (message: ZeroShotResponse, transfer?: Transferable[]) => void,
  factory: ZeroShotTowerFactory,
  { warmup = true }: { warmup?: boolean } = {},
) {
  let towers: ZeroShotTowers | null = null;
  let scoring: ScoringSpec | null = null;
  // Insertion-ordered, so the oldest key is the first one `keys()` yields —
  // which is all the eviction policy this needs.
  const textCache = new Map<string, Float32Array[]>();

  return async function handle(msg: ZeroShotRequest): Promise<void> {
    if (msg.type === "load") {
      try {
        const previous = towers;
        towers = null;
        textCache.clear(); // embeddings from the old checkpoint mean nothing here
        await disposeQuietly(previous);

        const opts = msg.opts ?? loadOpts(await pickBackend());
        const dtype = msg.dtypes?.[opts.device] ?? opts.dtype;
        scoring = msg.scoring;
        towers = await factory(msg.model, {
          device: opts.device,
          dtype,
          family: msg.family,
          progress_callback: (progress) => post({ type: "progress", progress }),
        });

        if (warmup) {
          post({ type: "progress", progress: { status: "warmup" } });
          try {
            // Both towers, because both compile shaders and the first real run
            // should pay for neither.
            await towers.encodeText(["a photo"]);
            await towers.encodeImage(warmupImage());
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
      if (!towers || !scoring) throw new Error("No model loaded");
      if (msg.prompts.length === 0) throw new Error("No labels to score");

      const key = promptsKey(msg.prompts);
      const cached = textCache.get(key);
      let textMs = 0;
      let textEmbeds: Float32Array[];

      if (cached) {
        // Re-insert so the most recently used key is the last to be evicted.
        textCache.delete(key);
        textCache.set(key, cached);
        textEmbeds = cached;
      } else {
        const started = now();
        const out = await towers.encodeText(msg.prompts);
        textMs = now() - started;
        textEmbeds = normalizedRows(out.data, out.rows, out.dim);
        textCache.set(key, textEmbeds);
        while (textCache.size > TEXT_CACHE_LIMIT) {
          const oldest = textCache.keys().next().value;
          if (oldest === undefined) break;
          textCache.delete(oldest);
        }
      }

      const imageStarted = now();
      const image = await towers.encodeImage(msg.image);
      const imageMs = now() - imageStarted;
      const [imageEmbed] = normalizedRows(image.data, 1, image.dim);

      post({
        type: "result",
        id: msg.id,
        result: {
          scores: scoreImage(imageEmbed, textEmbeds, scoring),
          textCached: cached !== undefined,
          textMs,
          imageMs,
        },
      });
    } catch (error) {
      post({ type: "error", id: msg.id, error: errMessage(error) });
    }
  };
}

const WARMUP_SIDE = 64;

function warmupImage(): ImagePayload {
  const data = new Uint8ClampedArray(WARMUP_SIDE * WARMUP_SIDE * 3);
  data.fill(128);
  return { data, width: WARMUP_SIDE, height: WARMUP_SIDE, channels: 3 };
}

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

async function disposeQuietly(t: ZeroShotTowers | null): Promise<void> {
  try {
    await t?.dispose?.();
  } catch {
    /* the reference is already dropped; GC and backend teardown reclaim it */
  }
}

function errMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
