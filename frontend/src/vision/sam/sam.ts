// The encode-once/decode-many cache, and the whole reason this task has its own
// module.
//
// SAM's architecture was designed for exactly the interaction a web page
// provides: the vision encoder is expensive and depends only on the image, and
// the mask decoder is cheap and depends on the click. Keeping the embedding
// between clicks is what makes the browser version feel as good as the desktop
// one — and it is the entire difference between a sub-100 ms click and a
// one-second one.
//
// Kept free of `@huggingface/transformers` on purpose, exactly as
// `zeroshot/engine.ts` is: the towers arrive as an interface, so the caching
// rules below are unit-testable with two counters and no download.
//
// The rules, and the failure each one prevents:
//
//   *Encode is keyed on a caller-supplied token, not on the pixels.* Hashing a
//   megapixel buffer to answer "is this the same picture" costs more than the
//   answer is worth, and the page already knows — it is the thing the user
//   picked. A token that never changes would pin a stale embedding; the page's
//   token therefore includes the image's identity *and* its dimensions.
//
//   *Decode never encodes.* If nothing is encoded it throws. A lazy re-encode
//   inside `decode` would hide the very cost this module exists to avoid, and
//   the page's "encoding" state would stop meaning anything.
//
//   *Clearing drops the embedding.* A SAM embedding is 256x64x64 floats — 4 MB
//   at fp32 — held for the tab's lifetime otherwise.

import type { ImagePayload } from "../image";
import type { SamMask, SamPoint } from "./types";

/** Whatever the towers need to decode against. Opaque here, by design. */
export type SamEncoded = unknown;

/** The two graphs, as the worker supplies them. */
export interface SamTowers {
  /** Preprocess and run the vision encoder. */
  encode: (image: ImagePayload) => Promise<SamEncoded>;
  /** Decode candidate masks for a set of clicks against a cached embedding. */
  decode: (encoded: SamEncoded, points: readonly SamPoint[]) => Promise<SamMask[]>;
  dispose?: () => Promise<void>;
}

export interface EncodeOutcome {
  cached: boolean;
  ms: number;
  width: number;
  height: number;
}

export interface DecodeOutcome {
  masks: SamMask[];
  ms: number;
}

/**
 * One image's embedding at a time, with the two calls that use it.
 *
 * A plain class rather than a hook or a closure over module state: the engine
 * owns exactly one of these and disposes it with the model, and a class makes
 * that lifetime obvious at the call site.
 */
export class SamSession {
  private encoded: SamEncoded | null = null;
  private token: string | null = null;
  private size: { width: number; height: number } | null = null;

  constructor(private readonly towers: SamTowers) {}

  /** True once an image has been encoded and clicks can be decoded. */
  get ready(): boolean {
    return this.encoded !== null;
  }

  /**
   * Encode `image`, unless `token` matches what is already encoded.
   *
   * The embedding is dropped **before** the new encode rather than after: an
   * encode that throws must not leave the previous image's embedding live to be
   * decoded against clicks on a picture the user is no longer looking at. Same
   * null-first rule as every engine's teardown, and for the same reason.
   */
  async encode(token: string, image: ImagePayload): Promise<EncodeOutcome> {
    if (this.token === token && this.encoded !== null && this.size) {
      return { cached: true, ms: 0, ...this.size };
    }
    this.clear();
    const started = now();
    const encoded = await this.towers.encode(image);
    this.encoded = encoded;
    this.token = token;
    this.size = { width: image.width, height: image.height };
    return {
      cached: false,
      ms: now() - started,
      width: image.width,
      height: image.height,
    };
  }

  /** Decode candidate masks for the current image. Throws if none is encoded. */
  async decode(points: readonly SamPoint[]): Promise<DecodeOutcome> {
    if (this.encoded === null) {
      throw new Error("Encode an image before asking for a mask");
    }
    if (points.length === 0) {
      throw new Error("Click the image to place at least one point");
    }
    const started = now();
    const masks = await this.towers.decode(this.encoded, points);
    return { masks, ms: now() - started };
  }

  /** Forget the current embedding. */
  clear(): void {
    this.encoded = null;
    this.token = null;
    this.size = null;
  }
}

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

/**
 * How much of the frame a mask covers, 0–1.
 *
 * The one number that tells a working point-coordinate space from a broken one.
 * A mask that covers ~0 or ~1 of the picture is what a mis-mapped click
 * produces — the decoder is handed a point outside the object (or outside the
 * image) and returns either nothing or everything, both of which look like a
 * plausible mask until you measure them. The `@slow` spec asserts a band.
 */
export function coverage(mask: SamMask): number {
  const total = mask.width * mask.height;
  if (total === 0) return 0;
  let covered = 0;
  for (let i = 0; i < mask.data.length; i++) {
    if (mask.data[i]) covered++;
  }
  return covered / total;
}
