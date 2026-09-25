// The pure half of `/text-features` and `/sentence-similarity`: turning a
// pipeline's tensor into a vector, truncating it the way a Matryoshka model
// invites you to, and remembering what has already been embedded.
//
// Nothing here touches a model, a worker or React, which is the point — every
// claim the two pages make about embeddings is arithmetic that can be pinned by
// a test. The three that matter:
//
//   **Truncation renormalises.** Dropping the tail of a unit vector leaves a
//   shorter vector that is *no longer unit length*, so a cosine computed
//   against a full-length vector is scaled by an arbitrary factor. The slider
//   still moves, the similarity still changes, and the numbers are wrong — a
//   working-looking control is the failure mode, which is why `truncate`
//   returns the norm it discarded as well as the vector it kept.
//
//   **The cache is keyed by the string itself, not by a hash of it.** A hash is
//   the obvious thing to reach for and it is strictly worse here: a collision
//   returns *another sentence's embedding* with nothing failing, which on a
//   similarity page is a confident wrong number. The strings are the user's own
//   input — kilobytes, in a Map, in one tab — so there is nothing to save.
//
//   **A cache outlives an input but never a checkpoint.** A 384-dim MiniLM
//   vector and a 768-dim BGE vector are not comparable, and neither are two
//   384-dim vectors from different models. Switching model clears it; changing
//   the text does not.

import type { PlainTensor } from "@/model/serialize";
import { l2norm, normalize } from "@/model/similarity";

/**
 * Flatten the `feature-extraction` pipeline's output to one vector.
 *
 * With `{ pooling: "mean" }` pinned in the engine the tensor is `[1, dim]`; a
 * bare `[dim]` is accepted for the same reason `poolEmbedding` accepts it. A
 * three-dimensional tensor is the *unpooled* last hidden state — i.e. the
 * pooling option did not reach the model — and it throws rather than silently
 * taking row 0, because row 0 of a BERT hidden state is a real vector that
 * would rank plausibly and be wrong.
 */
export function toVector(tensor: PlainTensor): Float32Array {
  const dims = tensor.dims ?? [];
  const data = tensor.data ?? [];
  if (dims.length === 3) {
    throw new Error(
      `Unpooled embedding [${dims.join(", ")}] — the pooling option did not reach the model`,
    );
  }
  const dim = dims.length > 0 ? dims[dims.length - 1] : data.length;
  if (!(dim > 0)) {
    throw new Error(`Empty embedding: dims [${dims.join(", ")}]`);
  }
  const out = new Float32Array(dim);
  for (let i = 0; i < dim; i++) out[i] = data[i];
  return out;
}

/** A vector truncated to a prefix, with what that cost in length. */
export interface Truncated {
  /** The prefix, renormalised to unit length. */
  vector: Float32Array;
  /** Dimensions kept. */
  dim: number;
  /**
   * The prefix's L2 norm **before** renormalising, as a fraction of the whole
   * vector's. 1 means the tail held nothing; 0.6 means 40% of the vector's
   * length was in the dimensions just dropped.
   *
   * This is the number the Matryoshka claim is actually about, and it is also
   * the page's own self-check: it is the factor a *missing* renormalisation
   * would scale every similarity by.
   */
  kept: number;
}

/**
 * Keep the first `dim` dimensions of a vector and renormalise.
 *
 * `dim` beyond the vector's width returns the whole vector (renormalised), and
 * a non-positive `dim` is treated as 1 — the caller is a slider, and a slider
 * that can produce an empty vector produces `NaN` scores instead of an error.
 */
export function truncate(vector: ArrayLike<number>, dim: number): Truncated {
  const width = Math.max(1, Math.min(Math.round(dim) || 1, vector.length));
  const prefix = new Float32Array(width);
  for (let i = 0; i < width; i++) prefix[i] = vector[i];

  const whole = l2norm(vector);
  const part = l2norm(prefix);
  return {
    vector: normalize(prefix),
    dim: width,
    kept: whole === 0 ? 0 : part / whole,
  };
}

/**
 * The truncation steps offered for a given embedding width.
 *
 * Halving down to 64, full width first. Derived from the model's own `dim`
 * rather than listed per entry: a 384-dim model has no 768 step to offer, and a
 * hard-coded list is how a control ends up proposing a width the vector does
 * not have.
 */
export function truncationSteps(dim: number): number[] {
  const steps: number[] = [];
  for (let d = dim; d >= 64; d = Math.floor(d / 2)) steps.push(d);
  return steps;
}

/**
 * Embeddings already computed, keyed by the exact text that produced them.
 *
 * Deliberately a plain `Map` with an insertion-order cap rather than anything
 * cleverer. `/text-ranking` pastes a corpus through this, so the cap is what
 * stops a session's worth of edits holding every intermediate version of every
 * document; the order is insertion rather than use because the expensive case
 * is a corpus embedded once and then queried, and re-ordering on read would
 * evict the corpus to keep the queries.
 */
export class EmbeddingCache {
  private map = new Map<string, Float32Array>();

  constructor(private readonly limit = 512) {}

  get(text: string): Float32Array | undefined {
    return this.map.get(text);
  }

  has(text: string): boolean {
    return this.map.has(text);
  }

  set(text: string, vector: Float32Array): void {
    if (this.map.size >= this.limit && !this.map.has(text)) {
      const oldest = this.map.keys().next();
      if (!oldest.done) this.map.delete(oldest.value);
    }
    this.map.set(text, vector);
  }

  /** Drop everything — the only correct response to a checkpoint change. */
  clear(): void {
    this.map.clear();
  }

  get size(): number {
    return this.map.size;
  }
}
