// Cosine similarity and top-k over an in-memory index. Pure and synchronous, so
// re-ranking never re-runs a model: changing `k`, or switching which vector the
// page is comparing, is arithmetic over numbers already in hand — the same
// pure-derivation rule the detection threshold and the VAD threshold follow.
//
// The index is a plain array rather than any kind of ANN structure on purpose.
// A dozen gallery images at 384 dimensions is 4,608 floats; an exhaustive scan
// is microseconds, and every approximate index would buy nothing while adding a
// build step whose bugs are invisible.

/** One vector in the index, with whatever the caller needs to identify it. */
export interface IndexedVector {
  id: string;
  /** Expected to be unit length — see {@link normalize}. */
  vector: Float32Array;
}

export interface Neighbour {
  id: string;
  /** Cosine similarity in [-1, 1]. 1 is the vector itself. */
  score: number;
}

/** Euclidean length. The number the page displays to make normalising visible. */
export function l2norm(vector: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < vector.length; i++) sum += vector[i] * vector[i];
  return Math.sqrt(sum);
}

/**
 * Scale a vector to unit length.
 *
 * A zero vector is returned unchanged rather than producing `NaN`s: it can only
 * come from a model that failed to produce anything, and a page full of `NaN`
 * scores hides that failure behind arithmetic noise.
 */
export function normalize(vector: ArrayLike<number>): Float32Array {
  const norm = l2norm(vector);
  const out = new Float32Array(vector.length);
  if (norm === 0) return out;
  for (let i = 0; i < vector.length; i++) out[i] = vector[i] / norm;
  return out;
}

/**
 * Cosine similarity. Computed from the dot product and both norms rather than
 * assuming unit inputs — the assumption is exactly the bug this page exists to
 * make visible, and it would make a broken embedding look plausible.
 */
export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * The `k` most similar entries, best first.
 *
 * Ties keep index order, so a re-rank at the same `k` is stable — a list that
 * reshuffles equal scores between renders reads as flicker, not as a result. An
 * empty index returns an empty list rather than throwing: a gallery that has not
 * finished embedding is the normal state of this page for its first few seconds.
 */
export function topK(
  query: ArrayLike<number>,
  index: readonly IndexedVector[],
  k: number,
): Neighbour[] {
  const scored = index.map((entry, i) => ({
    id: entry.id,
    score: cosine(query, entry.vector),
    i,
  }));
  scored.sort((a, b) => b.score - a.score || a.i - b.i);
  return scored.slice(0, Math.max(0, k)).map(({ id, score }) => ({ id, score }));
}
