// CLIP/SigLIP scoring, as pure arithmetic.
//
// The zero-shot route drives the text and vision towers separately so the text
// side can be encoded once and reused (docs/roadmaps/vision.md §5). That buys the
// caching, and it costs us the last few lines of the model: the full `model.onnx`
// graph ends with a normalise, a matmul and a scale, and running the towers alone
// means doing those three steps ourselves.
//
// **This is the part that fails silently**, which is why it lives in its own
// module with no runtime dependency and is tested against numbers taken from a
// real pipeline run. Get the scale wrong and every score is still in [0, 1],
// still sums to 1, still ranks the labels correctly — and is simply not what the
// model said. Ranking is preserved by *any* positive scale, so a ranking
// assertion cannot catch it. Only comparing the numbers can.
//
// The two families differ in their last step, and the difference is the whole
// reason SigLIP's scores mean something on their own:
//
//   CLIP    logits = scale · (img · txt)          then softmax over the labels
//   SigLIP  logits = scale · (img · txt) + bias   then sigmoid, per label
//
// `scale` is `exp(logit_scale)` and `bias` is `logit_bias`, both learned
// parameters read out of each checkpoint's own weights — see `zeroShot.ts`.

/** How a model turns cosine similarities into the numbers it reports. */
export interface ScoringSpec {
  kind: "softmax" | "sigmoid";
  /** `exp(logit_scale)` from the checkpoint. ~100 for CLIP. */
  scale: number;
  /** `logit_bias`. Sigmoid models only; ignored for softmax. */
  bias?: number;
}

/**
 * L2-normalise a row in place and return it. Both towers' outputs are normalised
 * before the matmul — that is what makes the product a cosine similarity rather
 * than an arbitrary dot product, and both `CLIPModel` and `SiglipModel` do it.
 */
export function l2Normalize(row: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < row.length; i++) sum += row[i] * row[i];
  // A zero vector has no direction; leaving it alone beats dividing by zero and
  // filling the row with NaN, which would poison every score downstream.
  const norm = Math.sqrt(sum);
  if (norm === 0) return row;
  for (let i = 0; i < row.length; i++) row[i] /= norm;
  return row;
}

/** Split a flat `[rows x dim]` embedding matrix into normalised rows. */
export function normalizedRows(
  flat: ArrayLike<number>,
  rows: number,
  dim: number,
): Float32Array[] {
  const out: Float32Array[] = [];
  for (let r = 0; r < rows; r++) {
    const row = new Float32Array(dim);
    for (let c = 0; c < dim; c++) row[c] = flat[r * dim + c];
    out.push(l2Normalize(row));
  }
  return out;
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let sum = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) sum += a[i] * b[i];
  return sum;
}

/** Numerically stable softmax — subtracting the max keeps `exp` in range. */
export function softmax(logits: readonly number[]): number[] {
  if (logits.length === 0) return [];
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const total = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((x) => x / total);
}

export function sigmoid(x: number): number {
  // Branch on the sign so `exp` never sees a large positive argument, which
  // would overflow to Infinity and return NaN for a confident negative score.
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

/**
 * Score one image against a set of label embeddings — the three lines of the
 * model we took on when we split the towers.
 *
 * Both arguments must already be L2-normalised (`normalizedRows` does it).
 */
export function scoreImage(
  imageEmbed: Float32Array,
  textEmbeds: readonly Float32Array[],
  spec: ScoringSpec,
): number[] {
  const bias = spec.kind === "sigmoid" ? (spec.bias ?? 0) : 0;
  const logits = textEmbeds.map((t) => spec.scale * dot(imageEmbed, t) + bias);
  return spec.kind === "sigmoid" ? logits.map(sigmoid) : softmax(logits);
}
