// Link prediction: the same encoder `/graph` trains, with a decoder instead of a
// classifier.
//
// The encoder is `GnnTrainer` unchanged — `Â(XW)` stacked, on the GPU where
// there is one — and its output is read as an **embedding** rather than as class
// logits. The decoder is one dot product per candidate edge:
//
//     score(u, v) = z_u · z_v            p(u, v) = σ(score)
//
// which is why the roadmap calls this the cheapest page in the category: no new
// kernel, no new dataset, and the expensive half of the arithmetic is already
// written. What is new is the **supervision**, and it is new in a way that is
// easy to get subtly wrong:
//
//   - The loss is over *edges*, not nodes, so a pair contributes to two rows of
//     the gradient. Scattering into only one halves it, and the model still
//     trains — just worse, with nothing on screen to say so.
//   - Negatives are resampled every epoch for training, so the encoder cannot
//     learn the sampler, and held fixed for evaluation, so the curve moves only
//     when the model does.
//   - The metric is a ranking metric. See `auc` below for why accuracy is not
//     available here even in principle.
//
// The split that decides which edges the encoder may aggregate over is
// `lib/edgeSplit.ts`, and it is the part that fails silently.

import type { EdgeList } from "@/lib/edgeSplit";
import { sampleNegatives } from "@/lib/edgeSplit";
import { mulberry32 } from "@/lib/random";

import type {
  GnnForward,
  GnnHyperparams,
  GnnInput,
  GnnTrainer,
} from "./gnn";

/** Scores for a list of `[u, v]` pairs, in the order they were given. */
export function edgeScores(
  z: Float32Array,
  dim: number,
  pairs: EdgeList,
): Float32Array {
  const n = pairs.length / 2;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = pairs[2 * i] * dim;
    const b = pairs[2 * i + 1] * dim;
    let dot = 0;
    for (let f = 0; f < dim; f++) dot += z[a + f] * z[b + f];
    out[i] = dot;
  }
  return out;
}

/** σ, but only ever called on scores we already hold. */
export const sigmoid = (x: number): number =>
  x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));

/**
 * Binary cross-entropy of a logit against a label, computed from the logit
 * rather than from σ(logit).
 *
 * `-log(σ(s))` overflows to Infinity for a confidently wrong score once σ
 * rounds to 0 in float64 (around s = -746), and a single Infinity turns the
 * whole epoch's mean loss into NaN — which shows up as a blank chart rather
 * than an error. The stable form is `max(s,0) - s·y + log(1 + exp(-|s|))`.
 */
export function bceWithLogits(score: number, label: number): number {
  return (
    Math.max(score, 0) - score * label + Math.log1p(Math.exp(-Math.abs(score)))
  );
}

export interface DecoderOutput {
  /** Mean BCE over positives and negatives together. */
  loss: number;
  /** ∂loss/∂z, `nNodes × dim`, ready for `GnnTrainer.backwardFrom`. */
  dOut: Float32Array;
}

/**
 * The decoder's loss and its gradient with respect to the embeddings.
 *
 * `∂loss/∂score = (σ(score) − y) / N`, and because `score = z_u · z_v` that
 * gradient flows into **both** endpoints: `∂score/∂z_u = z_v` and vice versa.
 * A pair that updates only `u` still produces a falling loss and a rising AUC,
 * which is exactly the kind of half-right that the finite-difference check in
 * the tests exists to catch.
 */
export function decodeAndGrad(
  z: Float32Array,
  dim: number,
  nNodes: number,
  pos: EdgeList,
  neg: EdgeList,
): DecoderOutput {
  const nPos = pos.length / 2;
  const nNeg = neg.length / 2;
  const total = nPos + nNeg;
  const dOut = new Float32Array(nNodes * dim);
  if (total === 0) return { loss: 0, dOut };

  const scale = 1 / total;
  let loss = 0;

  const accumulate = (pairs: EdgeList, label: number) => {
    const n = pairs.length / 2;
    for (let i = 0; i < n; i++) {
      const u = pairs[2 * i];
      const v = pairs[2 * i + 1];
      const a = u * dim;
      const b = v * dim;
      let score = 0;
      for (let f = 0; f < dim; f++) score += z[a + f] * z[b + f];
      loss += bceWithLogits(score, label);
      const g = (sigmoid(score) - label) * scale;
      for (let f = 0; f < dim; f++) {
        dOut[a + f] += g * z[b + f];
        dOut[b + f] += g * z[a + f];
      }
    }
  };

  accumulate(pos, 1);
  accumulate(neg, 0);
  return { loss: loss * scale, dOut };
}

/**
 * Area under the ROC curve, computed by rank rather than by integrating a curve.
 *
 * At one negative per positive this is exactly *the probability that a real
 * citation outranks a pair that is not one*, which is the sentence the page
 * prints. It is used instead of accuracy because the decoder produces a
 * ranking, and any accuracy number would need a threshold — and the class
 * balance here is a sampling choice, not a property of the data, so a threshold
 * would be measuring the sampler.
 *
 * Ties share the average rank, which is what makes a constant scorer score 0.5
 * instead of 1 or 0.
 */
export function auc(posScores: Float32Array, negScores: Float32Array): number {
  const nPos = posScores.length;
  const nNeg = negScores.length;
  if (nPos === 0 || nNeg === 0) return 0.5;

  const all = new Float64Array(nPos + nNeg);
  all.set(posScores, 0);
  all.set(negScores, nPos);
  const order = Array.from({ length: all.length }, (_, i) => i).sort(
    (a, b) => all[a] - all[b],
  );

  // Ranks, 1-based, averaged within a run of equal scores.
  const rank = new Float64Array(all.length);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && all[order[j + 1]] === all[order[i]]) j++;
    const shared = (i + j + 2) / 2; // mean of the 1-based ranks i+1 … j+1
    for (let k = i; k <= j; k++) rank[order[k]] = shared;
    i = j + 1;
  }

  let sumPos = 0;
  for (let p = 0; p < nPos; p++) sumPos += rank[p];
  return (sumPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

/**
 * Average precision — the area under the precision/recall curve.
 *
 * Reported beside AUC because AUC is generous when negatives outnumber
 * positives, and the candidate list the page draws is a top-k over the ~3.7 M
 * pairs Cora does *not* contain. AP is the number that notices.
 */
export function averagePrecision(
  posScores: Float32Array,
  negScores: Float32Array,
): number {
  const nPos = posScores.length;
  if (nPos === 0) return 0;

  const items: { score: number; positive: boolean }[] = [];
  for (const s of posScores) items.push({ score: s, positive: true });
  for (const s of negScores) items.push({ score: s, positive: false });
  items.sort((a, b) => b.score - a.score);

  let hits = 0;
  let sum = 0;
  for (let i = 0; i < items.length; i++) {
    if (items[i].positive) {
      hits++;
      sum += hits / (i + 1);
    }
  }
  return sum / nPos;
}

export interface LinkMetrics {
  epoch: number;
  totalEpochs: number;
  /** Mean BCE over the training edges and their sampled negatives. */
  loss: number;
  trainAuc: number;
  valAuc: number;
  testAuc: number;
  /** Average precision on the test edges. */
  testAp: number;
}

export interface LinkSplitEdges {
  trainPos: EdgeList;
  valPos: EdgeList;
  valNeg: EdgeList;
  testPos: EdgeList;
  testNeg: EdgeList;
}

export interface LinkFitCallbacks {
  onEpoch?: (metrics: LinkMetrics) => void;
  shouldStop?: () => boolean;
}

export interface LinkFitOptions extends Omit<GnnHyperparams, "arch"> {
  arch: GnnHyperparams["arch"];
  /**
   * The full graph's CSR, used only to reject sampled negatives that are really
   * held-out citations. Sampling against the *training* graph would label every
   * test edge as a negative example and train the model to get them wrong.
   */
  fullRowPtr: Uint32Array;
  fullColIdx: Uint32Array;
  seed?: number;
}

/**
 * Full-batch training of the encoder against the decoder's loss.
 *
 * The evaluation pass is separate from the training pass and runs with dropout
 * off, for the same reason `fitGnn` does it: the training pass's activations are
 * masked, and scoring edges from them understates the model.
 */
export async function fitLinkPredictor(
  trainer: GnnTrainer,
  input: GnnInput,
  edges: LinkSplitEdges,
  options: LinkFitOptions,
  cb: LinkFitCallbacks = {},
): Promise<LinkMetrics | null> {
  const { nNodes } = trainer.shape;
  const dim = trainer.dims[trainer.dims.length - 1];
  trainer.setInput(input);

  const rand = mulberry32(options.seed ?? 11);
  let last: LinkMetrics | null = null;

  for (let epoch = 0; epoch < options.epochs; epoch++) {
    if (cb.shouldStop?.()) return last;

    // Fresh negatives every epoch: with a fixed set the encoder can drive those
    // particular pairs apart instead of learning what a citation looks like.
    const trainNeg = sampleNegatives(
      options.fullRowPtr,
      options.fullColIdx,
      nNodes,
      edges.trainPos.length / 2,
      rand,
    );

    const pass = await trainer.forward(options.dropout, true);
    const { loss, dOut } = decodeAndGrad(
      pass.logits,
      dim,
      nNodes,
      edges.trainPos,
      trainNeg,
    );
    const grads = await trainer.backwardFrom(dOut, loss);
    trainer.applyGradients(grads, options);

    const evalPass: GnnForward = await trainer.forward(0, false);
    const z = evalPass.logits;

    last = {
      epoch,
      totalEpochs: options.epochs,
      loss,
      trainAuc: auc(
        edgeScores(z, dim, edges.trainPos),
        edgeScores(z, dim, trainNeg),
      ),
      valAuc: auc(
        edgeScores(z, dim, edges.valPos),
        edgeScores(z, dim, edges.valNeg),
      ),
      testAuc: auc(
        edgeScores(z, dim, edges.testPos),
        edgeScores(z, dim, edges.testNeg),
      ),
      testAp: averagePrecision(
        edgeScores(z, dim, edges.testPos),
        edgeScores(z, dim, edges.testNeg),
      ),
    };
    cb.onEpoch?.(last);
  }
  return last;
}
