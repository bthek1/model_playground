// Graph classification: one label per graph rather than one per node.
//
// The third head over the encoder in `gnn.ts`, after node classification's
// softmax and `/link-prediction`'s dot-product decoder. What is new is a single
// reduction — a **readout** that turns `nNodes × dim` node representations into
// `nGraphs × dim` — and its adjoint.
//
// **The head needs no parameters at all**, and that is worth the paragraph
// because the obvious design has some. Mean pooling and a linear layer commute:
//
//     mean_g(W · h_i)  =  W · mean_g(h_i)
//
// so a final GNN layer of width `nClasses` followed by a pool over each graph's
// nodes computes the same function as pooling into a hidden vector and
// classifying it. Reading out the *logits* therefore costs no second weight
// matrix, no second Adam state, and no new kernel — only this file. The page says
// it is a single linear readout, because the honest difference from GIN's paper
// (an MLP over summed per-layer representations) is worth one sentence rather
// than a silent approximation.
//
// Sum and mean are both offered, and the choice is the same *kind* of choice as
// GIN-versus-GCN: mean is invariant to graph size, sum is not — and on PROTEINS
// size is signal, because enzymes are bigger. The gradient differs by exactly the
// `1/n_g` that makes one invariant and the other not, which is why the
// finite-difference check in the tests runs over both modes.

import type { GnnArch, GnnForward, GnnTrainer } from "./gnn";
import { crossEntropyLoss, softmaxRows } from "./gnn";
import type { GraphSplitIndices } from "@/lib/proteins";

export type ReadoutMode = "mean" | "sum";

export const READOUTS: Record<ReadoutMode, { label: string; note: string }> = {
  mean: {
    label: "Mean",
    note: "Average over the graph's nodes — invariant to how big the graph is.",
  },
  sum: {
    label: "Sum",
    note: "Total over the graph's nodes — carries size, which on PROTEINS is itself a clue.",
  },
};

/**
 * Reduce `nNodes × dim` to `nGraphs × dim`.
 *
 * `graphPtr` is the union's node ranges: graph `g` owns `[graphPtr[g],
 * graphPtr[g+1])`. An empty graph contributes a zero row rather than a NaN one —
 * the union forbids zero-node graphs, but a divide that can produce NaN in
 * principle will do it in practice one dataset later.
 */
export function poolForward(
  h: Float32Array,
  dim: number,
  graphPtr: Uint32Array,
  mode: ReadoutMode,
): Float32Array {
  const nGraphs = graphPtr.length - 1;
  const out = new Float32Array(nGraphs * dim);
  for (let g = 0; g < nGraphs; g++) {
    const start = graphPtr[g];
    const end = graphPtr[g + 1];
    const count = end - start;
    if (count === 0) continue;
    const row = g * dim;
    for (let i = start; i < end; i++) {
      const src = i * dim;
      for (let f = 0; f < dim; f++) out[row + f] += h[src + f];
    }
    if (mode === "mean") {
      for (let f = 0; f < dim; f++) out[row + f] /= count;
    }
  }
  return out;
}

/**
 * The adjoint: scatter each graph's gradient back to its own nodes.
 *
 * Every node of graph `g` receives the whole of `dOut[g]`, divided by `n_g` for
 * a mean. That `1/n_g` is invisible in a loss curve — drop it and the model still
 * trains, just with a per-graph learning rate proportional to size — which is
 * exactly why this is checked against finite differences rather than by eye.
 */
export function poolBackward(
  dOut: Float32Array,
  dim: number,
  graphPtr: Uint32Array,
  mode: ReadoutMode,
): Float32Array {
  const nGraphs = graphPtr.length - 1;
  const nNodes = graphPtr[nGraphs];
  const out = new Float32Array(nNodes * dim);
  for (let g = 0; g < nGraphs; g++) {
    const start = graphPtr[g];
    const end = graphPtr[g + 1];
    const count = end - start;
    if (count === 0) continue;
    const row = g * dim;
    const scale = mode === "mean" ? 1 / count : 1;
    for (let i = start; i < end; i++) {
      const dst = i * dim;
      for (let f = 0; f < dim; f++) out[dst + f] = dOut[row + f] * scale;
    }
  }
  return out;
}

/** The class each graph is predicted to be, from pooled logits. */
export function predictGraphs(
  logits: Float32Array,
  nGraphs: number,
  nClasses: number,
): Uint8Array {
  const out = new Uint8Array(nGraphs);
  for (let g = 0; g < nGraphs; g++) {
    const row = g * nClasses;
    let best = 0;
    for (let c = 1; c < nClasses; c++) {
      if (logits[row + c] > logits[row + best]) best = c;
    }
    out[g] = best;
  }
  return out;
}

/** Accuracy over a subset of graphs. */
export function graphAccuracy(
  predicted: Uint8Array,
  labels: Uint8Array,
  idx: Uint32Array,
): number {
  if (idx.length === 0) return 0;
  let hits = 0;
  for (const g of idx) if (predicted[g] === labels[g]) hits++;
  return hits / idx.length;
}

export interface GraphClassMetrics {
  epoch: number;
  totalEpochs: number;
  loss: number;
  trainAcc: number;
  valAcc: number;
  testAcc: number;
  /**
   * What a model that ignores its input entirely would score on the same test
   * split. Carried in the metrics rather than computed by the page, so the two
   * numbers can never be derived from different splits.
   */
  baselineAcc: number;
}

export interface GraphClassOptions {
  arch: GnnArch;
  learningRate: number;
  weightDecay: number;
  dropout: number;
  epochs: number;
  readout: ReadoutMode;
}

export interface GraphClassCallbacks {
  onEpoch?: (metrics: GraphClassMetrics, predicted: Uint8Array) => void;
  shouldStop?: () => boolean;
}

/**
 * Full-batch training over the whole disjoint union.
 *
 * One forward pass covers every node of every graph, so the train, validation and
 * test accuracies all fall out of the pass the gradient needed anyway — the same
 * property `fitGnn` has, for the same reason.
 */
export async function fitGraphClassifier(
  trainer: GnnTrainer,
  graphPtr: Uint32Array,
  labels: Uint8Array,
  split: GraphSplitIndices,
  baselineAcc: number,
  options: GraphClassOptions,
  cb: GraphClassCallbacks = {},
): Promise<GraphClassMetrics | null> {
  const nGraphs = graphPtr.length - 1;
  const nClasses = trainer.shape.nClasses;
  let last: GraphClassMetrics | null = null;

  for (let epoch = 0; epoch < options.epochs; epoch++) {
    if (cb.shouldStop?.()) return last;

    const pass = await trainer.forward(options.dropout, true);
    const pooled = poolForward(pass.logits, nClasses, graphPtr, options.readout);

    const probs = softmaxRows(pooled, nGraphs, nClasses);
    const loss = crossEntropyLoss(probs, labels, split.train, nClasses);
    const dPooled = new Float32Array(nGraphs * nClasses);
    const scale = 1 / split.train.length;
    for (const g of split.train) {
      const row = g * nClasses;
      const label = labels[g];
      for (let c = 0; c < nClasses; c++) {
        dPooled[row + c] = (probs[row + c] - (c === label ? 1 : 0)) * scale;
      }
    }

    const grads = await trainer.backwardFrom(
      poolBackward(dPooled, nClasses, graphPtr, options.readout),
      loss,
    );
    trainer.applyGradients(grads, options);

    // Dropout off for the numbers on screen: the training pass's activations are
    // masked, and reporting accuracy from them understates the model.
    const evalPass: GnnForward = await trainer.forward(0, false);
    const evalPooled = poolForward(
      evalPass.logits,
      nClasses,
      graphPtr,
      options.readout,
    );
    const predicted = predictGraphs(evalPooled, nGraphs, nClasses);

    last = {
      epoch,
      totalEpochs: options.epochs,
      loss,
      trainAcc: graphAccuracy(predicted, labels, split.train),
      valAcc: graphAccuracy(predicted, labels, split.val),
      testAcc: graphAccuracy(predicted, labels, split.test),
      baselineAcc,
    };
    cb.onEpoch?.(last, predicted);
  }
  return last;
}
