// Logistic regression — the ladder's floor, and the half of it that belongs on
// the GPU.
//
// There is almost nothing here, which is the point: `webgpu/linearModel.ts`
// already owns multinomial logistic regression, its softmax, its cross-entropy
// and its mini-batch SGD loop, all of it behind the injected `MatmulFn` seam so
// the same arithmetic runs on WGSL in the worker and on a CPU reference in a
// test. Writing a second copy for this category would be writing a second set
// of gradients to get wrong.
//
// What this file adds is the adapter: a design matrix and a label vector in, a
// fitted model and its held-out probabilities out, with the iteration counter
// the FIT band draws.

import {
  cpuMatmul,
  LinearTrainer,
  type MatmulFn,
} from "@/webgpu/linearModel";

export { cpuMatmul, type MatmulFn };

export interface LinearFitOptions {
  epochs: number;
  learningRate: number;
  batchSize: number;
  seed: number;
  onProgress?: (done: number, total: number, loss: number | null) => void;
  shouldStop?: () => boolean;
}

export interface LinearModel {
  weights: Float32Array;
  bias: Float32Array;
  inputDim: number;
  numClasses: number;
  predict: (x: Float32Array, rows: number) => Promise<Float32Array>;
}

/**
 * Fit a softmax classifier on a standardised design matrix.
 *
 * `matmul` is supplied by the caller — `runMatmul` in the worker, `cpuMatmul`
 * in a test. The reason that seam exists at all is that a GPU kernel and a CPU
 * reference disagreeing is a *silent* failure: the loss still falls.
 */
export async function fitLogistic(
  matmul: MatmulFn,
  x: Float32Array,
  y: Uint8Array,
  rows: number,
  features: number,
  numClasses: number,
  options: LinearFitOptions,
): Promise<LinearModel> {
  const trainer = new LinearTrainer(
    matmul,
    { inputDim: features, numClasses },
    options.seed,
  );

  await trainer.fit(
    {
      xTrain: x,
      yTrain: y,
      // The loop evaluates held-out accuracy at each epoch boundary; this page
      // scores its own held-out half afterwards, with the full metric block and
      // its baseline, so the trainer is handed an empty test set rather than a
      // second copy of the data to walk every epoch.
      xTest: new Float32Array(0),
      yTest: new Uint8Array(0),
      trainSize: rows,
      testSize: 0,
    },
    {
      epochs: options.epochs,
      learningRate: options.learningRate,
      batchSize: Math.min(options.batchSize, Math.max(1, rows)),
    },
    {
      onMetrics: (m) => {
        // Reported per epoch, not per step: a fast worker posting a message per
        // mini-batch floods React, which is the whole reason `useLinearTraining`
        // rAF-batches its own flush.
        if (m.trainAcc != null || m.step % Math.max(1, Math.floor(m.totalSteps / options.epochs)) === 0) {
          options.onProgress?.(m.epoch + 1, options.epochs, m.loss);
        }
      },
      shouldStop: options.shouldStop,
    },
  );

  return {
    weights: trainer.weights,
    bias: trainer.bias,
    inputDim: features,
    numClasses,
    predict: async (xs, n) => {
      if (n === 0) return new Float32Array(0);
      const logits = await trainer.forward(xs, n);
      return LinearTrainer.softmaxRows(logits, n, numClasses);
    },
  };
}
