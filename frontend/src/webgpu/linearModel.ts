// A linear (softmax / multinomial-logistic-regression) classifier and its
// mini-batch SGD training loop. The two heavy matmuls per step — the forward
// logits and the weight gradient — are delegated to an injected `MatmulFn`, so
// the same math runs on the GPU (via runMatmul) in production and on a CPU
// reference in unit tests. Everything else (softmax, cross-entropy, gradients,
// SGD update) is plain, verifiable arithmetic.

/** Matrix multiply C(m×n) = A(m×k) · B(k×n), row-major flattened. */
export type MatmulFn = (
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number,
) => Promise<Float32Array>;

export interface LinearModelShape {
  inputDim: number;
  numClasses: number;
}

export interface Hyperparams {
  learningRate: number;
  batchSize: number;
  epochs: number;
}

export interface TrainMetrics {
  epoch: number;
  step: number;
  totalSteps: number;
  /** Cross-entropy loss on the most recent batch. */
  loss: number;
  /** Accuracy over the whole train set — only set at epoch boundaries. */
  trainAcc: number | null;
  /** Accuracy over the whole test set — only set at epoch boundaries. */
  testAcc: number | null;
}

export interface TrainDataset {
  xTrain: Float32Array;
  yTrain: Uint8Array;
  xTest: Float32Array;
  yTest: Uint8Array;
  trainSize: number;
  testSize: number;
}

export interface FitCallbacks {
  onMetrics?: (metrics: TrainMetrics) => void;
  /** Polled between steps; return true to stop early (used for cancellation). */
  shouldStop?: () => boolean;
}

/** Payload for a `train` worker request. */
export interface TrainRequest {
  shape: LinearModelShape;
  hp: Hyperparams;
  data: TrainDataset;
  seed?: number;
}

/** Final result of a completed (or cancelled) training run. */
export interface TrainResult {
  testAcc: number;
  weights: Float32Array;
  bias: Float32Array;
}

/** Reference CPU matmul — the ground truth the GPU kernel is checked against. */
export function cpuMatmul(
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number,
): Float32Array {
  const c = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const aip = a[i * k + p];
      if (aip === 0) continue;
      for (let j = 0; j < n; j++) c[i * n + j] += aip * b[p * n + j];
    }
  }
  return c;
}

/** Deterministic small-state PRNG so training/shuffling is reproducible. */
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class LinearTrainer {
  /** Weights, `inputDim × numClasses`, row-major. */
  readonly weights: Float32Array;
  /** Bias, one per class. */
  readonly bias: Float32Array;
  private readonly D: number;
  private readonly C: number;
  private readonly rand: () => number;

  constructor(
    private readonly matmul: MatmulFn,
    shape: LinearModelShape,
    seed = 42,
  ) {
    this.D = shape.inputDim;
    this.C = shape.numClasses;
    // Zero init is fine (and deterministic) for a convex softmax classifier.
    this.weights = new Float32Array(this.D * this.C);
    this.bias = new Float32Array(this.C);
    this.rand = mulberry32(seed);
  }

  /** Logits for `n` rows of `x` (n×D) → n×C, with bias added. */
  async forward(x: Float32Array, n: number): Promise<Float32Array> {
    const logits = await this.matmul(x, this.weights, n, this.D, this.C);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < this.C; c++) logits[i * this.C + c] += this.bias[c];
    }
    return logits;
  }

  /** Row-wise softmax (numerically stabilised) over an n×C logit buffer. */
  static softmaxRows(logits: Float32Array, n: number, c: number): Float32Array {
    const probs = new Float32Array(n * c);
    for (let i = 0; i < n; i++) {
      let max = -Infinity;
      for (let j = 0; j < c; j++) max = Math.max(max, logits[i * c + j]);
      let sum = 0;
      for (let j = 0; j < c; j++) {
        const e = Math.exp(logits[i * c + j] - max);
        probs[i * c + j] = e;
        sum += e;
      }
      for (let j = 0; j < c; j++) probs[i * c + j] /= sum;
    }
    return probs;
  }

  /** Mean cross-entropy loss and number correct for softmax `probs` vs labels. */
  static crossEntropy(
    probs: Float32Array,
    y: Uint8Array,
    n: number,
    c: number,
  ): { loss: number; correct: number } {
    let loss = 0;
    let correct = 0;
    for (let i = 0; i < n; i++) {
      const label = y[i];
      loss += -Math.log(Math.max(probs[i * c + label], 1e-12));
      let argmax = 0;
      let best = -Infinity;
      for (let j = 0; j < c; j++) {
        if (probs[i * c + j] > best) {
          best = probs[i * c + j];
          argmax = j;
        }
      }
      if (argmax === label) correct++;
    }
    return { loss: loss / n, correct };
  }

  /**
   * Analytic gradient of the mean cross-entropy over a batch:
   *   dLogits = (softmax − onehot) / B
   *   dW      = Xᵀ · dLogits          (D×C)
   *   db      = colsum(dLogits)        (C)
   */
  async computeGradients(
    x: Float32Array,
    y: Uint8Array,
    batch: number,
  ): Promise<{ dW: Float32Array; db: Float32Array; loss: number; correct: number }> {
    const { C, D } = this;
    const logits = await this.forward(x, batch);
    const probs = LinearTrainer.softmaxRows(logits, batch, C);
    const { loss, correct } = LinearTrainer.crossEntropy(probs, y, batch, C);

    const dLogits = new Float32Array(batch * C);
    for (let i = 0; i < batch; i++) {
      for (let c = 0; c < C; c++) {
        const onehot = c === y[i] ? 1 : 0;
        dLogits[i * C + c] = (probs[i * C + c] - onehot) / batch;
      }
    }

    // xT is D×batch so that xT · dLogits gives dW (D×C).
    const xT = new Float32Array(D * batch);
    for (let i = 0; i < batch; i++) {
      for (let d = 0; d < D; d++) xT[d * batch + i] = x[i * D + d];
    }
    const dW = await this.matmul(xT, dLogits, D, batch, C);

    const db = new Float32Array(C);
    for (let i = 0; i < batch; i++) {
      for (let c = 0; c < C; c++) db[c] += dLogits[i * C + c];
    }

    return { dW, db, loss, correct };
  }

  private applyGradients(dW: Float32Array, db: Float32Array, lr: number): void {
    for (let i = 0; i < this.weights.length; i++) this.weights[i] -= lr * dW[i];
    for (let c = 0; c < this.C; c++) this.bias[c] -= lr * db[c];
  }

  /** One SGD step on a batch; returns the batch loss and count correct. */
  async trainStep(
    x: Float32Array,
    y: Uint8Array,
    batch: number,
    lr: number,
  ): Promise<{ loss: number; correct: number }> {
    const { dW, db, loss, correct } = await this.computeGradients(x, y, batch);
    this.applyGradients(dW, db, lr);
    return { loss, correct };
  }

  /** Fraction of `n` rows of `x` classified correctly. */
  async accuracy(x: Float32Array, y: Uint8Array, n: number): Promise<number> {
    if (n === 0) return 0;
    const logits = await this.forward(x, n);
    let correct = 0;
    for (let i = 0; i < n; i++) {
      let argmax = 0;
      let best = -Infinity;
      for (let c = 0; c < this.C; c++) {
        if (logits[i * this.C + c] > best) {
          best = logits[i * this.C + c];
          argmax = c;
        }
      }
      if (argmax === y[i]) correct++;
    }
    return correct / n;
  }

  /** Run the full mini-batch SGD training loop over `data`. */
  async fit(
    data: TrainDataset,
    hp: Hyperparams,
    cb: FitCallbacks = {},
  ): Promise<void> {
    const { D } = this;
    const { xTrain, yTrain, trainSize } = data;
    const B = hp.batchSize;
    const stepsPerEpoch = Math.ceil(trainSize / B);
    const totalSteps = stepsPerEpoch * hp.epochs;

    // Reusable scratch buffers sized to a full batch.
    const xb = new Float32Array(B * D);
    const yb = new Uint8Array(B);
    const indices = new Int32Array(trainSize);
    for (let i = 0; i < trainSize; i++) indices[i] = i;

    let step = 0;
    for (let epoch = 0; epoch < hp.epochs; epoch++) {
      shuffle(indices, this.rand);

      for (let start = 0; start < trainSize; start += B) {
        if (cb.shouldStop?.()) return;
        const curB = Math.min(B, trainSize - start);
        for (let i = 0; i < curB; i++) {
          const src = indices[start + i] * D;
          xb.set(xTrain.subarray(src, src + D), i * D);
          yb[i] = yTrain[indices[start + i]];
        }

        const { loss } = await this.trainStep(
          xb.subarray(0, curB * D),
          yb.subarray(0, curB),
          curB,
          hp.learningRate,
        );
        step++;

        const lastStepOfEpoch = start + B >= trainSize;
        let trainAcc: number | null = null;
        let testAcc: number | null = null;
        if (lastStepOfEpoch) {
          trainAcc = await this.accuracy(xTrain, yTrain, trainSize);
          testAcc = await this.accuracy(data.xTest, data.yTest, data.testSize);
        }
        cb.onMetrics?.({ epoch, step, totalSteps, loss, trainAcc, testAcc });
      }
    }
  }
}

/** In-place Fisher–Yates shuffle driven by a supplied RNG. */
function shuffle(arr: Int32Array, rand: () => number): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}
