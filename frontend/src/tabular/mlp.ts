// The ladder's deep baseline — one hidden layer, ReLU, softmax — shipped
// **because it loses**.
//
// "Deep learning does not win on tabular data" is the most useful thing this
// category has to say, and it is a claim people do not believe until they watch
// it happen on their own file. A page that asserted it, or that omitted the MLP
// and explained why, would prove nothing. So this is a genuine fit: same design
// matrix, same split, same held-out metric block, same GPU matmuls as the
// logistic floor — and it is allowed to be beaten by a hundred depth-6 trees.
//
// The two heavy matmuls per layer go through the same `MatmulFn` seam as
// `webgpu/linearModel.ts`, so a CPU reference checks the arithmetic in a test.
// Everything else — ReLU, its derivative, the softmax cross-entropy, the SGD
// update — is plain verifiable arithmetic, and is checked by finite differences
// at a **generic point**: zero-initialised biases put some preactivations
// exactly on ReLU's kink, where a central difference reports half the true
// gradient and a correct implementation looks broken (`webgpu/gnn.test.ts` paid
// for that one).

import { gaussian, mulberry32, shuffle } from "@/lib/random";
import { LinearTrainer, type MatmulFn } from "@/webgpu/linearModel";

export interface MlpOptions {
  hidden: number;
  epochs: number;
  learningRate: number;
  batchSize: number;
  seed: number;
  onProgress?: (done: number, total: number, loss: number | null) => void;
  shouldStop?: () => boolean;
}

export interface MlpParams {
  /** `features × hidden`, row-major. */
  w1: Float32Array;
  b1: Float32Array;
  /** `hidden × classes`, row-major. */
  w2: Float32Array;
  b2: Float32Array;
  features: number;
  hidden: number;
  classes: number;
}

/** He initialisation — the variance ReLU was analysed with. */
export function initMlp(
  features: number,
  hidden: number,
  classes: number,
  seed: number,
): MlpParams {
  const rand = mulberry32(seed);
  const w1 = new Float32Array(features * hidden);
  const w2 = new Float32Array(hidden * classes);
  const s1 = Math.sqrt(2 / features);
  const s2 = Math.sqrt(2 / hidden);
  for (let i = 0; i < w1.length; i++) w1[i] = gaussian(rand) * s1;
  for (let i = 0; i < w2.length; i++) w2[i] = gaussian(rand) * s2;
  return {
    w1,
    b1: new Float32Array(hidden),
    w2,
    b2: new Float32Array(classes),
    features,
    hidden,
    classes,
  };
}

/** Hidden preactivations, activations and class probabilities for `n` rows. */
export async function forwardMlp(
  matmul: MatmulFn,
  p: MlpParams,
  x: Float32Array,
  n: number,
): Promise<{ z1: Float32Array; a1: Float32Array; probs: Float32Array }> {
  const z1 = await matmul(x, p.w1, n, p.features, p.hidden);
  const a1 = new Float32Array(n * p.hidden);
  for (let i = 0; i < n; i++) {
    for (let h = 0; h < p.hidden; h++) {
      const v = z1[i * p.hidden + h] + p.b1[h];
      z1[i * p.hidden + h] = v;
      a1[i * p.hidden + h] = v > 0 ? v : 0;
    }
  }
  const logits = await matmul(a1, p.w2, n, p.hidden, p.classes);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < p.classes; c++) logits[i * p.classes + c] += p.b2[c];
  }
  return { z1, a1, probs: LinearTrainer.softmaxRows(logits, n, p.classes) };
}

export interface MlpGradients {
  dw1: Float32Array;
  db1: Float32Array;
  dw2: Float32Array;
  db2: Float32Array;
  loss: number;
}

/** Analytic gradients of the mean cross-entropy over a batch. */
export async function backwardMlp(
  matmul: MatmulFn,
  p: MlpParams,
  x: Float32Array,
  y: Uint8Array,
  n: number,
): Promise<MlpGradients> {
  const { z1, a1, probs } = await forwardMlp(matmul, p, x, n);
  const { loss } = LinearTrainer.crossEntropy(probs, y, n, p.classes);

  // dLogits = (softmax − onehot) / n
  const dLogits = new Float32Array(n * p.classes);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < p.classes; c++) {
      dLogits[i * p.classes + c] =
        (probs[i * p.classes + c] - (y[i] === c ? 1 : 0)) / n;
    }
  }

  const a1T = transpose(a1, n, p.hidden);
  const dw2 = await matmul(a1T, dLogits, p.hidden, n, p.classes);
  const db2 = new Float32Array(p.classes);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < p.classes; c++) db2[c] += dLogits[i * p.classes + c];
  }

  // dA1 = dLogits · W2ᵀ, then through ReLU's derivative.
  const w2T = transpose(p.w2, p.hidden, p.classes);
  const dA1 = await matmul(dLogits, w2T, n, p.classes, p.hidden);
  const dZ1 = new Float32Array(n * p.hidden);
  for (let i = 0; i < n * p.hidden; i++) {
    dZ1[i] = z1[i] > 0 ? dA1[i] : 0;
  }

  const xT = transpose(x, n, p.features);
  const dw1 = await matmul(xT, dZ1, p.features, n, p.hidden);
  const db1 = new Float32Array(p.hidden);
  for (let i = 0; i < n; i++) {
    for (let h = 0; h < p.hidden; h++) db1[h] += dZ1[i * p.hidden + h];
  }

  return { dw1, db1, dw2, db2, loss };
}

function transpose(a: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) out[j * rows + i] = a[i * cols + j];
  }
  return out;
}

export interface MlpModel {
  params: MlpParams;
  predict: (x: Float32Array, rows: number) => Promise<Float32Array>;
}

export async function fitMlp(
  matmul: MatmulFn,
  x: Float32Array,
  y: Uint8Array,
  rows: number,
  features: number,
  classes: number,
  options: MlpOptions,
): Promise<MlpModel> {
  const p = initMlp(features, options.hidden, classes, options.seed);
  const rand = mulberry32(options.seed ^ 0x5bf03635);
  const B = Math.min(options.batchSize, Math.max(1, rows));
  const indices = new Int32Array(rows);
  for (let i = 0; i < rows; i++) indices[i] = i;
  const xb = new Float32Array(B * features);
  const yb = new Uint8Array(B);

  for (let epoch = 0; epoch < options.epochs; epoch++) {
    if (options.shouldStop?.()) break;
    shuffle(indices, rand);
    let epochLoss = 0;
    let batches = 0;
    for (let start = 0; start < rows; start += B) {
      if (options.shouldStop?.()) break;
      const cur = Math.min(B, rows - start);
      for (let i = 0; i < cur; i++) {
        const src = indices[start + i] * features;
        xb.set(x.subarray(src, src + features), i * features);
        yb[i] = y[indices[start + i]];
      }
      const g = await backwardMlp(
        matmul,
        p,
        xb.subarray(0, cur * features),
        yb.subarray(0, cur),
        cur,
      );
      const lr = options.learningRate;
      for (let i = 0; i < p.w1.length; i++) p.w1[i] -= lr * g.dw1[i];
      for (let i = 0; i < p.b1.length; i++) p.b1[i] -= lr * g.db1[i];
      for (let i = 0; i < p.w2.length; i++) p.w2[i] -= lr * g.dw2[i];
      for (let i = 0; i < p.b2.length; i++) p.b2[i] -= lr * g.db2[i];
      epochLoss += g.loss;
      batches++;
    }
    options.onProgress?.(epoch + 1, options.epochs, batches ? epochLoss / batches : null);
  }

  return {
    params: p,
    predict: async (xs, n) =>
      n === 0 ? new Float32Array(0) : (await forwardMlp(matmul, p, xs, n)).probs,
  };
}
