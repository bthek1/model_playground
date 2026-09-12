// A graph convolutional network and its full-batch training loop, written the
// same way linearModel.ts is: the heavy operations are **injected**, so the
// identical arithmetic runs on the GPU in the worker and against a CPU reference
// in unit tests. Everything else — softmax, cross-entropy, the backward pass,
// Adam — is plain verifiable arithmetic with a finite-difference gradient check
// over it.
//
// The whole of message passing here is one operation:
//
//     out[i] = Σ_{j ∈ N(i) ∪ {i}}  α_i · β_j · x[j]
//
// a **scaled gather** over the graph's CSR arrays. Every architecture but GAT is
// a choice of the two scale vectors:
//
//     GCN        α_i = β_i = 1/√(deg_i + 1)      symmetric, D^-1/2 (A+I) D^-1/2
//     GraphSAGE  α_i = 1/(deg_i + 1),  β_i = 1    mean,      D^-1 (A+I)
//     GIN        α_i = β_i = 1                    sum,       (A+I), ε = 0
//
// Two things about that formulation are load-bearing:
//
//   - **The self-loop is not stored, it is added here.** `A_hat = A + I`. Without
//     it a node's own features are discarded at every layer and the model can
//     only ever see its neighbourhood. The CSR arrays hold no self-loops
//     precisely so this term cannot be double-counted.
//   - **The transpose the backward pass needs is the same kernel with α and β
//     swapped.** `A + I` is symmetric, so (α_i β_j)ᵀ = α_j β_i. That is why one
//     shader serves forward and backward for all three architectures, and why
//     `cora.test.ts` asserts the graph really is symmetric.

import { gaussian, mulberry32 } from "@/lib/random";

import type { MatmulFn } from "./linearModel";

export type GnnArch = "gcn" | "sage" | "gin";

/** Human-facing names and the one-line description each row needs in SELECT. */
export const GNN_ARCHITECTURES: Record<
  GnnArch,
  { label: string; aggregation: string; note: string }
> = {
  gcn: {
    label: "GCN",
    aggregation: "D^-1/2 (A+I) D^-1/2",
    note: "Symmetric normalisation — a high-degree neighbour counts for less in both directions.",
  },
  sage: {
    label: "GraphSAGE",
    aggregation: "D^-1 (A+I)",
    note: "Plain mean over the neighbourhood. Mean-pooling variant, with the self term sharing one weight matrix.",
  },
  gin: {
    label: "GIN",
    aggregation: "(A+I)",
    note: "Unnormalised sum (ε = 0). The most expressive aggregation, and the one whose activations grow with depth.",
  },
};

/**
 * A scaled gather over a fixed graph: `out[i] = Σ_{j ∈ N(i) ∪ {i}} α_i·β_j·x[j]`
 * for an `nNodes × nFeat` row-major `x`.
 *
 * Implementations close over both the graph and the architecture's scale
 * vectors, because neither changes during a run — on the GPU they are uploaded
 * once and left on the device for every epoch. `transposed` asks for Âᵀ instead,
 * which is the same gather with α and β swapped; the trainer therefore never
 * handles the scales itself and cannot mix the two up.
 */
export type AggregateFn = (
  x: Float32Array,
  nFeat: number,
  transposed: boolean,
) => Promise<Float32Array>;

/**
 * The operations the trainer does not implement itself. `matmul` and `aggregate`
 * are the two that run on the GPU in production and on a CPU reference in tests;
 * `smoothness` needs the graph's edge list, which the trainer is deliberately not
 * given (it only ever sees the two scale vectors).
 */
export interface GnnOps {
  matmul: MatmulFn;
  aggregate: AggregateFn;
  /** Mean cosine similarity between adjacent nodes' rows of an `n × nFeat` H. */
  smoothness: (h: Float32Array, nFeat: number) => number;
}

/**
 * Where the nonzero entries of a sparse feature matrix live, in both the
 * row-major and the transposed dense layouts. Cora's bag-of-words is 1.27 %
 * dense, so this is 49 216 entries rather than 3.9 M, and it is what makes input
 * dropout affordable — see `GnnTrainer.forward`.
 */
export interface SparsePattern {
  /** Index into the dense row-major X. */
  nz: Uint32Array;
  /** The same entry's index into Xᵀ. */
  nzT: Uint32Array;
}

export interface GnnInput {
  /** Node features, `nNodes × nFeat` row-major. */
  x: Float32Array;
  /** Its transpose, `nFeat × nNodes`. Computed once; X never changes. */
  xT: Float32Array;
  /** Optional; without it the input features are never dropped. */
  pattern?: SparsePattern;
}

export interface GnnShape {
  nNodes: number;
  nFeat: number;
  nClasses: number;
  /** Width of every hidden layer. */
  hidden: number;
  /** Number of weight matrices — 2 is the classic Kipf GCN. Minimum 1. */
  layers: number;
}

export interface GnnHyperparams {
  learningRate: number;
  /** L2 penalty, applied to the weights only (not the biases). */
  weightDecay: number;
  /** Dropout probability on hidden activations. See `forward` for the caveat. */
  dropout: number;
  epochs: number;
  arch: GnnArch;
}

export interface GnnMetrics {
  epoch: number;
  totalEpochs: number;
  /** Cross-entropy over the labelled training nodes. */
  loss: number;
  trainAcc: number;
  valAcc: number;
  testAcc: number;
  /**
   * Mean cosine similarity between **adjacent** nodes' final representations —
   * the number that makes oversmoothing legible. It climbs toward 1 as depth
   * grows, because repeated averaging drives every node's representation toward
   * its neighbours'. Measured between neighbours rather than over all pairs
   * because that is the quantity repeated message passing actually acts on: over
   * all pairs the number is dominated by how well the model separates classes,
   * and on real Cora it does not move monotonically with depth. (It is the same
   * quantity as the Dirichlet energy of the row-normalised representations, which
   * is the literature's measure: `E = 1 − mean neighbour cosine`, up to a factor
   * of two.)
   */
  smoothness: number;
  /**
   * Fraction of nodes whose final representation is entirely zero. This is a
   * different failure from oversmoothing and the page must not conflate them:
   * GIN's unnormalised sum multiplies activations by roughly the mean degree at
   * every layer, so past four or five layers it overflows and ReLU zeroes
   * everything. Smoothness is undefined there — there are no directions left to
   * compare — so this number is what tells the difference between "every node
   * looks the same" and "every node is gone".
   */
  deadFraction: number;
}

export interface GnnSplitIndices {
  train: Uint32Array;
  val: Uint32Array;
  test: Uint32Array;
}

/** The per-architecture scale vectors. `degree` excludes the self-loop. */
export function archScales(
  arch: GnnArch,
  degree: Uint32Array,
): { alpha: Float32Array; beta: Float32Array } {
  const n = degree.length;
  const alpha = new Float32Array(n);
  const beta = new Float32Array(n);

  if (arch === "gin") {
    alpha.fill(1);
    beta.fill(1);
    return { alpha, beta };
  }
  if (arch === "sage") {
    for (let i = 0; i < n; i++) alpha[i] = 1 / (degree[i] + 1);
    beta.fill(1);
    return { alpha, beta };
  }
  for (let i = 0; i < n; i++) {
    const s = 1 / Math.sqrt(degree[i] + 1);
    alpha[i] = s;
    beta[i] = s;
  }
  return { alpha, beta };
}

/**
 * The CPU reference gather — the ground truth the WGSL kernel is checked
 * against, and the implementation the unit tests exercise end to end.
 */
export function makeCpuAggregate(
  rowPtr: Uint32Array,
  colIdx: Uint32Array,
  nNodes: number,
  alphaIn: Float32Array,
  betaIn: Float32Array,
): AggregateFn {
  return async (x, nFeat, transposed) => {
    // Âᵀ is the same gather with the scales swapped — see the file header.
    const alpha = transposed ? betaIn : alphaIn;
    const beta = transposed ? alphaIn : betaIn;
    const out = new Float32Array(nNodes * nFeat);
    for (let i = 0; i < nNodes; i++) {
      const base = i * nFeat;
      const selfScale = beta[i];
      for (let f = 0; f < nFeat; f++) out[base + f] = selfScale * x[base + f];
      for (let e = rowPtr[i]; e < rowPtr[i + 1]; e++) {
        const j = colIdx[e];
        const jb = j * nFeat;
        const bj = beta[j];
        for (let f = 0; f < nFeat; f++) out[base + f] += bj * x[jb + f];
      }
      const ai = alpha[i];
      for (let f = 0; f < nFeat; f++) out[base + f] *= ai;
    }
    return out;
  };
}

/** Reference CPU matmul, for tests and for the aggregate-only code paths. */
export const cpuMatmulFn: MatmulFn = async (a, b, m, k, n) => {
  const c = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const aip = a[i * k + p];
      if (aip === 0) continue;
      const bRow = p * n;
      const cRow = i * n;
      for (let j = 0; j < n; j++) c[cRow + j] += aip * b[bRow + j];
    }
  }
  return c;
};

function transpose(a: Float32Array, rows: number, cols: number): Float32Array {
  const t = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) t[j * rows + i] = a[i * cols + j];
  }
  return t;
}

/**
 * Package a dense feature matrix for training: its transpose, plus the positions
 * of its nonzeros in both layouts. Done once per dataset.
 */
export function prepareInput(
  x: Float32Array,
  rows: number,
  cols: number,
): GnnInput {
  const nzList: number[] = [];
  const nzTList: number[] = [];
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      if (x[i * cols + j] !== 0) {
        nzList.push(i * cols + j);
        nzTList.push(j * rows + i);
      }
    }
  }
  return {
    x,
    xT: transpose(x, rows, cols),
    pattern: { nz: Uint32Array.from(nzList), nzT: Uint32Array.from(nzTList) },
  };
}

/** Adam moments for one parameter tensor. */
interface Moment {
  m: Float32Array;
  v: Float32Array;
}

export interface GnnForward {
  /** Logits, `nNodes × nClasses`. */
  logits: Float32Array;
  /** The last hidden representation, i.e. the input to the final layer. */
  embedding: Float32Array;
  embeddingDim: number;
}

export interface GnnGradients {
  dW: Float32Array[];
  db: Float32Array[];
  loss: number;
}

export class GnnTrainer {
  /** Per layer, `dims[l] × dims[l+1]` row-major. */
  readonly weights: Float32Array[] = [];
  readonly biases: Float32Array[] = [];
  /** Layer widths: `[nFeat, hidden, …, hidden, nClasses]`. */
  readonly dims: number[];

  private readonly rand: () => number;
  private readonly mW: Moment[] = [];
  private readonly mB: Moment[] = [];
  private adamStep = 0;

  // Activations kept from the last forward pass, for the backward pass.
  private acts: Float32Array[] = []; // acts[l] is the input to layer l
  private preAct: Float32Array[] = []; // Z_l, before ReLU
  private masks: (Float32Array | null)[] = [];

  constructor(
    private readonly ops: GnnOps,
    readonly shape: GnnShape,
    seed = 42,
  ) {
    if (shape.layers < 1) throw new Error("a GNN needs at least one layer");
    this.rand = mulberry32(seed);

    this.dims = [shape.nFeat];
    for (let l = 1; l < shape.layers; l++) this.dims.push(shape.hidden);
    this.dims.push(shape.nClasses);

    for (let l = 0; l < shape.layers; l++) {
      const fanIn = this.dims[l];
      const fanOut = this.dims[l + 1];
      // Glorot: keeps the activation variance roughly constant across layers,
      // which matters here because the page deliberately stacks up to 8 of them.
      const scale = Math.sqrt(2 / (fanIn + fanOut));
      const w = new Float32Array(fanIn * fanOut);
      for (let i = 0; i < w.length; i++) w[i] = gaussian(this.rand) * scale;
      this.weights.push(w);
      this.biases.push(new Float32Array(fanOut));
      this.mW.push({ m: new Float32Array(w.length), v: new Float32Array(w.length) });
      this.mB.push({ m: new Float32Array(fanOut), v: new Float32Array(fanOut) });
    }
  }

  /**
   * Full-graph forward pass. Transductive: every node is embedded on every pass,
   * and only the loss is restricted to the labelled ones.
   *
   * Dropout on the **input** features is handled separately from dropout on the
   * hidden activations, and the difference is worth the paragraph. The first
   * layer's weight gradient is `Xᵀ · dS`, so a masked input needs a masked
   * transpose too, and X is 2708×1433 — re-masking and re-transposing 3.9 M
   * elements per epoch would cost more than the model. But X is 1.27 % dense, so
   * `input.pattern` lists the 49 216 positions that can be anything but zero,
   * along with where each one lands in Xᵀ. Masking those two buffers is 50 k
   * writes and a memcpy instead of 8 M multiplies, and the input dropout Kipf's
   * GCN relies on — worth several points on Cora, where 1433 sparse features over
   * 140 labelled nodes overfits immediately — comes back for free.
   */
  async forward(dropout = 0, training = false): Promise<GnnForward> {
    const { nNodes } = this.shape;
    const L = this.shape.layers;
    if (!this.input) throw new Error("call setInput() before forward()");
    this.acts = [];
    this.preAct = [];
    this.masks = [];
    this.droppedInput = this.maskInput(dropout, training);

    let h: Float32Array = this.droppedInput ?? this.input.x;
    for (let l = 0; l < L; l++) {
      let mask: Float32Array | null = null;
      if (training && dropout > 0 && l > 0) {
        // Inverted dropout: scale at train time so inference needs no rescaling.
        const keep = 1 - dropout;
        mask = new Float32Array(h.length);
        const dropped = new Float32Array(h.length);
        for (let i = 0; i < h.length; i++) {
          const on = this.rand() < keep ? 1 / keep : 0;
          mask[i] = on;
          dropped[i] = h[i] * on;
        }
        h = dropped;
      }
      this.acts.push(h);
      this.masks.push(mask);

      const fanIn = this.dims[l];
      const fanOut = this.dims[l + 1];
      // X·W before Â·(·): aggregating an nNodes×fanOut matrix instead of an
      // nNodes×1433 one is the difference between a responsive page and a stall.
      const s = await this.ops.matmul(h, this.weights[l], nNodes, fanIn, fanOut);
      const z = await this.ops.aggregate(s, fanOut, false);

      const bias = this.biases[l];
      for (let i = 0; i < nNodes; i++) {
        const row = i * fanOut;
        for (let f = 0; f < fanOut; f++) z[row + f] += bias[f];
      }
      this.preAct.push(z);

      if (l < L - 1) {
        const relu = new Float32Array(z.length);
        for (let i = 0; i < z.length; i++) relu[i] = z[i] > 0 ? z[i] : 0;
        h = relu;
      } else {
        h = z;
      }
    }

    return {
      logits: h,
      embedding: this.acts[L - 1],
      embeddingDim: this.dims[L - 1],
    };
  }

  /**
   * Gradients of the mean cross-entropy over `trainIdx` only. The unlabelled
   * nodes still take part in the forward pass — that is what makes this
   * semi-supervised — they just contribute no error term.
   */
  async backward(
    labels: Uint8Array,
    trainIdx: Uint32Array,
    forward: GnnForward,
  ): Promise<GnnGradients> {
    const { nNodes, nClasses } = this.shape;
    const L = this.shape.layers;

    const probs = softmaxRows(forward.logits, nNodes, nClasses);
    const loss = crossEntropyLoss(probs, labels, trainIdx, nClasses);
    const dOut = new Float32Array(nNodes * nClasses);
    const scale = 1 / trainIdx.length;
    for (const i of trainIdx) {
      const row = i * nClasses;
      const label = labels[i];
      for (let c = 0; c < nClasses; c++) {
        dOut[row + c] = (probs[row + c] - (c === label ? 1 : 0)) * scale;
      }
    }

    const dW: Float32Array[] = new Array(L);
    const db: Float32Array[] = new Array(L);
    let dH: Float32Array<ArrayBufferLike> = dOut;

    for (let l = L - 1; l >= 0; l--) {
      const fanIn = this.dims[l];
      const fanOut = this.dims[l + 1];

      // ReLU' — the last layer has no activation, so its dZ is dH untouched.
      let dZ = dH;
      if (l < L - 1) {
        const z = this.preAct[l];
        dZ = new Float32Array(dH.length);
        for (let i = 0; i < dH.length; i++) dZ[i] = z[i] > 0 ? dH[i] : 0;
      }

      const bias = new Float32Array(fanOut);
      for (let i = 0; i < nNodes; i++) {
        const row = i * fanOut;
        for (let f = 0; f < fanOut; f++) bias[f] += dZ[row + f];
      }
      db[l] = bias;

      // Âᵀ · dZ — the same gather with the scales swapped.
      const dS = await this.ops.aggregate(dZ, fanOut, true);

      const hIn = this.acts[l];
      const hInT =
        l === 0
          ? (this.droppedInput ? this.droppedInputT! : this.input!.xT)
          : transpose(hIn, nNodes, fanIn);
      dW[l] = await this.ops.matmul(hInT, dS, fanIn, nNodes, fanOut);

      if (l > 0) {
        const wT = transpose(this.weights[l], fanIn, fanOut);
        const dHprev = await this.ops.matmul(dS, wT, nNodes, fanOut, fanIn);
        const mask = this.masks[l];
        if (mask) {
          for (let i = 0; i < dHprev.length; i++) dHprev[i] *= mask[i];
        }
        dH = dHprev;
      }
    }

    return { dW, db, loss };
  }

  /**
   * The input features and their transpose, set once because X never changes.
   * 2708×1433 is 15.5 MB; transposing it per epoch would cost more than the
   * model does, so it is transposed once and reused.
   */
  private input: GnnInput | null = null;
  /** Scratch buffers for the masked input, reused across epochs. */
  private droppedInput: Float32Array | null = null;
  private droppedInputT: Float32Array | null = null;

  setInput(input: GnnInput): void {
    this.input = input;
    this.droppedInput = null;
    this.droppedInputT = null;
  }

  /**
   * Apply input dropout to the stored X and Xᵀ at once, touching only the
   * positions the sparsity pattern says can be nonzero. Returns null when there
   * is nothing to do, in which case the unmasked buffers are used directly.
   */
  private maskInput(dropout: number, training: boolean): Float32Array | null {
    const input = this.input;
    if (!input || !training || dropout <= 0 || !input.pattern) return null;

    const { nz, nzT } = input.pattern;
    const keep = 1 - dropout;
    const size = this.shape.nNodes * this.shape.nFeat;
    // Every position outside the pattern is zero in X and stays zero here, so
    // the scratch buffers only ever need their pattern entries rewritten.
    this.droppedInput ??= new Float32Array(size);
    this.droppedInputT ??= new Float32Array(size);

    for (let e = 0; e < nz.length; e++) {
      const on = this.rand() < keep ? 1 / keep : 0;
      const value = input.x[nz[e]] * on;
      this.droppedInput[nz[e]] = value;
      this.droppedInputT[nzT[e]] = value;
    }
    return this.droppedInput;
  }

  /** Neighbour smoothness of a forward pass's final representation. */
  smoothness(forward: GnnForward): number {
    return this.ops.smoothness(forward.embedding, forward.embeddingDim);
  }

  /** One Adam step, with decoupled L2 on the weights only. */
  applyGradients(grads: GnnGradients, hp: GnnHyperparams): void {
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    this.adamStep++;
    const bc1 = 1 - Math.pow(beta1, this.adamStep);
    const bc2 = 1 - Math.pow(beta2, this.adamStep);

    const step = (
      params: Float32Array,
      grad: Float32Array,
      moment: Moment,
      decay: number,
    ) => {
      for (let i = 0; i < params.length; i++) {
        const g = grad[i] + decay * params[i];
        moment.m[i] = beta1 * moment.m[i] + (1 - beta1) * g;
        moment.v[i] = beta2 * moment.v[i] + (1 - beta2) * g * g;
        const mHat = moment.m[i] / bc1;
        const vHat = moment.v[i] / bc2;
        params[i] -= (hp.learningRate * mHat) / (Math.sqrt(vHat) + eps);
      }
    };

    for (let l = 0; l < this.shape.layers; l++) {
      step(this.weights[l], grads.dW[l], this.mW[l], hp.weightDecay);
      step(this.biases[l], grads.db[l], this.mB[l], 0);
    }
  }
}

/** Row-wise softmax, numerically stabilised. */
export function softmaxRows(
  logits: Float32Array,
  n: number,
  c: number,
): Float32Array {
  const probs = new Float32Array(n * c);
  for (let i = 0; i < n; i++) {
    const row = i * c;
    let max = -Infinity;
    for (let j = 0; j < c; j++) if (logits[row + j] > max) max = logits[row + j];
    let sum = 0;
    for (let j = 0; j < c; j++) {
      const e = Math.exp(logits[row + j] - max);
      probs[row + j] = e;
      sum += e;
    }
    for (let j = 0; j < c; j++) probs[row + j] /= sum;
  }
  return probs;
}

/**
 * Mean cross-entropy of softmax `probs` over the labelled nodes only. Separate
 * from `backward` so a test can measure the loss the gradient claims to be the
 * slope of, without going through the gradient.
 */
export function crossEntropyLoss(
  probs: Float32Array,
  labels: Uint8Array,
  idx: Uint32Array,
  nClasses: number,
): number {
  if (idx.length === 0) return 0;
  let loss = 0;
  for (const i of idx) {
    loss -= Math.log(Math.max(probs[i * nClasses + labels[i]], 1e-12));
  }
  return loss / idx.length;
}

/** Argmax class per row. */
export function predict(
  logits: Float32Array,
  n: number,
  c: number,
): Uint8Array {
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const row = i * c;
    let best = -Infinity;
    let argmax = 0;
    for (let j = 0; j < c; j++) {
      if (logits[row + j] > best) {
        best = logits[row + j];
        argmax = j;
      }
    }
    out[i] = argmax;
  }
  return out;
}

export function accuracyOn(
  pred: Uint8Array,
  labels: Uint8Array,
  idx: Uint32Array,
): number {
  if (idx.length === 0) return 0;
  let correct = 0;
  for (const i of idx) if (pred[i] === labels[i]) correct++;
  return correct / idx.length;
}

/**
 * Mean cosine similarity between the representations of adjacent nodes.
 *
 * Rows are normalised first, so this measures *direction* only: a representation
 * that merely shrinks with depth is not oversmoothed, and conflating the two is
 * the standard trap in measuring this. Dead rows — all-zero after ReLU — have no
 * direction, so edges touching them are skipped rather than counted as zero,
 * which would report a deeply saturated network as perfectly unsmoothed.
 *
 * O(|E|·d): 10 556 edges by 16 features on Cora, i.e. free next to the epoch it
 * reports on. The all-pairs alternative is 3.7 M pairs and would dominate it.
 */
export function makeNeighbourSmoothness(
  rowPtr: Uint32Array,
  colIdx: Uint32Array,
): (h: Float32Array, d: number) => number {
  return (h, d) => {
    const nNodes = rowPtr.length - 1;
    const norms = new Float32Array(nNodes);
    for (let i = 0; i < nNodes; i++) {
      let acc = 0;
      const row = i * d;
      for (let f = 0; f < d; f++) acc += h[row + f] * h[row + f];
      norms[i] = Math.sqrt(acc);
    }

    let sum = 0;
    let pairs = 0;
    for (let i = 0; i < nNodes; i++) {
      if (norms[i] < 1e-12) continue;
      for (let e = rowPtr[i]; e < rowPtr[i + 1]; e++) {
        const j = colIdx[e];
        // Each undirected edge is stored twice; count it once.
        if (j <= i || norms[j] < 1e-12) continue;
        let dot = 0;
        for (let f = 0; f < d; f++) dot += h[i * d + f] * h[j * d + f];
        sum += dot / (norms[i] * norms[j]);
        pairs++;
      }
    }
    return pairs === 0 ? 0 : sum / pairs;
  };
}

/**
 * Fraction of rows that are entirely zero. See `GnnMetrics.deadFraction` for why
 * this is reported separately from smoothness rather than folded into it.
 */
export function deadFraction(h: Float32Array, n: number, d: number): number {
  if (n === 0) return 0;
  let dead = 0;
  for (let i = 0; i < n; i++) {
    const row = i * d;
    let norm = 0;
    for (let f = 0; f < d; f++) norm += h[row + f] * h[row + f];
    if (!(norm > 1e-24)) dead++; // `!(x > y)` also catches NaN
  }
  return dead / n;
}

export interface FitCallbacks {
  onMetrics?: (metrics: GnnMetrics) => void;
  /** Per-epoch snapshot of the predictions, for the canvas. */
  onSnapshot?: (epoch: number, predictions: Uint8Array) => void;
  /** Polled between epochs; return true to stop early. */
  shouldStop?: () => boolean;
}

/**
 * Full-batch training. One forward pass per epoch covers every node, so the
 * train / val / test accuracies and the smoothness number all fall out of the
 * pass the gradient needed anyway — there is no separate evaluation cost.
 */
export async function fitGnn(
  trainer: GnnTrainer,
  input: GnnInput,
  labels: Uint8Array,
  split: GnnSplitIndices,
  hp: GnnHyperparams,
  cb: FitCallbacks = {},
): Promise<GnnMetrics | null> {
  const { nNodes, nClasses } = trainer.shape;
  trainer.setInput(input);

  let last: GnnMetrics | null = null;
  for (let epoch = 0; epoch < hp.epochs; epoch++) {
    if (cb.shouldStop?.()) return last;

    const forward = await trainer.forward(hp.dropout, true);
    const grads = await trainer.backward(labels, split.train, forward);
    trainer.applyGradients(grads, hp);

    // Evaluate with dropout off — the training pass's activations are masked,
    // and reporting accuracy from them would understate the model.
    const evalPass = await trainer.forward(0, false);
    const pred = predict(evalPass.logits, nNodes, nClasses);

    last = {
      epoch,
      totalEpochs: hp.epochs,
      loss: grads.loss,
      trainAcc: accuracyOn(pred, labels, split.train),
      valAcc: accuracyOn(pred, labels, split.val),
      testAcc: accuracyOn(pred, labels, split.test),
      smoothness: trainer.smoothness(evalPass),
      deadFraction: deadFraction(
        evalPass.embedding,
        nNodes,
        evalPass.embeddingDim,
      ),
    };
    cb.onMetrics?.(last);
    cb.onSnapshot?.(epoch, pred);
  }
  return last;
}
