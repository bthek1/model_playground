// A two-layer network, forward and backward by hand — the teaching artefact of
// the policy-gradient pages, so it is written out rather than hidden.
//
//   z1  = x · W1 + b1          (n × hidden)
//   h   = tanh(z1)
//   out = h · W2 + b2          (n × outputs)
//
// One network, three heads, and the head is not the network's business: the
// caller turns `out` into whatever it needs (a softmax policy, a value, an
// action) and hands back **dOut**, the gradient of its loss with respect to
// `out`. So REINFORCE, the critic and behaviour cloning each own the one line
// that differs between them, and this file owns the part they share:
//
//   dW2 = hᵀ · dOut            db2 = Σ_rows dOut
//   dH  = dOut · W2ᵀ
//   dZ1 = dH ⊙ (1 − h²)        ← tanh′, written in terms of the output
//   dW1 = xᵀ · dZ1             db1 = Σ_rows dZ1
//
// **tanh, not ReLU**, which is what the standard CartPole policies use. It also
// has no kink: `/graph`'s gradient check had to be moved to a generic point
// because a central difference across ReLU's corner reports half the gradient.
// The check here still runs at a generic point, because a symmetric one can
// hide a transposed weight.
//
// **The matmul is synchronous, and that is Phase 0's finding** (`limits.ts`).
// `linearModel.ts`'s `MatmulFn` is async because the GPU is; an RL step calls
// the network once per environment step, and at these sizes the round trip to
// the device costs more than the arithmetic by orders of magnitude. So the seam
// is `cpuMatmul` itself — injected, so the tests can check the arithmetic
// against a stub — and nothing here awaits.

import { gaussian, mulberry32 } from "@/lib/random";
import { cpuMatmul, LinearTrainer } from "@/webgpu/linearModel";

/** `cpuMatmul`'s own signature: C(m×n) = A(m×k) · B(k×n), row-major. */
export type SyncMatmul = typeof cpuMatmul;

export interface NetShape {
  inputs: number;
  hidden: number;
  outputs: number;
}

export interface NetParams extends NetShape {
  w1: Float32Array;
  b1: Float32Array;
  w2: Float32Array;
  b2: Float32Array;
}

export interface NetCache {
  x: Float32Array;
  h: Float32Array;
  out: Float32Array;
  n: number;
}

export type NetGrads = Pick<NetParams, "w1" | "b1" | "w2" | "b2">;

/**
 * Glorot-scaled Gaussian init — the variance tanh was analysed with. Biases
 * start at zero; `outScale` shrinks the last layer, the usual trick for a
 * policy head so the first episodes act near-uniformly rather than committing
 * to whatever the random init preferred.
 */
export function initNet(shape: NetShape, seed: number, outScale = 1): NetParams {
  const rand = mulberry32(seed);
  const w1 = new Float32Array(shape.inputs * shape.hidden);
  const w2 = new Float32Array(shape.hidden * shape.outputs);
  const s1 = Math.sqrt(2 / (shape.inputs + shape.hidden));
  const s2 = Math.sqrt(2 / (shape.hidden + shape.outputs)) * outScale;
  for (let i = 0; i < w1.length; i++) w1[i] = gaussian(rand) * s1;
  for (let i = 0; i < w2.length; i++) w2[i] = gaussian(rand) * s2;
  return {
    ...shape,
    w1,
    b1: new Float32Array(shape.hidden),
    w2,
    b2: new Float32Array(shape.outputs),
  };
}

export function forward(
  p: NetParams,
  x: Float32Array,
  n: number,
  matmul: SyncMatmul = cpuMatmul,
): NetCache {
  const h = matmul(x, p.w1, n, p.inputs, p.hidden);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p.hidden; j++) {
      h[i * p.hidden + j] = Math.tanh(h[i * p.hidden + j] + p.b1[j]);
    }
  }
  const out = matmul(h, p.w2, n, p.hidden, p.outputs);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < p.outputs; k++) out[i * p.outputs + k] += p.b2[k];
  }
  return { x, h, out, n };
}

/** Gradients of the caller's loss, given `dOut = ∂loss/∂out` (n × outputs). */
export function backward(
  p: NetParams,
  cache: NetCache,
  dOut: Float32Array,
  matmul: SyncMatmul = cpuMatmul,
): NetGrads {
  const { x, h, n } = cache;
  const { inputs: I, hidden: H, outputs: O } = p;

  const w2 = matmul(transpose(h, n, H), dOut, H, n, O);
  const b2 = new Float32Array(O);
  for (let i = 0; i < n; i++) for (let k = 0; k < O; k++) b2[k] += dOut[i * O + k];

  const dH = matmul(dOut, transpose(p.w2, H, O), n, O, H);
  for (let i = 0; i < n * H; i++) dH[i] *= 1 - h[i] * h[i]; // now dZ1

  const w1 = matmul(transpose(x, n, I), dH, I, n, H);
  const b1 = new Float32Array(H);
  for (let i = 0; i < n; i++) for (let j = 0; j < H; j++) b1[j] += dH[i * H + j];

  return { w1, b1, w2, b2 };
}

/** Row-wise softmax — `LinearTrainer`'s, not a second copy. */
export function softmax(logits: Float32Array, n: number, c: number): Float32Array {
  return LinearTrainer.softmaxRows(logits, n, c);
}

/**
 * Adam. Plain SGD on a policy gradient is famously touchy about the learning
 * rate — one large-return episode moves the weights a long way — and the page
 * is about the *estimator's* variance, not the optimiser's.
 */
export class Adam {
  private readonly m: NetGrads;
  private readonly v: NetGrads;
  private t = 0;

  constructor(
    private readonly params: NetParams,
    public lr: number,
    private readonly beta1 = 0.9,
    private readonly beta2 = 0.999,
    private readonly eps = 1e-8,
  ) {
    const zeros = (a: Float32Array) => new Float32Array(a.length);
    this.m = { w1: zeros(params.w1), b1: zeros(params.b1), w2: zeros(params.w2), b2: zeros(params.b2) };
    this.v = { w1: zeros(params.w1), b1: zeros(params.b1), w2: zeros(params.w2), b2: zeros(params.b2) };
  }

  /** Descend: `params -= lr · m̂ / (√v̂ + ε)`. Pass the gradient of a loss to minimise. */
  step(g: NetGrads): void {
    this.t++;
    const c1 = 1 - this.beta1 ** this.t;
    const c2 = 1 - this.beta2 ** this.t;
    for (const key of ["w1", "b1", "w2", "b2"] as const) {
      const p = this.params[key];
      const grad = g[key];
      const m = this.m[key];
      const v = this.v[key];
      for (let i = 0; i < p.length; i++) {
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * grad[i];
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * grad[i] * grad[i];
        p[i] -= (this.lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + this.eps);
      }
    }
  }
}

export function transpose(a: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) out[j * rows + i] = a[i * cols + j];
  }
  return out;
}
