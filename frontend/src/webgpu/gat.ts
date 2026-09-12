// Graph Attention, and the one architecture in the playground that is not a
// scaled gather.
//
// The roadmap's premise for this category — "they differ only in the aggregation
// step, which is one shader each" — holds for GCN, GraphSAGE and GIN, whose
// coefficients are fixed functions of the degrees and so reduce to two scale
// vectors uploaded once. It does not hold here. GAT's coefficient on an edge is
// **computed from the features being propagated**:
//
//     e_ij = LeakyReLU(a_srcᵀ s_i + a_dstᵀ s_j)
//     α_ij = softmax over j ∈ N(i) ∪ {i}
//     out_i = Σ_j α_ij s_j
//
// which has three consequences the other three do not have: the layer owns two
// learnable vectors, the coefficients are per **edge** rather than a product of
// two per-node scales, and the backward pass flows into `s` twice — once through
// the values and once through the softmax that weighted them.
//
// **It is therefore not dispatched through `gnn_aggregate.wgsl`, and that is a
// decision rather than an omission.** The gather is O(|E|·d) — 10 556 edges by
// 16 features on Cora, about 0.2 ms — while the projection sitting next to it is
// 2708x1433x16, and *that* is what runs on the GPU through the shared matmul. A
// second shader here would move a fifth of a millisecond and would have to carry
// a per-edge softmax and its backward into WGSL to do it. The page says which
// half runs where rather than implying the whole model moved to the CPU.
//
// The backward pass needs no reverse-edge index, which is the thing that usually
// makes attention awkward on a CSR graph: every term that would need one is an
// **accumulation** into a dense per-node buffer, and the loop already visits
// every directed edge exactly once.

import { gaussian } from "@/lib/random";

import type { Parameter, Propagator } from "./gnn";

/** The slope LeakyReLU uses below zero. 0.2 is the GAT paper's value. */
export const NEGATIVE_SLOPE = 0.2;

interface Stash {
  s: Float32Array;
  nFeat: number;
  /** α_ii, the self-loop's attention weight. */
  alphaSelf: Float32Array;
  /** α_ij for the edge stored at slot e. */
  alphaEdge: Float32Array;
  /** LeakyReLU's derivative at each logit: 1 above zero, NEGATIVE_SLOPE below. */
  slopeSelf: Float32Array;
  slopeEdge: Float32Array;
}

/**
 * A single-head attention propagator over a fixed CSR graph.
 *
 * Single head, and the page says so: multi-head is a concatenation of h
 * independent copies of exactly this, which multiplies the code by nothing
 * interesting and the width by h.
 */
export class AttentionPropagator implements Propagator {
  private readonly aSrc: Parameter;
  private readonly aDst: Parameter;
  private stash: Stash | null = null;

  constructor(
    private readonly rowPtr: Uint32Array,
    private readonly colIdx: Uint32Array,
    private readonly nNodes: number,
    private readonly nFeat: number,
    rand: () => number,
  ) {
    // Glorot for a fan-in of nFeat and a fan-out of 1: the logit is a dot
    // product, so its variance is what needs controlling.
    const scale = Math.sqrt(2 / (nFeat + 1));
    const init = () => {
      const v = new Float32Array(nFeat);
      for (let i = 0; i < nFeat; i++) v[i] = gaussian(rand) * scale;
      return { value: v, grad: new Float32Array(nFeat) };
    };
    this.aSrc = init();
    this.aDst = init();
  }

  parameters(): Parameter[] {
    return [this.aSrc, this.aDst];
  }

  async forward(s: Float32Array, nFeat: number): Promise<Float32Array> {
    if (nFeat !== this.nFeat) {
      throw new Error(
        `attention was built for ${this.nFeat} features, got ${nFeat}`,
      );
    }
    const { rowPtr, colIdx, nNodes } = this;
    const src = project(s, nNodes, nFeat, this.aSrc.value);
    const dst = project(s, nNodes, nFeat, this.aDst.value);

    const alphaSelf = new Float32Array(nNodes);
    const alphaEdge = new Float32Array(colIdx.length);
    const slopeSelf = new Float32Array(nNodes);
    const slopeEdge = new Float32Array(colIdx.length);
    const out = new Float32Array(nNodes * nFeat);

    for (let i = 0; i < nNodes; i++) {
      const start = rowPtr[i];
      const end = rowPtr[i + 1];

      // Logits, with the self-loop's kept separate from the stored edges — the
      // CSR holds no self-loop, exactly as the scaled gather assumes.
      const rawSelf = src[i] + dst[i];
      slopeSelf[i] = rawSelf > 0 ? 1 : NEGATIVE_SLOPE;
      let max = rawSelf * slopeSelf[i];
      for (let e = start; e < end; e++) {
        const raw = src[i] + dst[colIdx[e]];
        const slope = raw > 0 ? 1 : NEGATIVE_SLOPE;
        slopeEdge[e] = slope;
        const value = raw * slope;
        alphaEdge[e] = value; // holds the logit until it is normalised below
        if (value > max) max = value;
      }
      const logitSelf = rawSelf * slopeSelf[i];

      // Softmax over the closed neighbourhood, stabilised by the max.
      let sum = Math.exp(logitSelf - max);
      alphaSelf[i] = sum;
      for (let e = start; e < end; e++) {
        const value = Math.exp(alphaEdge[e] - max);
        alphaEdge[e] = value;
        sum += value;
      }
      alphaSelf[i] /= sum;
      for (let e = start; e < end; e++) alphaEdge[e] /= sum;

      const base = i * nFeat;
      const aSelf = alphaSelf[i];
      for (let f = 0; f < nFeat; f++) out[base + f] = aSelf * s[base + f];
      for (let e = start; e < end; e++) {
        const jb = colIdx[e] * nFeat;
        const a = alphaEdge[e];
        for (let f = 0; f < nFeat; f++) out[base + f] += a * s[jb + f];
      }
    }

    this.stash = { s, nFeat, alphaSelf, alphaEdge, slopeSelf, slopeEdge };
    return out;
  }

  async backward(dz: Float32Array, nFeat: number): Promise<Float32Array> {
    const stash = this.stash;
    if (!stash) throw new Error("call forward() before backward()");
    if (nFeat !== stash.nFeat) {
      throw new Error("backward was given a different width than forward");
    }
    const { rowPtr, colIdx, nNodes } = this;
    const { s, alphaSelf, alphaEdge, slopeSelf, slopeEdge } = stash;

    const ds = new Float32Array(nNodes * nFeat);
    // Gradients w.r.t. the two scalar projections, accumulated per node. Every
    // scatter below lands in one of these, which is why no reverse-edge index is
    // needed: the loop already visits each directed edge exactly once.
    const dSrc = new Float32Array(nNodes);
    const dDst = new Float32Array(nNodes);

    for (let i = 0; i < nNodes; i++) {
      const start = rowPtr[i];
      const end = rowPtr[i + 1];
      const base = i * nFeat;

      // ∂L/∂α for each neighbour, and the value path back into s.
      let dotSelf = 0;
      for (let f = 0; f < nFeat; f++) {
        dotSelf += dz[base + f] * s[base + f];
        ds[base + f] += alphaSelf[i] * dz[base + f];
      }
      let weighted = alphaSelf[i] * dotSelf;

      for (let e = start; e < end; e++) {
        const jb = colIdx[e] * nFeat;
        let dot = 0;
        const a = alphaEdge[e];
        for (let f = 0; f < nFeat; f++) {
          dot += dz[base + f] * s[jb + f];
          ds[jb + f] += a * dz[base + f];
        }
        // Reuse the buffer: from here alphaEdge[e] is still needed, so the
        // per-edge ∂L/∂α is kept on the stack via a second pass below.
        weighted += a * dot;
      }

      // Softmax backward: dLogit_k = α_k (dα_k − Σ_m α_m dα_m).
      const dLogitSelf = alphaSelf[i] * (dotSelf - weighted);
      const dRawSelf = dLogitSelf * slopeSelf[i];
      dSrc[i] += dRawSelf;
      dDst[i] += dRawSelf;

      for (let e = start; e < end; e++) {
        const j = colIdx[e];
        const jb = j * nFeat;
        let dot = 0;
        for (let f = 0; f < nFeat; f++) dot += dz[base + f] * s[jb + f];
        const dRaw = alphaEdge[e] * (dot - weighted) * slopeEdge[e];
        dSrc[i] += dRaw;
        dDst[j] += dRaw;
      }
    }

    // The logits' own path into s and into the attention vectors.
    const gSrc = this.aSrc.grad;
    const gDst = this.aDst.grad;
    const vSrc = this.aSrc.value;
    const vDst = this.aDst.value;
    for (let i = 0; i < nNodes; i++) {
      const base = i * nFeat;
      const a = dSrc[i];
      const b = dDst[i];
      for (let f = 0; f < nFeat; f++) {
        gSrc[f] += a * s[base + f];
        gDst[f] += b * s[base + f];
        ds[base + f] += a * vSrc[f] + b * vDst[f];
      }
    }

    return ds;
  }
}

/** Row-wise dot product of an `n × d` matrix with a `d`-vector. */
function project(
  s: Float32Array,
  n: number,
  d: number,
  a: Float32Array,
): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    const base = i * d;
    for (let f = 0; f < d; f++) acc += s[base + f] * a[f];
    out[i] = acc;
  }
  return out;
}
