import { describe, expect, it } from "vitest";

import { mulberry32 } from "@/lib/random";

import { AttentionPropagator, NEGATIVE_SLOPE } from "./gat";

// gnn.test.ts already checks GAT's *derivative* against finite differences.
// What it cannot check is the forward pass's defining property: a wrong softmax
// still differentiates consistently, so the gradient check would pass while the
// attention weights were not attention weights at all. That is what is pinned
// here.

function csr(nNodes: number, edges: [number, number][]) {
  const degree = new Uint32Array(nNodes);
  for (const [u, v] of edges) {
    degree[u]++;
    degree[v]++;
  }
  const rowPtr = new Uint32Array(nNodes + 1);
  for (let i = 0; i < nNodes; i++) rowPtr[i + 1] = rowPtr[i] + degree[i];
  const colIdx = new Uint32Array(rowPtr[nNodes]);
  const cursor = rowPtr.slice(0, nNodes);
  for (const [u, v] of edges) {
    colIdx[cursor[u]++] = v;
    colIdx[cursor[v]++] = u;
  }
  return { rowPtr, colIdx };
}

/** Path graph 0–1–2, plus an isolated node 3. */
const N = 4;
const GRAPH = csr(N, [
  [0, 1],
  [1, 2],
]);
const NFEAT = 3;

function propagator(seed = 4) {
  return new AttentionPropagator(
    GRAPH.rowPtr,
    GRAPH.colIdx,
    N,
    NFEAT,
    mulberry32(seed),
  );
}

function features(seed = 9): Float32Array {
  const rand = mulberry32(seed);
  const s = new Float32Array(N * NFEAT);
  for (let i = 0; i < s.length; i++) s[i] = rand() * 4 - 2;
  return s;
}

describe("AttentionPropagator", () => {
  it("owns exactly the two attention vectors, sized by the layer width", () => {
    const params = propagator().parameters();
    expect(params).toHaveLength(2);
    for (const p of params) {
      expect(p.value).toHaveLength(NFEAT);
      expect(p.grad).toHaveLength(NFEAT);
    }
  });

  it("produces a convex combination of each closed neighbourhood", async () => {
    // The defining property: α sums to 1 over N(i) ∪ {i} and every α ≥ 0, so
    // each output feature lies between the smallest and largest value that
    // feature takes over the neighbourhood. A scaled gather does not satisfy
    // this, and neither does a softmax normalised over the wrong set.
    const s = features();
    const out = await propagator().forward(s, NFEAT);

    for (let i = 0; i < N; i++) {
      const neighbourhood = [i];
      for (let e = GRAPH.rowPtr[i]; e < GRAPH.rowPtr[i + 1]; e++) {
        neighbourhood.push(GRAPH.colIdx[e]);
      }
      for (let f = 0; f < NFEAT; f++) {
        const values = neighbourhood.map((j) => s[j * NFEAT + f]);
        expect(out[i * NFEAT + f]).toBeGreaterThanOrEqual(Math.min(...values) - 1e-6);
        expect(out[i * NFEAT + f]).toBeLessThanOrEqual(Math.max(...values) + 1e-6);
      }
    }
  });

  it("attends entirely to itself when a node has no neighbours", async () => {
    // Node 3 is isolated, so its softmax is over one element: α = 1, and the
    // output must be its own features untouched. A missing self-loop would make
    // this NaN (an empty softmax) rather than the identity.
    const s = features();
    const out = await propagator().forward(s, NFEAT);
    for (let f = 0; f < NFEAT; f++) {
      expect(out[3 * NFEAT + f]).toBeCloseTo(s[3 * NFEAT + f], 5);
    }
  });

  it("weights neighbours unequally — otherwise it is just a mean", async () => {
    // Node 1 sees itself and nodes 0 and 2. If attention collapsed to a uniform
    // 1/3 the layer would be GraphSAGE with extra steps, so check the output is
    // not the plain mean of the neighbourhood.
    const s = features();
    const out = await propagator().forward(s, NFEAT);

    let differs = false;
    for (let f = 0; f < NFEAT; f++) {
      const mean =
        (s[0 * NFEAT + f] + s[1 * NFEAT + f] + s[2 * NFEAT + f]) / 3;
      if (Math.abs(out[1 * NFEAT + f] - mean) > 1e-4) differs = true;
    }
    expect(differs).toBe(true);
  });

  it("is uniform when the attention vectors are zero", async () => {
    // With a = 0 every logit is LeakyReLU(0) = 0, so the softmax is uniform and
    // the layer degenerates to an exact mean. This is the calibration for the
    // test above: it shows the inequality there comes from the weights, not
    // from an arithmetic slip.
    const p = propagator();
    for (const param of p.parameters()) param.value.fill(0);

    const s = features();
    const out = await p.forward(s, NFEAT);
    for (let f = 0; f < NFEAT; f++) {
      const mean = (s[0 * NFEAT + f] + s[1 * NFEAT + f] + s[2 * NFEAT + f]) / 3;
      expect(out[1 * NFEAT + f]).toBeCloseTo(mean, 5);
    }
  });

  it("survives logits large enough to overflow an unstabilised softmax", async () => {
    // exp(1000) is Infinity; the max-subtraction is what keeps this finite.
    const p = propagator();
    p.parameters()[0].value.fill(500);
    p.parameters()[1].value.fill(500);

    const out = await p.forward(features(), NFEAT);
    for (const v of out) expect(Number.isFinite(v)).toBe(true);
  });

  it("uses the paper's negative slope below zero", () => {
    expect(NEGATIVE_SLOPE).toBe(0.2);
  });

  it("refuses a width it was not built for", async () => {
    await expect(propagator().forward(new Float32Array(N * 5), 5)).rejects.toThrow(
      /built for 3 features/,
    );
  });

  it("refuses to run backward before forward", async () => {
    await expect(
      propagator().backward(new Float32Array(N * NFEAT), NFEAT),
    ).rejects.toThrow(/call forward\(\) before backward\(\)/);
  });

  it("accumulates into its gradient buffers rather than replacing them", async () => {
    // The trainer zeroes these once per backward pass; a propagator that reset
    // them itself would silently discard a multi-layer network's earlier terms.
    const p = propagator();
    const s = features();
    const dz = features(11);

    await p.forward(s, NFEAT);
    await p.backward(dz, NFEAT);
    const once = Float32Array.from(p.parameters()[0].grad);

    await p.forward(s, NFEAT);
    await p.backward(dz, NFEAT);
    const twice = p.parameters()[0].grad;

    for (let i = 0; i < once.length; i++) {
      expect(twice[i]).toBeCloseTo(2 * once[i], 4);
    }
  });
});
