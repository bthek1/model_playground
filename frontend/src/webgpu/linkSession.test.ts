import { beforeEach, describe, expect, it, vi } from "vitest";

import type { CoraGraph } from "@/lib/cora";
import { hasEdge } from "@/lib/edgeSplit";

/**
 * A stand-in for Cora big enough to hold a split: two dense-ish clusters joined
 * by a bridge, so every node keeps a neighbour when 10 % of the edges go.
 */
const N = 10;
const EDGES: [number, number][] = [
  [0, 1],
  [0, 2],
  [0, 3],
  [1, 2],
  [1, 3],
  [2, 3],
  [5, 6],
  [5, 7],
  [5, 8],
  [6, 7],
  [6, 8],
  [7, 8],
  [3, 5],
  [4, 0],
  [4, 1],
  [9, 6],
  [9, 7],
];

function toyGraph(): CoraGraph {
  const degree = new Uint32Array(N);
  for (const [u, v] of EDGES) {
    degree[u]++;
    degree[v]++;
  }
  const rowPtr = new Uint32Array(N + 1);
  for (let i = 0; i < N; i++) rowPtr[i + 1] = rowPtr[i] + degree[i];
  const colIdx = new Uint32Array(rowPtr[N]);
  const cursor = rowPtr.slice(0, N);
  for (const [u, v] of EDGES) {
    colIdx[cursor[u]++] = v;
    colIdx[cursor[v]++] = u;
  }
  for (let u = 0; u < N; u++) colIdx.subarray(rowPtr[u], rowPtr[u + 1]).sort();

  const nFeat = 4;
  const features = new Float32Array(N * nFeat);
  for (let i = 0; i < N; i++) features[i * nFeat + (i % nFeat)] = 1;

  return {
    nNodes: N,
    nFeat,
    nClasses: 2,
    rowPtr,
    colIdx,
    features,
    labels: Uint8Array.from([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
    degree,
  };
}

const loadCora = vi.fn(async () => toyGraph());
vi.mock("@/lib/cora", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, loadCora: () => loadCora() };
});

// No GPU in happy-dom, and none needed: the CPU aggregation is the reference
// implementation the shader is checked against, so the whole path runs here.
vi.mock("./capabilities", () => ({
  detectWebGPU: vi.fn(async () => ({ status: "no-adapter" })),
}));

const { LinkSession, scorePair, topCandidates } = await import("./linkSession");

const REQUEST = {
  arch: "gcn" as const,
  layers: 2,
  hidden: 4,
  embedding: 3,
  learningRate: 0.05,
  weightDecay: 0,
  dropout: 0,
  epochs: 4,
  topK: 5,
  seed: 7,
};

describe("LinkSession", () => {
  beforeEach(() => loadCora.mockClear());

  it("reports the split, and never the feature matrix", async () => {
    const summary = await new LinkSession().load();

    expect(summary.nNodes).toBe(N);
    expect(summary.nTrainEdges + summary.nValEdges + summary.nTestEdges).toBe(
      EDGES.length,
    );
    expect(summary.nEdges).toBe(summary.nTrainEdges * 2);
    expect(summary.x).toHaveLength(N);
    expect(summary).not.toHaveProperty("features");
  });

  it("hands the model a graph without the held-out citations", async () => {
    // The assertion the route exists to protect. The summary's CSR is the one
    // the encoder aggregates over; every edge it is later scored on must be
    // missing from it.
    const session = new LinkSession();
    const summary = await session.load({ testFrac: 0.2, valFrac: 0.1 });
    const full = toyGraph();

    let missing = 0;
    for (const [u, v] of EDGES) {
      if (!hasEdge(summary.rowPtr, summary.colIdx, u, v)) missing++;
      expect(hasEdge(full.rowPtr, full.colIdx, u, v)).toBe(true);
    }
    expect(missing).toBe(summary.nValEdges + summary.nTestEdges);
    expect(missing).toBeGreaterThan(0);
  });

  it("splits and lays out once for the same held-out fraction", async () => {
    const session = new LinkSession();
    const first = await session.load();
    const second = await session.load();

    expect(loadCora).toHaveBeenCalledTimes(1);
    expect([...second.x]).toEqual([...first.x]);
    // A fresh copy each time, because these are transferred to the page.
    expect(second.x).not.toBe(first.x);
  });

  it("re-splits when the held-out fraction changes", async () => {
    // A different fraction is a different graph, so the layout has to move with
    // it — this is the one control on the page that legitimately costs a reload.
    const session = new LinkSession();
    const a = await session.load({ testFrac: 0.1, valFrac: 0.05 });
    const b = await session.load({ testFrac: 0.3, valFrac: 0.05 });
    expect(b.nTestEdges).toBeGreaterThan(a.nTestEdges);
    expect(b.nTrainEdges).toBeLessThan(a.nTrainEdges);
  });

  it("refuses to train before it has a graph", async () => {
    await expect(new LinkSession().train(REQUEST)).rejects.toThrow(/load/i);
  });

  it("trains, and reports an AUC for each split", async () => {
    const session = new LinkSession();
    await session.load();
    const seen: number[] = [];
    const result = await session.train(REQUEST, (m) => seen.push(m.epoch));

    expect(seen).toEqual([0, 1, 2, 3]);
    expect(result.metrics?.epoch).toBe(3);
    for (const value of [
      result.metrics!.trainAuc,
      result.metrics!.valAuc,
      result.metrics!.testAuc,
    ]) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    }
    expect(result.backend).toBe("cpu");
  });

  it("returns candidates that are not already citations", async () => {
    const session = new LinkSession();
    await session.load();
    const result = await session.train(REQUEST);
    const full = toyGraph();

    expect(result.candidates.length / 2).toBe(REQUEST.topK);
    for (let i = 0; i < result.candidates.length; i += 2) {
      const [u, v] = [result.candidates[i], result.candidates[i + 1]];
      expect(u).not.toBe(v);
      expect(hasEdge(full.rowPtr, full.colIdx, u, v)).toBe(false);
    }
    // Best first — the page draws the list in order and says so.
    for (let i = 1; i < result.candidateScores.length; i++) {
      expect(result.candidateScores[i]).toBeLessThanOrEqual(
        result.candidateScores[i - 1],
      );
    }
  });

  it("hands back embeddings the page can score a pair from", async () => {
    const session = new LinkSession();
    await session.load();
    const result = await session.train(REQUEST);

    expect(result.embeddingDim).toBe(REQUEST.embedding);
    expect(result.embedding).toHaveLength(N * REQUEST.embedding);
    // The score the page computes for a candidate must be the one the worker
    // ranked it by, or clicking the two ends of a drawn line disagrees with the
    // line.
    const [u, v] = [result.candidates[0], result.candidates[1]];
    expect(scorePair(result.embedding, result.embeddingDim, u, v)).toBeCloseTo(
      result.candidateScores[0],
      4,
    );
  });

  it("returns null metrics when stopped before its first epoch", async () => {
    const session = new LinkSession();
    await session.load();
    const result = await session.train(REQUEST, undefined, () => true);
    // Cancelling is a normal outcome, not a failure: throwing here would put
    // "training produced no metrics" in the error slot on every quick Stop.
    expect(result.metrics).toBeNull();
    expect(result.embedding).toHaveLength(N * REQUEST.embedding);
  });
});

describe("topCandidates", () => {
  // 4 nodes, dim 1: scores are just the products, and node 3 is the outlier.
  const z = Float32Array.from([1, 2, 3, -1]);
  const { rowPtr, colIdx } = {
    rowPtr: Uint32Array.from([0, 1, 2, 2, 2]),
    colIdx: Uint32Array.from([1, 0]),
  };

  it("ranks the non-edges best first", () => {
    const { pairs, scores } = topCandidates(z, 1, 4, rowPtr, colIdx, 3);
    // 0-1 is an edge and is skipped. 1-2 = 6, 0-2 = 3, then the negatives.
    expect(Array.from(pairs.slice(0, 4))).toEqual([1, 2, 0, 2]);
    expect(scores[0]).toBeCloseTo(6, 6);
    expect(scores[1]).toBeCloseTo(3, 6);
  });

  it("skips pairs that are already edges", () => {
    const { pairs } = topCandidates(z, 1, 4, rowPtr, colIdx, 10);
    for (let i = 0; i < pairs.length; i += 2) {
      expect(hasEdge(rowPtr, colIdx, pairs[i], pairs[i + 1])).toBe(false);
    }
    // 6 pairs in K4, minus the one real edge.
    expect(pairs.length / 2).toBe(5);
  });

  it("asks for nothing and gets nothing", () => {
    const { pairs, scores } = topCandidates(z, 1, 4, rowPtr, colIdx, 0);
    expect(pairs).toHaveLength(0);
    expect(scores).toHaveLength(0);
  });

  it("returns fewer than k when the graph has fewer non-edges", () => {
    // K3: every pair is an edge, so there is nothing to predict.
    const full = {
      rowPtr: Uint32Array.from([0, 2, 4, 6]),
      colIdx: Uint32Array.from([1, 2, 0, 2, 0, 1]),
    };
    const { pairs } = topCandidates(
      Float32Array.from([1, 1, 1]),
      1,
      3,
      full.rowPtr,
      full.colIdx,
      5,
    );
    expect(pairs).toHaveLength(0);
  });
});
