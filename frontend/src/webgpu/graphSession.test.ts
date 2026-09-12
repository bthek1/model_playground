import { beforeEach, describe, expect, it, vi } from "vitest";

import type { CoraGraph } from "@/lib/cora";

/**
 * A tiny stand-in for Cora: two triangles joined by a bridge, three features,
 * two classes. Enough for every code path, small enough to train in a test.
 */
const N = 6;
const EDGES: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 0],
  [3, 4],
  [4, 5],
  [5, 3],
  [2, 3],
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

  const nFeat = 3;
  const features = new Float32Array(N * nFeat);
  for (let i = 0; i < N; i++) features[i * nFeat + (i % nFeat)] = 1;

  return {
    nNodes: N,
    nFeat,
    nClasses: 2,
    rowPtr,
    colIdx,
    features,
    labels: Uint8Array.from([0, 0, 0, 1, 1, 1]),
    degree,
  };
}

const loadCora = vi.fn(async () => toyGraph());
vi.mock("@/lib/cora", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, loadCora: () => loadCora() };
});

// No GPU in happy-dom, and none needed: the CPU aggregation is the reference
// implementation, so the whole training path is exercised without a device.
vi.mock("./capabilities", () => ({
  detectWebGPU: vi.fn(async () => ({ status: "no-adapter" })),
}));

const { GraphSession } = await import("./graphSession");

const REQUEST = {
  arch: "gcn" as const,
  layers: 2,
  hidden: 4,
  learningRate: 0.05,
  weightDecay: 0,
  dropout: 0,
  epochs: 5,
  seed: 7,
};

/** The toy graph has 3 nodes per class, so Cora's 20/500/1000 will not fit. */
const SPLIT = { perClass: 1, valSize: 1, testSize: 2 };
const makeSession = () => new GraphSession(SPLIT);

describe("GraphSession", () => {
  beforeEach(() => {
    loadCora.mockClear();
  });

  it("reports only what the page draws, and never the feature matrix", async () => {
    const summary = await makeSession().load();

    expect(summary.nNodes).toBe(N);
    expect(summary.nEdges).toBe(14); // 7 citations, stored in both directions
    expect(summary.x).toHaveLength(N);
    expect(summary.y).toHaveLength(N);
    expect(summary.layoutMs).toBeGreaterThanOrEqual(0);
    // 15.5 MB on the real dataset; the page has no use for it.
    expect(summary).not.toHaveProperty("features");
  });

  it("lays the graph out once, however often it is loaded", async () => {
    const session = makeSession();
    const first = await session.load();
    const second = await session.load();

    expect(loadCora).toHaveBeenCalledTimes(1);
    // Same coordinates, and a fresh copy each time so the page cannot mutate
    // the session's own arrays (they are transferred to it).
    expect([...second.x]).toEqual([...first.x]);
    expect(second.x).not.toBe(first.x);
  });

  it("falls back to the CPU when no GPU device is offered", async () => {
    const session = makeSession();
    expect((await session.load()).backend).toBe("cpu");
    expect((await session.train(REQUEST)).backend).toBe("cpu");
  });

  it("refuses to train before the graph is loaded", async () => {
    await expect(makeSession().train(REQUEST)).rejects.toThrow(
      /load the graph before training/,
    );
  });

  it("streams one epoch at a time, and returns the last of them", async () => {
    const session = makeSession();
    await session.load();

    const seen: number[] = [];
    const result = await session.train(REQUEST, (m, predictions) => {
      seen.push(m.epoch);
      expect(predictions).toHaveLength(N);
    });

    expect(seen).toEqual([0, 1, 2, 3, 4]);
    expect(result.metrics?.epoch).toBe(4);
    expect(result.predictions).toHaveLength(N);
  });

  it("treats an immediate stop as a normal outcome, not a failure", async () => {
    // Pressing Stop before the first epoch finishes used to throw "training
    // produced no metrics", which put a cancel in the error slot.
    const session = makeSession();
    await session.load();

    const result = await session.train(REQUEST, undefined, () => true);
    expect(result.metrics).toBeNull();
    expect(result.backend).toBe("cpu");
  });

  it("runs GAT, which does not go through the aggregation kernel", async () => {
    const session = makeSession();
    await session.load();
    const result = await session.train({ ...REQUEST, arch: "gat" });
    expect(result.metrics?.epoch).toBe(4);
    expect(Number.isFinite(result.metrics?.loss)).toBe(true);
  });
});
