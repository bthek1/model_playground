import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ProteinRow } from "@/lib/proteins";

/** Six graphs, three of each label, so a stratified split has something to do. */
function rows(): ProteinRow[] {
  const triangle = (y: number): ProteinRow => ({
    edge_index: [
      [0, 1, 1, 2, 2, 0],
      [1, 0, 2, 1, 0, 2],
    ],
    node_feat: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
    y: [y],
    num_nodes: 3,
  });
  const pair = (y: number): ProteinRow => ({
    edge_index: [
      [0, 1],
      [1, 0],
    ],
    node_feat: [
      [1, 1, 0],
      [0, 1, 1],
    ],
    y: [y],
    num_nodes: 2,
  });
  return [triangle(1), pair(0), triangle(1), pair(0), triangle(1), pair(0)];
}

const fetchProteins = vi.fn(async () => rows());
vi.mock("@/lib/proteins", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, fetchProteins: () => fetchProteins() };
});

const loadCachedUnion = vi.fn(async () => null);
const saveCachedUnion = vi.fn(async (_union: unknown) => {});
vi.mock("@/lib/proteinsCache", () => ({
  loadCachedUnion: () => loadCachedUnion(),
  saveCachedUnion: (u: unknown) => saveCachedUnion(u),
  clearCachedUnion: async () => {},
}));

// No GPU in happy-dom, and none needed: the CPU aggregation is the reference the
// shader is checked against, so the whole training path runs here.
vi.mock("./capabilities", () => ({
  detectWebGPU: vi.fn(async () => ({ status: "no-adapter" })),
}));

const { ProteinSession } = await import("./proteinSession");

const REQUEST = {
  arch: "gcn" as const,
  readout: "mean" as const,
  layers: 2,
  hidden: 4,
  learningRate: 0.05,
  weightDecay: 0,
  dropout: 0,
  epochs: 4,
  seed: 7,
};

const SPLIT = { trainFrac: 0.5, valFrac: 0.2 };

describe("ProteinSession", () => {
  beforeEach(() => {
    fetchProteins.mockClear();
    loadCachedUnion.mockClear();
    saveCachedUnion.mockClear();
  });

  it("reports the dataset and its split, and never the feature matrix", async () => {
    const summary = await new ProteinSession(SPLIT).load();

    expect(summary.nGraphs).toBe(6);
    expect(summary.nNodes).toBe(15);
    expect(summary.nTrain + summary.nVal + summary.nTest).toBe(6);
    expect(summary).not.toHaveProperty("features");
    // The page reads the accuracy against this, so it travels with the summary.
    expect(summary.baselineAcc).toBeGreaterThanOrEqual(0);
    expect(summary.baselineAcc).toBeLessThanOrEqual(1);
  });

  it("downloads once, however often it is loaded", async () => {
    const session = new ProteinSession(SPLIT);
    const first = await session.load();
    const second = await session.load();

    expect(fetchProteins).toHaveBeenCalledTimes(1);
    expect(second.nGraphs).toBe(first.nGraphs);
    // Fresh copies each time, because these are transferred to the page.
    expect(second.labels).not.toBe(first.labels);
  });

  it("prefers the cache to the network, and says which it used", async () => {
    const network = new ProteinSession(SPLIT);
    const fresh = await network.load();
    expect(fresh.fromCache).toBe(false);
    expect(saveCachedUnion).toHaveBeenCalledTimes(1);

    const stored = saveCachedUnion.mock.calls[0]![0];
    loadCachedUnion.mockResolvedValueOnce(stored as never);
    fetchProteins.mockClear();

    const cached = await new ProteinSession(SPLIT).load();
    expect(cached.fromCache).toBe(true);
    expect(fetchProteins).not.toHaveBeenCalled();
  });

  it("still loads when the cache cannot be written", async () => {
    // A quota failure must not fail the load — the page just re-downloads next
    // time, which is slower and correct.
    saveCachedUnion.mockRejectedValueOnce(new Error("QuotaExceededError"));
    await expect(new ProteinSession(SPLIT).load()).resolves.toMatchObject({
      nGraphs: 6,
    });
  });

  it("refuses to train or lay out before it has data", async () => {
    const session = new ProteinSession(SPLIT);
    await expect(session.train(REQUEST)).rejects.toThrow(/load/i);
    expect(() => session.layoutFor(0)).toThrow(/load/i);
  });

  it("trains, and carries the baseline through to the metrics", async () => {
    const session = new ProteinSession(SPLIT);
    const summary = await session.load();
    const seen: number[] = [];
    const result = await session.train(REQUEST, (m) => seen.push(m.epoch));

    expect(seen).toEqual([0, 1, 2, 3]);
    expect(result.metrics?.epoch).toBe(3);
    expect(result.metrics?.baselineAcc).toBe(summary.baselineAcc);
    expect(result.predicted).toHaveLength(6);
    expect(result.backend).toBe("cpu");
  });

  it("returns null metrics when stopped before its first epoch", async () => {
    const session = new ProteinSession(SPLIT);
    await session.load();
    const result = await session.train(REQUEST, undefined, () => true);
    expect(result.metrics).toBeNull();
    expect(result.predicted).toHaveLength(6);
  });

  it("lays a graph out in its own coordinates, not the union's", async () => {
    // A tile that drew the union's global indices would point at whatever node
    // happened to sit at that offset inside the first graph.
    const session = new ProteinSession(SPLIT);
    await session.load();
    const second = session.layoutFor(1); // the first `pair`, nodes 3 and 4

    expect(second.index).toBe(1);
    expect(second.nNodes).toBe(2);
    expect(Array.from(second.colIdx).every((v) => v < 2)).toBe(true);
    expect(second.x).toHaveLength(2);
    for (const v of [...second.x, ...second.y]) {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("lays each graph out once and remembers it", async () => {
    const session = new ProteinSession(SPLIT);
    await session.load();
    expect(session.layoutFor(2)).toBe(session.layoutFor(2));
  });

  it("refuses a graph index the dataset does not have", async () => {
    const session = new ProteinSession(SPLIT);
    await session.load();
    expect(() => session.layoutFor(99)).toThrow(/no graph 99/i);
  });
});
