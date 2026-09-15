import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { decodeCora } from "./cora";
import {
  buildCsr,
  hasEdge,
  sampleNegatives,
  splitEdges,
  undirectedEdges,
  type EdgeList,
} from "./edgeSplit";
import { mulberry32 } from "./random";

// The split is where this route's one real bug lives, and it fails upward: an
// encoder that aggregates over an edge it is later asked to predict scores it
// from memory, and the test number goes *up*. So these run against the real
// committed cora.bin as well as a hand-built graph — asserting that the training
// CSR is a graph, that the held-out edges are genuinely gone from it, and that a
// "negative" is never a real citation.

/** A path 0-1-2-3-4 plus a chord 0-4: every node has degree ≥ 2. */
const ring = () =>
  buildCsr(Uint32Array.from([0, 1, 1, 2, 2, 3, 3, 4, 0, 4]), 5);

const pairsOf = (list: EdgeList): string[] => {
  const out: string[] = [];
  for (let i = 0; i < list.length; i += 2) out.push(`${list[i]}-${list[i + 1]}`);
  return out;
};

describe("buildCsr", () => {
  it("writes each undirected pair in both directions", () => {
    const { rowPtr, colIdx } = buildCsr(Uint32Array.from([0, 2, 1, 2]), 3);
    expect(Array.from(rowPtr)).toEqual([0, 1, 2, 4]);
    expect(Array.from(colIdx)).toEqual([2, 2, 0, 1]);
  });

  it("leaves every row ascending, which is what hasEdge searches", () => {
    // Deliberately out of order: 3 is given neighbours 2 then 0 then 1.
    const { rowPtr, colIdx } = buildCsr(
      Uint32Array.from([2, 3, 0, 3, 1, 3]),
      4,
    );
    const row = Array.from(colIdx.slice(rowPtr[3], rowPtr[4]));
    expect(row).toEqual([0, 1, 2]);
  });

  it("produces an empty graph from no edges", () => {
    const { rowPtr, colIdx } = buildCsr(new Uint32Array(0), 3);
    expect(Array.from(rowPtr)).toEqual([0, 0, 0, 0]);
    expect(colIdx).toHaveLength(0);
  });
});

describe("hasEdge", () => {
  it("finds a neighbour and rejects a non-neighbour", () => {
    const { rowPtr, colIdx } = ring();
    expect(hasEdge(rowPtr, colIdx, 0, 1)).toBe(true);
    expect(hasEdge(rowPtr, colIdx, 0, 4)).toBe(true);
    expect(hasEdge(rowPtr, colIdx, 0, 2)).toBe(false);
    expect(hasEdge(rowPtr, colIdx, 1, 3)).toBe(false);
  });

  it("says no for a node with no neighbours at all", () => {
    const { rowPtr, colIdx } = buildCsr(Uint32Array.from([0, 1]), 3);
    expect(hasEdge(rowPtr, colIdx, 2, 0)).toBe(false);
  });
});

describe("undirectedEdges", () => {
  it("returns each citation once, not twice", () => {
    const { rowPtr, colIdx } = ring();
    const edges = undirectedEdges(rowPtr, colIdx, 5);
    expect(edges.length / 2).toBe(5);
    expect(pairsOf(edges).sort()).toEqual(
      ["0-1", "0-4", "1-2", "2-3", "3-4"].sort(),
    );
  });
});

describe("sampleNegatives", () => {
  it("never returns a real edge, a self-pair or a repeat", () => {
    const { rowPtr, colIdx } = ring();
    const neg = sampleNegatives(rowPtr, colIdx, 5, 5, mulberry32(3));
    const keys = new Set<string>();
    for (let i = 0; i < neg.length; i += 2) {
      const [u, v] = [neg[i], neg[i + 1]];
      expect(u).not.toBe(v);
      expect(hasEdge(rowPtr, colIdx, u, v)).toBe(false);
      keys.add(`${u}-${v}`);
    }
    expect(keys.size).toBe(5);
  });

  it("refuses rather than hangs when the graph has no room", () => {
    // K4: every pair is an edge, so there are no negatives to find.
    const { rowPtr, colIdx } = buildCsr(
      Uint32Array.from([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3]),
      4,
    );
    expect(() => sampleNegatives(rowPtr, colIdx, 4, 1, mulberry32(1))).toThrow(
      /could not sample/i,
    );
  });
});

describe("splitEdges", () => {
  it("rejects fractions that would leave no training graph", () => {
    const { rowPtr, colIdx } = ring();
    expect(() =>
      splitEdges(rowPtr, colIdx, 5, { testFrac: 0.8, valFrac: 0.3 }),
    ).toThrow(/training graph/i);
  });

  it("keeps an edge whose removal would isolate an endpoint", () => {
    // A star: every leaf has degree 1, so nothing at all may be held out.
    const { rowPtr, colIdx } = buildCsr(
      Uint32Array.from([0, 1, 0, 2, 0, 3, 0, 4]),
      5,
    );
    const split = splitEdges(rowPtr, colIdx, 5, { testFrac: 0.5, valFrac: 0 });
    expect(split.testPos).toHaveLength(0);
    expect(split.rescued).toBeGreaterThan(0);
    expect(split.trainPos.length / 2).toBe(4);
  });
});

describe("splitEdges over the real Cora", () => {
  // One decode for the whole block: cora.bin is 161 KB and the parse is the
  // slow part, not the split. Read off disk rather than through loadCora(),
  // which fetches — the same thing cora.test.ts does for the committed asset.
  const file = readFileSync(
    join(dirname(fileURLToPath(import.meta.url)), "data", "cora.bin"),
  );
  const graph = decodeCora(
    file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength),
  );
  const split = splitEdges(graph.rowPtr, graph.colIdx, graph.nNodes, {
    testFrac: 0.1,
    valFrac: 0.05,
    seed: 7,
  });
  const nUndirected = graph.colIdx.length / 2;

  it("holds out about the fraction it was asked for", () => {
    expect(split.testPos.length / 2).toBeGreaterThan(nUndirected * 0.08);
    expect(split.testPos.length / 2).toBeLessThanOrEqual(nUndirected * 0.1);
    expect(split.valPos.length / 2).toBeGreaterThan(nUndirected * 0.03);
  });

  it("accounts for every edge exactly once", () => {
    const total =
      (split.trainPos.length + split.valPos.length + split.testPos.length) / 2;
    expect(total).toBe(nUndirected);
  });

  it("leaves a symmetric training graph with ascending rows", () => {
    const { rowPtr, colIdx } = split;
    expect(rowPtr).toHaveLength(graph.nNodes + 1);
    expect(rowPtr[graph.nNodes]).toBe(colIdx.length);
    for (let u = 0; u < graph.nNodes; u++) {
      expect(rowPtr[u + 1]).toBeGreaterThanOrEqual(rowPtr[u]);
      for (let e = rowPtr[u]; e < rowPtr[u + 1]; e++) {
        const v = colIdx[e];
        expect(v).not.toBe(u); // no self-loop: the kernel adds I itself
        expect(hasEdge(rowPtr, colIdx, v, u)).toBe(true);
        if (e > rowPtr[u]) expect(v).toBeGreaterThan(colIdx[e - 1]);
      }
    }
  });

  it("removes every held-out edge from the graph the model sees", () => {
    // The assertion this whole module exists for. An encoder that can still
    // aggregate over a test edge scores it from memory, and the number goes up.
    for (const held of [split.valPos, split.testPos]) {
      for (let i = 0; i < held.length; i += 2) {
        expect(hasEdge(split.rowPtr, split.colIdx, held[i], held[i + 1])).toBe(
          false,
        );
        expect(hasEdge(split.rowPtr, split.colIdx, held[i + 1], held[i])).toBe(
          false,
        );
        // …and it really was an edge to begin with.
        expect(hasEdge(graph.rowPtr, graph.colIdx, held[i], held[i + 1])).toBe(
          true,
        );
      }
    }
  });

  it("keeps every training edge", () => {
    for (let i = 0; i < split.trainPos.length; i += 2) {
      expect(
        hasEdge(split.rowPtr, split.colIdx, split.trainPos[i], split.trainPos[i + 1]),
      ).toBe(true);
    }
  });

  it("reports the degrees of the training graph, not the original", () => {
    // archScales reads this to build GCN's D^-1/2. A degree left over from the
    // full graph leaks the held-out edge's existence into the normalisation.
    for (let u = 0; u < graph.nNodes; u++) {
      expect(split.degree[u]).toBe(split.rowPtr[u + 1] - split.rowPtr[u]);
    }
  });

  it("isolates nobody", () => {
    for (let u = 0; u < graph.nNodes; u++) {
      if (graph.degree[u] > 0) expect(split.degree[u]).toBeGreaterThan(0);
    }
  });

  it("draws negatives that are not citations in the full graph", () => {
    for (const neg of [split.valNeg, split.testNeg]) {
      for (let i = 0; i < neg.length; i += 2) {
        expect(neg[i]).not.toBe(neg[i + 1]);
        expect(hasEdge(graph.rowPtr, graph.colIdx, neg[i], neg[i + 1])).toBe(
          false,
        );
      }
    }
    expect(split.valNeg.length).toBe(split.valPos.length);
    expect(split.testNeg.length).toBe(split.testPos.length);
  });

  it("is the same split for the same seed, and a different one otherwise", () => {
    const again = splitEdges(graph.rowPtr, graph.colIdx, graph.nNodes, {
      seed: 7,
    });
    expect(Array.from(again.testPos)).toEqual(Array.from(split.testPos));

    const other = splitEdges(graph.rowPtr, graph.colIdx, graph.nNodes, {
      seed: 8,
    });
    expect(Array.from(other.testPos)).not.toEqual(Array.from(split.testPos));
  });
});
