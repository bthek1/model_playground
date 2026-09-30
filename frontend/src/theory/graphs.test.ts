import { describe, expect, it } from "vitest";

import { DEFAULT_GRAPH, SMALL_GRAPHS, graphById, neighbours, toCsr } from "./graphs";
import { adjacencyPower } from "./walks";

describe("SMALL_GRAPHS", () => {
  it("has unique ids and a default that is in the catalogue", () => {
    const ids = SMALL_GRAPHS.map((g) => g.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(graphById(DEFAULT_GRAPH.id)).toBe(DEFAULT_GRAPH);
  });

  it.each(SMALL_GRAPHS.map((g) => [g.id, g] as const))(
    "%s is simple: no self-loops, no repeated edges, every endpoint a node",
    (_, g) => {
      const seen = new Set<string>();
      for (const [u, v] of g.edges) {
        expect(u).toBeLessThan(v); // stored once, and never a loop
        expect(v).toBeLessThan(g.n);
        expect(u).toBeGreaterThanOrEqual(0);
        const key = `${u}-${v}`;
        expect(seen.has(key)).toBe(false);
        seen.add(key);
      }
    },
  );

  // The page draws every node and steps BFS across the whole graph; a
  // disconnected entry would leave nodes BFS can never reach, which the page
  // does not claim to explain.
  it.each(SMALL_GRAPHS.map((g) => [g.id, g] as const))("%s is connected", (_, g) => {
    const nbrs = neighbours(g);
    const seen = new Set([0]);
    const stack = [0];
    while (stack.length) {
      for (const v of nbrs[stack.pop()!]) {
        if (seen.has(v)) continue;
        seen.add(v);
        stack.push(v);
      }
    }
    expect(seen.size).toBe(g.n);
  });

  // Checked against the definition that does not go through BFS: a graph is
  // bipartite iff it has no closed walk of odd length, i.e. trace(Aᵏ) = 0 for
  // every odd k. An odd cycle, if there is one, has length at most n.
  it.each(SMALL_GRAPHS.map((g) => [g.id, g] as const))(
    "%s declares bipartite correctly (no odd closed walk up to length n)",
    (_, g) => {
      let oddClosedWalk = false;
      for (let k = 1; k <= g.n && !oddClosedWalk; k += 2) {
        const p = adjacencyPower(g, k, false);
        for (let i = 0; i < g.n; i++) if (p[i * g.n + i] > 0) oddClosedWalk = true;
      }
      expect(g.bipartite).toBe(!oddClosedWalk);
    },
  );

  it.each(SMALL_GRAPHS.map((g) => [g.id, g] as const))(
    "%s lays out one point per node, inside the unit square",
    (_, g) => {
      const { x, y } = g.layout();
      expect(x).toHaveLength(g.n);
      expect(y).toHaveLength(g.n);
      for (const v of [...x, ...y]) {
        expect(v).toBeGreaterThanOrEqual(-1e-9);
        expect(v).toBeLessThanOrEqual(1 + 1e-9);
      }
    },
  );

  it("builds the Petersen graph: 10 nodes, 15 edges, 3-regular", () => {
    const g = graphById("petersen")!;
    expect(g.n).toBe(10);
    expect(g.edges).toHaveLength(15);
    for (const list of neighbours(g)) expect(list).toHaveLength(3);
  });

  // Pinned to networkx's karate_club_graph(): edge count, the full degree
  // sequence, and the triangle count. A transcription slip in one pair moves at
  // least two degrees, so the sequence catches what a count alone would not.
  it("transcribes Zachary's karate club exactly", () => {
    const g = graphById("karate")!;
    expect(g.n).toBe(34);
    expect(g.edges).toHaveLength(78);
    expect(neighbours(g).map((l) => l.length)).toEqual([
      16, 9, 10, 6, 3, 4, 4, 4, 5, 2, 3, 1, 2, 5, 2, 2, 2, 2, 2, 3, 2, 2, 2, 5, 3, 3, 2, 4, 3, 4,
      4, 6, 12, 17,
    ]);
    const a3 = adjacencyPower(g, 3, false);
    let trace = 0;
    for (let i = 0; i < g.n; i++) trace += a3[i * g.n + i];
    expect(trace / 6).toBe(45);
  });

  it("stores each edge in both directions in CSR", () => {
    const g = graphById("path-6")!;
    const { rowPtr, colIdx } = toCsr(g);
    expect(Array.from(rowPtr)).toEqual([0, 1, 3, 5, 7, 9, 10]);
    expect(Array.from(colIdx)).toEqual([1, 0, 2, 1, 3, 2, 4, 3, 5, 4]);
  });
});
