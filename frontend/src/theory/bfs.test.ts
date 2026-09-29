import { describe, expect, it } from "vitest";

import { bfs } from "./bfs";
import { SMALL_GRAPHS, graphById, neighbours } from "./graphs";

/** All-pairs shortest paths by Floyd–Warshall — a reference that shares no code with BFS. */
function floydWarshall(n: number, edges: readonly (readonly [number, number])[]): number[][] {
  const d = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 0 : Infinity)),
  );
  for (const [u, v] of edges) d[u][v] = d[v][u] = 1;
  for (let k = 0; k < n; k++)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) if (d[i][k] + d[k][j] < d[i][j]) d[i][j] = d[i][k] + d[k][j];
  return d;
}

describe("bfs", () => {
  it.each(SMALL_GRAPHS.map((g) => [g.id, g] as const))(
    "%s: distances from every source match Floyd–Warshall",
    (_, g) => {
      const reference = floydWarshall(g.n, g.edges);
      const nbrs = neighbours(g);
      for (let s = 0; s < g.n; s++) {
        expect(Array.from(bfs(nbrs, s).dist)).toEqual(reference[s]);
      }
    },
  );

  it.each(SMALL_GRAPHS.map((g) => [g.id, g] as const))(
    "%s: the layers partition the nodes by distance",
    (_, g) => {
      const nbrs = neighbours(g);
      for (let s = 0; s < g.n; s++) {
        const { dist, layers, eccentricity } = bfs(nbrs, s);
        expect(layers[0]).toEqual([s]);
        expect(layers.flat().sort((a, b) => a - b)).toEqual(Array.from({ length: g.n }, (_, i) => i));
        layers.forEach((layer, d) => layer.forEach((v) => expect(dist[v]).toBe(d)));
        expect(eccentricity).toBe(Math.max(...dist));
      }
    },
  );

  // Known values, not derived ones.
  it("gives the published eccentricities and diameters", () => {
    const ecc = (id: string, s: number) => bfs(neighbours(graphById(id)!), s).eccentricity;
    const diameter = (id: string) => {
      const g = graphById(id)!;
      return Math.max(...Array.from({ length: g.n }, (_, s) => ecc(id, s)));
    };
    expect(ecc("path-6", 0)).toBe(5);
    expect(ecc("cycle-8", 0)).toBe(4);
    expect(ecc("cycle-7", 0)).toBe(3);
    expect(ecc("grid-4", 0)).toBe(6); // corner to corner
    expect(diameter("petersen")).toBe(2);
    // networkx: diameter 5, radius 3, eccentricity 3 at the instructor and 4 at the officer.
    expect(diameter("karate")).toBe(5);
    expect(ecc("karate", 0)).toBe(3);
    expect(ecc("karate", 33)).toBe(4);
  });

  it("marks unreachable nodes −1 and stops", () => {
    const { dist, layers, eccentricity } = bfs([[1], [0], []], 0);
    expect(Array.from(dist)).toEqual([0, 1, -1]);
    expect(layers).toEqual([[0], [1]]);
    expect(eccentricity).toBe(1);
  });

  it("refuses a source outside the graph", () => {
    expect(() => bfs([[]], 1)).toThrow(RangeError);
  });
});
