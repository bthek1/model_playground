import { describe, expect, it } from "vitest";

import { bfs } from "./bfs";
import { SMALL_GRAPHS, graphById, neighbours } from "./graphs";
import { adjacencyMatrix, adjacencyPower, row, support } from "./walks";

/** Count walks of length k from s to every node by enumerating them. */
function enumerateWalks(nbrs: number[][], s: number, k: number, selfLoops: boolean): number[] {
  const counts = new Array(nbrs.length).fill(0);
  const go = (at: number, left: number) => {
    if (left === 0) {
      counts[at]++;
      return;
    }
    for (const v of nbrs[at]) go(v, left - 1);
    if (selfLoops) go(at, left - 1);
  };
  go(s, k);
  return counts;
}

function trace(m: Float64Array, n: number): number {
  let t = 0;
  for (let i = 0; i < n; i++) t += m[i * n + i];
  return t;
}

describe("adjacencyPower", () => {
  it("is the identity at k = 0 and the adjacency matrix at k = 1", () => {
    const g = graphById("cycle-7")!;
    const id = adjacencyPower(g, 0, false);
    for (let i = 0; i < g.n; i++) for (let j = 0; j < g.n; j++) expect(id[i * g.n + j]).toBe(i === j ? 1 : 0);
    expect(adjacencyPower(g, 1, false)).toEqual(adjacencyMatrix(g));
    expect(adjacencyPower(g, 1, true)).toEqual(adjacencyMatrix(g, true));
  });

  // The reference shares nothing with the matrix product: it walks.
  it.each(SMALL_GRAPHS.filter((g) => g.n <= 16).map((g) => [g.id, g] as const))(
    "%s: counts every walk of length ≤ 4 exactly, with and without the self-loop",
    (_, g) => {
      const nbrs = neighbours(g);
      for (const selfLoops of [false, true]) {
        for (let k = 0; k <= 4; k++) {
          const p = adjacencyPower(g, k, selfLoops);
          for (let s = 0; s < g.n; s++) {
            expect(Array.from(row(p, g.n, s))).toEqual(enumerateWalks(nbrs, s, k, selfLoops));
          }
        }
      }
    },
  );

  // Petersen's spectrum is 3 (once), 1 (five times) and −2 (four times), and
  // trace(Aᵏ) is the sum of the eigenvalues' k-th powers: 3ᵏ + 5 + 4·(−2)ᵏ.
  it("matches the Petersen graph's spectrum: trace(Aᵏ) = 3ᵏ + 5 + 4(−2)ᵏ", () => {
    const g = graphById("petersen")!;
    for (let k = 1; k <= 6; k++) {
      expect(trace(adjacencyPower(g, k, false), g.n)).toBe(3 ** k + 5 + 4 * (-2) ** k);
    }
    expect(trace(adjacencyPower(g, 3, false), g.n)).toBe(0); // no triangles
    expect(trace(adjacencyPower(g, 4, false), g.n)).toBe(150);
  });

  it("stays exact on the largest power the page can ask for", () => {
    const g = graphById("karate")!;
    const p = adjacencyPower(g, 5, true);
    const max = Math.max(...p);
    expect(Number.isSafeInteger(max)).toBe(true);
    expect(max).toBe(4744); // the number walks.ts quotes
  });

  it("refuses a negative or fractional power", () => {
    const g = graphById("path-6")!;
    expect(() => adjacencyPower(g, -1, false)).toThrow(RangeError);
    expect(() => adjacencyPower(g, 1.5, false)).toThrow(RangeError);
  });
});

// The page's claim, proved over every graph, every source and every step it
// can show. This is the property a GCN's receptive field rests on.
describe("the receptive-field theorem", () => {
  it.each(SMALL_GRAPHS.map((g) => [g.id, g] as const))(
    "%s: row s of (A+I)ᵏ is nonzero exactly within k hops of s",
    (_, g) => {
      const nbrs = neighbours(g);
      for (let s = 0; s < g.n; s++) {
        const { dist, eccentricity } = bfs(nbrs, s);
        for (let k = 0; k <= eccentricity; k++) {
          const ball = support(Array.from(dist, (d) => (d >= 0 && d <= k ? 1 : 0)));
          expect(support(row(adjacencyPower(g, k, true), g.n, s))).toEqual(ball);
        }
      }
    },
  );

  it.each(SMALL_GRAPHS.filter((g) => g.bipartite).map((g) => [g.id, g] as const))(
    "%s (bipartite): row s of Aᵏ is nonzero exactly within k hops at k's parity",
    (_, g) => {
      const nbrs = neighbours(g);
      for (let s = 0; s < g.n; s++) {
        const { dist, eccentricity } = bfs(nbrs, s);
        for (let k = 1; k <= eccentricity; k++) {
          const expected = support(Array.from(dist, (d) => (d <= k && d % 2 === k % 2 ? 1 : 0)));
          expect(support(row(adjacencyPower(g, k, false), g.n, s))).toEqual(expected);
        }
      }
    },
  );

  it("on C₈ the source cannot hear itself at odd k without the self-loop", () => {
    const g = graphById("cycle-8")!;
    expect(row(adjacencyPower(g, 3, false), g.n, 0)[0]).toBe(0);
    expect(row(adjacencyPower(g, 3, true), g.n, 0)[0]).toBeGreaterThan(0);
  });

  // The contrast the page draws: an odd cycle breaks parity, but only at its own
  // length. On C₇ no closed walk of length 5 exists; one of length 7 goes round.
  it("on C₇ an odd closed walk exists, at length 7", () => {
    const g = graphById("cycle-7")!;
    expect(row(adjacencyPower(g, 5, false), g.n, 0)[0]).toBe(0);
    expect(row(adjacencyPower(g, 7, false), g.n, 0)[0]).toBe(2); // once each way round
  });
});

describe("support", () => {
  it("lists the nonzero positions in order", () => {
    expect(support([0, 2, 0, 1])).toEqual([1, 3]);
    expect(support([])).toEqual([]);
  });
});
