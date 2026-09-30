// The small graphs `/discrete-maths` walks over.
//
// Chosen in pairs, because the page's lesson is a *contrast*: a bipartite graph
// and a nearly identical one that is not (C8 against C7), a sparse one and a
// dense one (the path against the Petersen graph), and one real network so the
// arithmetic is seen on something nobody drew to make a point (Zachary's
// karate club). Every property the page states about one of these is pinned in
// `graphs.test.ts` against an independent reference, never against the page.
//
// Positions are hand-placed where a graph has a canonical drawing — a cycle is
// a circle, a grid is a grid, Petersen is a pentagon around a pentagram —
// because a force layout of a cycle is a lumpy loop and hides the symmetry the
// walk counts come from. The karate club has no canonical drawing, so it gets
// the same seeded force layout `/graph` uses.

import { forceLayout } from "@/lib/graphLayout";

export interface SmallGraph {
  id: string;
  label: string;
  /** Node count. Nodes are `0 … n-1`. */
  n: number;
  /** Each undirected edge once, `[u, v]` with `u < v`. No self-loops. */
  edges: readonly (readonly [number, number])[];
  /**
   * Whether the nodes 2-colour with every edge crossing — equivalently, no
   * cycle of odd length. Catalogue data, checked in the test against the
   * spectrum-free definition (no closed walk of odd length), not against BFS.
   */
  bipartite: boolean;
  /** One sentence: why this graph is on the page. */
  note: string;
  /** Where the data came from, when it did not come from a definition. */
  source?: string;
  /** Node coordinates in the unit square. */
  layout: () => { x: number[]; y: number[] };
}

function pathEdges(n: number): [number, number][] {
  return Array.from({ length: n - 1 }, (_, i) => [i, i + 1] as [number, number]);
}

function cycleEdges(n: number): [number, number][] {
  return [...pathEdges(n), [0, n - 1]];
}

function circle(n: number, radius = 0.5, phase = -Math.PI / 2, offset = 0) {
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = phase + (2 * Math.PI * (i + offset)) / n;
    x.push(0.5 + radius * Math.cos(a));
    y.push(0.5 + radius * Math.sin(a));
  }
  return { x, y };
}

function gridEdges(side: number): [number, number][] {
  const edges: [number, number][] = [];
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      const i = r * side + c;
      if (c + 1 < side) edges.push([i, i + 1]);
      if (r + 1 < side) edges.push([i, i + side]);
    }
  }
  return edges;
}

// Outer 5-cycle 0–4, spokes i—i+5, inner pentagram 5–9 joined i—i+2.
const PETERSEN_EDGES: [number, number][] = [
  ...cycleEdges(5),
  ...Array.from({ length: 5 }, (_, i) => [i, i + 5] as [number, number]),
  ...Array.from(
    { length: 5 },
    (_, i) => [5 + i, 5 + ((i + 2) % 5)].sort((a, b) => a - b) as [number, number],
  ),
];

// Zachary (1977), as networkx's `karate_club_graph()` ships it: an edge wherever
// either direction of the weighted interaction matrix is nonzero. 34 members,
// 78 friendships; node 0 is the instructor ("Mr. Hi") and node 33 the club
// officer whose falling-out split the club in two.
const KARATE_EDGES: [number, number][] = [
  [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 10], [0, 11], [0, 12],
  [0, 13], [0, 17], [0, 19], [0, 21], [0, 31], [1, 2], [1, 3], [1, 7], [1, 13], [1, 17],
  [1, 19], [1, 21], [1, 30], [2, 3], [2, 7], [2, 8], [2, 9], [2, 13], [2, 27], [2, 28],
  [2, 32], [3, 7], [3, 12], [3, 13], [4, 6], [4, 10], [5, 6], [5, 10], [5, 16], [6, 16],
  [8, 30], [8, 32], [8, 33], [9, 33], [13, 33], [14, 32], [14, 33], [15, 32], [15, 33],
  [18, 32], [18, 33], [19, 33], [20, 32], [20, 33], [22, 32], [22, 33], [23, 25], [23, 27],
  [23, 29], [23, 32], [23, 33], [24, 25], [24, 27], [24, 31], [25, 31], [26, 29], [26, 33],
  [27, 33], [28, 31], [28, 33], [29, 32], [29, 33], [30, 32], [30, 33], [31, 32], [31, 33],
  [32, 33],
];

/** CSR with each undirected edge stored in both directions — `forceLayout`'s input. */
export function toCsr(graph: Pick<SmallGraph, "n" | "edges">): {
  rowPtr: Uint32Array;
  colIdx: Uint32Array;
} {
  const nbrs = neighbours(graph);
  const rowPtr = new Uint32Array(graph.n + 1);
  for (let i = 0; i < graph.n; i++) rowPtr[i + 1] = rowPtr[i] + nbrs[i].length;
  const colIdx = new Uint32Array(rowPtr[graph.n]);
  for (let i = 0; i < graph.n; i++) colIdx.set(nbrs[i], rowPtr[i]);
  return { rowPtr, colIdx };
}

/** Sorted neighbour lists. */
export function neighbours(graph: Pick<SmallGraph, "n" | "edges">): number[][] {
  const out: number[][] = Array.from({ length: graph.n }, () => []);
  for (const [u, v] of graph.edges) {
    out[u].push(v);
    out[v].push(u);
  }
  for (const list of out) list.sort((a, b) => a - b);
  return out;
}

export const SMALL_GRAPHS: readonly SmallGraph[] = [
  {
    id: "path-6",
    label: "Path P₆",
    n: 6,
    edges: pathEdges(6),
    bipartite: true,
    note: "The sparsest connected graph: one route between any two nodes, so every walk count is small.",
    layout: () => ({
      x: Array.from({ length: 6 }, (_, i) => i / 5),
      y: Array.from({ length: 6 }, (_, i) => 0.5 + (i % 2 ? 0.08 : -0.08)),
    }),
  },
  {
    id: "cycle-8",
    label: "Cycle C₈",
    n: 8,
    edges: cycleEdges(8),
    bipartite: true,
    note: "Even cycle, so bipartite: without a self-loop, a node hears from itself only at even hops.",
    layout: () => circle(8),
  },
  {
    id: "cycle-7",
    label: "Cycle C₇",
    n: 7,
    edges: cycleEdges(7),
    bipartite: false,
    note: "One node fewer, and the odd cycle breaks the parity — compare it with C₈.",
    layout: () => circle(7),
  },
  {
    id: "grid-4",
    label: "Grid 4×4",
    n: 16,
    edges: gridEdges(4),
    bipartite: true,
    note: "A chessboard is bipartite: the colours alternate, and so do the walk counts.",
    layout: () => ({
      x: Array.from({ length: 16 }, (_, i) => (i % 4) / 3),
      y: Array.from({ length: 16 }, (_, i) => Math.floor(i / 4) / 3),
    }),
  },
  {
    id: "petersen",
    label: "Petersen graph",
    n: 10,
    edges: PETERSEN_EDGES,
    bipartite: false,
    note: "Every node within two hops of every other, and no triangle anywhere: diameter 2, girth 5.",
    layout: () => {
      const outer = circle(5, 0.5);
      const inner = circle(5, 0.22);
      return { x: [...outer.x, ...inner.x], y: [...outer.y, ...inner.y] };
    },
  },
  {
    id: "karate",
    label: "Zachary's karate club",
    n: 34,
    edges: KARATE_EDGES,
    bipartite: false,
    note: "A real friendship network: 34 members, 78 ties, and two hubs — the instructor (0) and the officer (33).",
    source: "W. W. Zachary (1977), as shipped by networkx's karate_club_graph()",
    layout: () => {
      const { rowPtr, colIdx } = toCsr({ n: 34, edges: KARATE_EDGES });
      const { x, y } = forceLayout(rowPtr, colIdx, 34, { seed: 7 });
      return { x: Array.from(x), y: Array.from(y) };
    },
  },
];

export const DEFAULT_GRAPH = SMALL_GRAPHS[1];

export function graphById(id: string): SmallGraph | undefined {
  return SMALL_GRAPHS.find((g) => g.id === id);
}
