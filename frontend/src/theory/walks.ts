// Powers of the adjacency matrix, counted exactly.
//
// `(Aᵏ)[s][v]` is the number of walks of length k from s to v — the one fact
// in discrete maths that message passing is built on. A GCN layer multiplies
// by `Â`, so k layers multiply by `Âᵏ`, and node s can only have heard from the
// nodes where row s of that power is nonzero. Which power matters:
//
//   - `(A + I)ᵏ`: a walk may pause, so row s is nonzero at exactly the nodes
//     within k hops — the BFS ball. This is what GCN computes, and why its
//     kernel adds the self-loop (`/graph` stores A and adds I in the shader).
//   - `Aᵏ` alone: a walk must move every step. On a bipartite graph it can only
//     end at nodes whose distance has k's parity, so row s *flickers*: at odd k
//     the source cannot hear itself at all.
//
// Counts are exact integers in float64. The page caps k at the graph's
// eccentricity, and the largest count that reaches is 4744 (the karate club's
// (A + I)⁵) — far inside 2⁵³. The guard below makes a larger catalogue entry
// fail loudly rather than round.

import type { SmallGraph } from "./graphs";

/** The n×n adjacency matrix, row-major, optionally with the self-loop added. */
export function adjacencyMatrix(
  graph: Pick<SmallGraph, "n" | "edges">,
  selfLoops = false,
): Float64Array {
  const { n } = graph;
  const a = new Float64Array(n * n);
  for (const [u, v] of graph.edges) {
    a[u * n + v] = 1;
    a[v * n + u] = 1;
  }
  if (selfLoops) for (let i = 0; i < n; i++) a[i * n + i] += 1;
  return a;
}

function multiply(a: Float64Array, b: Float64Array, n: number): Float64Array<ArrayBuffer> {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      const aik = a[i * n + k];
      if (aik === 0) continue;
      for (let j = 0; j < n; j++) out[i * n + j] += aik * b[k * n + j];
    }
  }
  for (let i = 0; i < out.length; i++) {
    if (out[i] > Number.MAX_SAFE_INTEGER) {
      throw new RangeError("walk count exceeds 2^53 — it would no longer be exact");
    }
  }
  return out;
}

/** `Mᵏ` where `M` is A or A + I. `k = 0` is the identity. */
export function adjacencyPower(
  graph: Pick<SmallGraph, "n" | "edges">,
  k: number,
  selfLoops: boolean,
): Float64Array {
  if (!Number.isInteger(k) || k < 0) throw new RangeError(`k must be a non-negative integer, got ${k}`);
  const { n } = graph;
  const m = adjacencyMatrix(graph, selfLoops);
  let out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) out[i * n + i] = 1;
  for (let step = 0; step < k; step++) out = multiply(out, m, n);
  return out;
}

/** Row `source` of a row-major n×n matrix. */
export function row(matrix: Float64Array, n: number, source: number): Float64Array {
  return matrix.subarray(source * n, source * n + n);
}

/** The nodes where `values` is nonzero, ascending. */
export function support(values: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < values.length; i++) if (values[i] !== 0) out.push(i);
  return out;
}
