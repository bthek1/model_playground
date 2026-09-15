// Splitting a graph's *edges* into train / validation / test, for link
// prediction.
//
// This is the whole correctness surface of the /link-prediction route, and it
// fails in the one direction nobody notices: **upward**. If the encoder is
// allowed to aggregate over an edge it is later asked to predict, it has already
// averaged that edge's two endpoints together, and the score it gives the pair is
// memorised rather than inferred. Test AUC goes to ~0.99 and the page looks like
// it works better than it does. So the held-out edges are removed from the CSR
// the model propagates over, before anything else happens.
//
// Three things follow from that removal, each of which fails silently on its own:
//
//   - **Both directed copies go.** The graph has to stay symmetric, because that
//     is what makes the backward pass's `Âᵀ` the same kernel with α and β
//     swapped (see gnn.ts).
//   - **The degrees are recomputed.** `archScales` reads `degree` to build GCN's
//     `D^-1/2`, so a degree that still counts a removed edge leaks the edge's
//     existence into the normalisation even though the gather no longer visits
//     it.
//   - **Nothing may be isolated.** A node whose last neighbour was held out is
//     embedded from its own features alone, which is not a link prediction
//     problem any more. On Cora that would be up to 485 degree-1 nodes — enough
//     to move the number — so such an edge is kept instead.
//
// Negatives are non-edges of the **full** graph, not of the training graph: a
// "negative" that is really a held-out positive is a mislabelled example, and it
// would be scored against the model twice, once in each direction.

import { mulberry32, shuffle } from "./random";

/** A list of node pairs, flattened: `[u0, v0, u1, v1, …]`. */
export type EdgeList = Uint32Array;

export interface EdgeSplitOptions {
  /** Fraction of undirected edges held out for test. */
  testFrac?: number;
  /** Fraction held out for validation. */
  valFrac?: number;
  seed?: number;
}

export interface EdgeSplit {
  /** CSR over the training edges only — what the model is allowed to see. */
  rowPtr: Uint32Array;
  colIdx: Uint32Array;
  /** Neighbour count per node in the *training* graph, excluding self-loops. */
  degree: Uint32Array;

  /** Undirected training edges, as `u < v` pairs. The positive examples. */
  trainPos: EdgeList;
  valPos: EdgeList;
  testPos: EdgeList;
  /** Fixed negatives, one per positive, so the curve does not wander. */
  valNeg: EdgeList;
  testNeg: EdgeList;

  /** Undirected edges that were kept despite being drawn, to avoid isolating. */
  rescued: number;
}

/**
 * Every undirected edge of a symmetric CSR, once, as `u < v` pairs.
 *
 * Exported because it is also how the page counts what it is looking at, and
 * because "each undirected citation appears twice" is exactly the sort of thing
 * that is easy to halve or double by accident.
 */
export function undirectedEdges(
  rowPtr: Uint32Array,
  colIdx: Uint32Array,
  nNodes: number,
): EdgeList {
  const out = new Uint32Array(colIdx.length); // ≥ 2 * nUndirected
  let at = 0;
  for (let u = 0; u < nNodes; u++) {
    for (let e = rowPtr[u]; e < rowPtr[u + 1]; e++) {
      const v = colIdx[e];
      if (v > u) {
        out[at++] = u;
        out[at++] = v;
      }
    }
  }
  return out.slice(0, at);
}

/**
 * Is `v` a neighbour of `u`? Binary search within `u`'s CSR slice.
 *
 * Requires each row's neighbours to be ascending, which `decodeCora` guarantees
 * and `splitEdges` preserves — a linear scan would make negative sampling
 * O(|E|) per draw and the rejection loop is the hot path here.
 */
export function hasEdge(
  rowPtr: Uint32Array,
  colIdx: Uint32Array,
  u: number,
  v: number,
): boolean {
  let lo = rowPtr[u];
  let hi = rowPtr[u + 1] - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const got = colIdx[mid];
    if (got === v) return true;
    if (got < v) lo = mid + 1;
    else hi = mid - 1;
  }
  return false;
}

/**
 * Hold out a fraction of the edges, and rebuild the graph without them.
 *
 * The defaults are the usual 85 / 5 / 10. `seed` is what makes a comparison
 * between two architectures on this page mean anything: a split that moved
 * between runs would change the question as well as the answer.
 */
export function splitEdges(
  rowPtr: Uint32Array,
  colIdx: Uint32Array,
  nNodes: number,
  options: EdgeSplitOptions = {},
): EdgeSplit {
  const { testFrac = 0.1, valFrac = 0.05, seed = 7 } = options;
  if (testFrac < 0 || valFrac < 0 || testFrac + valFrac >= 1) {
    throw new Error(
      `held-out fractions must leave a training graph: got val=${valFrac} test=${testFrac}`,
    );
  }

  const edges = undirectedEdges(rowPtr, colIdx, nNodes);
  const nEdges = edges.length / 2;
  const rand = mulberry32(seed);

  const order = new Uint32Array(nEdges);
  for (let i = 0; i < nEdges; i++) order[i] = i;
  shuffle(order, rand);

  // Degrees are decremented as edges are removed, so the isolation guard sees
  // the graph as it will actually be, not as it started.
  const degree = new Uint32Array(nNodes);
  for (let u = 0; u < nNodes; u++) degree[u] = rowPtr[u + 1] - rowPtr[u];

  const wantTest = Math.floor(nEdges * testFrac);
  const wantVal = Math.floor(nEdges * valFrac);

  const held = new Uint8Array(nEdges); // 0 train, 1 val, 2 test
  let nTest = 0;
  let nVal = 0;
  let rescued = 0;

  for (const e of order) {
    if (nTest >= wantTest && nVal >= wantVal) break;
    const u = edges[2 * e];
    const v = edges[2 * e + 1];
    // Removing this would leave an endpoint with nothing to aggregate over.
    if (degree[u] <= 1 || degree[v] <= 1) {
      rescued++;
      continue;
    }
    degree[u]--;
    degree[v]--;
    if (nTest < wantTest) {
      held[e] = 2;
      nTest++;
    } else {
      held[e] = 1;
      nVal++;
    }
  }

  const trainPos = new Uint32Array((nEdges - nTest - nVal) * 2);
  const valPos = new Uint32Array(nVal * 2);
  const testPos = new Uint32Array(nTest * 2);
  let tAt = 0;
  let vAt = 0;
  let sAt = 0;
  for (let e = 0; e < nEdges; e++) {
    const u = edges[2 * e];
    const v = edges[2 * e + 1];
    if (held[e] === 2) {
      testPos[sAt++] = u;
      testPos[sAt++] = v;
    } else if (held[e] === 1) {
      valPos[vAt++] = u;
      valPos[vAt++] = v;
    } else {
      trainPos[tAt++] = u;
      trainPos[tAt++] = v;
    }
  }

  const { rowPtr: trainRowPtr, colIdx: trainColIdx } = buildCsr(
    trainPos,
    nNodes,
  );

  // Negatives are drawn against the *full* graph: a pair that is really a
  // held-out citation is not a negative example, it is a mislabelled one.
  const valNeg = sampleNegatives(rowPtr, colIdx, nNodes, nVal, rand);
  const testNeg = sampleNegatives(rowPtr, colIdx, nNodes, nTest, rand);

  return {
    rowPtr: trainRowPtr,
    colIdx: trainColIdx,
    degree,
    trainPos,
    valPos,
    testPos,
    valNeg,
    testNeg,
    rescued,
  };
}

/**
 * CSR from a list of undirected `u < v` pairs, each written in both directions
 * with every row ascending — the shape `hasEdge`, the aggregation kernel and the
 * backward pass all assume.
 */
export function buildCsr(
  pairs: EdgeList,
  nNodes: number,
): { rowPtr: Uint32Array; colIdx: Uint32Array } {
  const nPairs = pairs.length / 2;
  const rowPtr = new Uint32Array(nNodes + 1);
  for (let i = 0; i < nPairs; i++) {
    rowPtr[pairs[2 * i] + 1]++;
    rowPtr[pairs[2 * i + 1] + 1]++;
  }
  for (let u = 0; u < nNodes; u++) rowPtr[u + 1] += rowPtr[u];

  const colIdx = new Uint32Array(nPairs * 2);
  const at = rowPtr.slice(0, nNodes);
  for (let i = 0; i < nPairs; i++) {
    const u = pairs[2 * i];
    const v = pairs[2 * i + 1];
    colIdx[at[u]++] = v;
    colIdx[at[v]++] = u;
  }
  // Each row is filled in pair order, which is ascending in `v` only for the
  // forward direction. Sorting is what `hasEdge`'s binary search needs.
  for (let u = 0; u < nNodes; u++) {
    const row = colIdx.subarray(rowPtr[u], rowPtr[u + 1]);
    row.sort();
  }
  return { rowPtr, colIdx };
}

/**
 * `count` pairs that are not edges of the graph, not self-pairs and not repeats.
 *
 * Uniform rejection sampling. Cora is 0.07 % dense, so a draw is almost never
 * rejected; the loop is bounded anyway, because a caller that asks for more
 * negatives than a dense graph has should get an error rather than a hang.
 */
export function sampleNegatives(
  rowPtr: Uint32Array,
  colIdx: Uint32Array,
  nNodes: number,
  count: number,
  rand: () => number,
): EdgeList {
  const out = new Uint32Array(count * 2);
  const seen = new Set<number>();
  let at = 0;
  let tries = 0;
  const limit = Math.max(1000, count * 100);

  while (at < count * 2) {
    if (tries++ > limit) {
      throw new Error(
        `could not sample ${count} negatives after ${limit} draws — is the graph dense?`,
      );
    }
    const u = Math.floor(rand() * nNodes);
    const v = Math.floor(rand() * nNodes);
    if (u === v) continue;
    const a = Math.min(u, v);
    const b = Math.max(u, v);
    const key = a * nNodes + b;
    if (seen.has(key)) continue;
    if (hasEdge(rowPtr, colIdx, a, b)) continue;
    seen.add(key);
    out[at++] = a;
    out[at++] = b;
  }
  return out;
}
