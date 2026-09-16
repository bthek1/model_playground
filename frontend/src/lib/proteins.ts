// PROTEINS — 1113 protein graphs, one label each: enzyme or not.
//
// The third dataset on the raw-WebGPU path and the only one that is fetched
// rather than bundled. `graphs-datasets/PROTEINS` publishes a single 2.06 MB
// `full.jsonl`, and huggingface.co echoes the caller's origin, so a browser on
// our own origin may read it directly. (A `curl -I` with no `Origin` header
// answers `access-control-allow-origin: https://huggingface.co` and looks like a
// blocker — it is not. Send the header and the answer changes.)
//
// Two things make this dataset fit the machinery already here rather than needing
// new machinery:
//
//   - **Every graph is symmetric and none has a self-loop.** Those are exactly
//     the two invariants `webgpu/gnn.ts` depends on: symmetry is what makes the
//     backward pass's `Âᵀ` the same kernel with α and β swapped, and the missing
//     self-loop is what the kernel's own `+I` supplies. Both hold over all 1113
//     graphs — `buildUnion` **checks** rather than assuming, because a violation
//     of either produces a model that trains and is quietly wrong.
//   - **The whole dataset is one graph.** Batching graph classification means a
//     block-diagonal disjoint union plus a node→graph vector, which is what PyG
//     builds per mini-batch. The union of all 1113 graphs is 43 471 nodes and
//     162 088 directed entries — smaller in every dimension than the Cora matmul
//     /graph already runs — so there is no mini-batching here at all. It is built
//     once and trained full-batch, like the other two routes.
//
// The label balance is 663 / 450, so **the majority-class baseline is 59.6 %**.
// Published GNNs score ~73-76 %. That gap is the whole reason `majorityBaseline`
// lives here and the page renders it: an accuracy printed on its own cannot be
// told apart from a model that learned the prior and ignored its input.

import { mulberry32, shuffle } from "./random";

export const PROTEINS_URL =
  "https://huggingface.co/datasets/graphs-datasets/PROTEINS/resolve/main/full.jsonl";

/** Bytes of the published file, for the size the LOAD slot quotes. */
export const PROTEINS_BYTES = 2_058_444;

export const PROTEINS_CLASSES = ["Not an enzyme", "Enzyme"] as const;

/** One row of `full.jsonl`, as published. */
export interface ProteinRow {
  /** `[src[], dst[]]`, both directions present. */
  edge_index: number[][];
  /** `num_nodes × 3`. */
  node_feat: number[][];
  y: number[];
  num_nodes: number;
}

/**
 * Every graph concatenated into one, block-diagonally.
 *
 * `rowPtr`/`colIdx` are the union's CSR — node indices are **global**, and the
 * single most important property of this structure is that no edge crosses from
 * one graph's node range into another's. A union that leaks an edge across the
 * boundary trains perfectly happily and scores a plausible accuracy; nothing
 * downstream can notice.
 */
export interface ProteinUnion {
  nGraphs: number;
  nNodes: number;
  nFeat: number;
  nClasses: number;
  rowPtr: Uint32Array;
  colIdx: Uint32Array;
  /** Neighbour count per node, excluding the self-loop the kernel adds. */
  degree: Uint32Array;
  /** Dense `nNodes × nFeat`, row-major. */
  features: Float32Array;
  /** Which graph each node belongs to. */
  graphOf: Uint32Array;
  /** Node range of each graph: graph `g` owns `[graphPtr[g], graphPtr[g+1])`. */
  graphPtr: Uint32Array;
  /** One label per graph. */
  labels: Uint8Array;
}

export interface GraphSplitIndices {
  train: Uint32Array;
  val: Uint32Array;
  test: Uint32Array;
}

export interface ProteinSplitOptions {
  trainFrac?: number;
  valFrac?: number;
  seed?: number;
}

/** Parse the published JSONL. Blank lines are tolerated; bad JSON is not. */
export function parseProteinsJsonl(text: string): ProteinRow[] {
  const rows: ProteinRow[] = [];
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.length === 0) continue;
    rows.push(JSON.parse(trimmed) as ProteinRow);
  }
  if (rows.length === 0) throw new Error("PROTEINS: the file held no graphs");
  return rows;
}

/**
 * Build the disjoint union, checking the two invariants as it goes.
 *
 * The checks are not defensive programming for its own sake. A self-loop in the
 * stored edges would be counted twice, because the aggregation kernel adds `I`
 * itself; an asymmetric edge would make the backward pass's transpose wrong. Both
 * produce a falling loss and a plausible accuracy curve, which is this whole
 * category's documented failure mode.
 */
export function buildUnion(rows: ProteinRow[]): ProteinUnion {
  const nGraphs = rows.length;
  const nFeat = rows[0].node_feat[0]?.length ?? 0;
  if (nFeat === 0) throw new Error("PROTEINS: graphs have no node features");

  const graphPtr = new Uint32Array(nGraphs + 1);
  for (let g = 0; g < nGraphs; g++) {
    graphPtr[g + 1] = graphPtr[g] + rows[g].num_nodes;
  }
  const nNodes = graphPtr[nGraphs];

  const degree = new Uint32Array(nNodes);
  const graphOf = new Uint32Array(nNodes);
  const features = new Float32Array(nNodes * nFeat);
  const labels = new Uint8Array(nGraphs);

  // Pass one: node-level data, degrees, and the invariant checks.
  let totalEntries = 0;
  for (let g = 0; g < nGraphs; g++) {
    const row = rows[g];
    const base = graphPtr[g];
    labels[g] = row.y[0] ?? 0;

    if (row.node_feat.length !== row.num_nodes) {
      throw new Error(
        `PROTEINS: graph ${g} says ${row.num_nodes} nodes but has ${row.node_feat.length} feature rows`,
      );
    }
    for (let i = 0; i < row.num_nodes; i++) {
      graphOf[base + i] = g;
      const feat = row.node_feat[i];
      for (let f = 0; f < nFeat; f++) features[(base + i) * nFeat + f] = feat[f];
    }

    const [src, dst] = row.edge_index;
    if (src.length !== dst.length) {
      throw new Error(`PROTEINS: graph ${g} has a ragged edge_index`);
    }
    const seen = new Set<number>();
    for (let e = 0; e < src.length; e++) {
      const u = src[e];
      const v = dst[e];
      if (u === v) {
        throw new Error(
          `PROTEINS: graph ${g} stores a self-loop at node ${u}; the kernel adds I itself and would count it twice`,
        );
      }
      if (u < 0 || v < 0 || u >= row.num_nodes || v >= row.num_nodes) {
        throw new Error(
          `PROTEINS: graph ${g} has an edge ${u}→${v} outside its ${row.num_nodes} nodes`,
        );
      }
      seen.add(u * row.num_nodes + v);
      degree[base + u]++;
    }
    for (let e = 0; e < src.length; e++) {
      if (!seen.has(dst[e] * row.num_nodes + src[e])) {
        throw new Error(
          `PROTEINS: graph ${g} has ${src[e]}→${dst[e]} but not its reverse; the backward pass's transpose needs symmetry`,
        );
      }
    }
    totalEntries += src.length;
  }

  // Pass two: the CSR itself.
  const rowPtr = new Uint32Array(nNodes + 1);
  for (let i = 0; i < nNodes; i++) rowPtr[i + 1] = rowPtr[i] + degree[i];
  const colIdx = new Uint32Array(totalEntries);
  const cursor = rowPtr.slice(0, nNodes);
  for (let g = 0; g < nGraphs; g++) {
    const base = graphPtr[g];
    const [src, dst] = rows[g].edge_index;
    for (let e = 0; e < src.length; e++) {
      colIdx[cursor[base + src[e]]++] = base + dst[e];
    }
  }
  // Ascending rows, the shape every consumer of a CSR here assumes.
  for (let i = 0; i < nNodes; i++) {
    colIdx.subarray(rowPtr[i], rowPtr[i + 1]).sort();
  }

  return {
    nGraphs,
    nNodes,
    nFeat,
    nClasses: PROTEINS_CLASSES.length,
    rowPtr,
    colIdx,
    degree,
    features,
    graphOf,
    graphPtr,
    labels,
  };
}

/**
 * A seeded split over **graphs**, stratified by label.
 *
 * Stratified because the page is read against the majority-class baseline, and an
 * unstratified test split moves that baseline away from the 59.6 % the page
 * quotes — so the number on screen would be compared against the wrong null.
 */
export function makeGraphSplit(
  labels: Uint8Array,
  nClasses: number,
  options: ProteinSplitOptions = {},
): GraphSplitIndices {
  const { trainFrac = 0.8, valFrac = 0.1, seed = 5 } = options;
  if (trainFrac <= 0 || valFrac < 0 || trainFrac + valFrac >= 1) {
    throw new Error(
      `the split must leave a test set: got train=${trainFrac} val=${valFrac}`,
    );
  }
  const rand = mulberry32(seed);
  const train: number[] = [];
  const val: number[] = [];
  const test: number[] = [];

  for (let c = 0; c < nClasses; c++) {
    const members: number[] = [];
    for (let g = 0; g < labels.length; g++) if (labels[g] === c) members.push(g);
    shuffle(members, rand);

    const nTrain = Math.round(members.length * trainFrac);
    const nVal = Math.round(members.length * valFrac);
    train.push(...members.slice(0, nTrain));
    val.push(...members.slice(nTrain, nTrain + nVal));
    test.push(...members.slice(nTrain + nVal));
  }

  const sorted = (xs: number[]) => Uint32Array.from(xs.sort((a, b) => a - b));
  return { train: sorted(train), val: sorted(val), test: sorted(test) };
}

/**
 * The accuracy of answering the commonest training class every time.
 *
 * The number the page prints beside its own. On PROTEINS it is 59.6 %, and a
 * model that learned nothing but the prior scores exactly that — which is why an
 * accuracy rendered on its own is unreadable rather than merely unadorned.
 */
export function majorityBaseline(
  labels: Uint8Array,
  trainIdx: Uint32Array,
  evalIdx: Uint32Array,
  nClasses: number,
): { label: number; accuracy: number } {
  const counts = new Uint32Array(nClasses);
  for (const g of trainIdx) counts[labels[g]]++;
  let label = 0;
  for (let c = 1; c < nClasses; c++) if (counts[c] > counts[label]) label = c;

  if (evalIdx.length === 0) return { label, accuracy: 0 };
  let hits = 0;
  for (const g of evalIdx) if (labels[g] === label) hits++;
  return { label, accuracy: hits / evalIdx.length };
}

/** Fetch and parse the published file. The caller owns caching. */
export async function fetchProteins(signal?: AbortSignal): Promise<ProteinRow[]> {
  const response = await fetch(PROTEINS_URL, { signal });
  if (!response.ok) {
    throw new Error(
      `PROTEINS: the dataset could not be fetched (${response.status} ${response.statusText})`,
    );
  }
  return parseProteinsJsonl(await response.text());
}
