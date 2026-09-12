// The Cora citation network, decoded from the binary asset that ships with the
// app. 2708 machine-learning papers, 5278 undirected citations, a 1433-word
// binary bag-of-words per paper and one of seven topic labels.
//
// Nothing here touches the network: `cora.bin` is produced offline by
// scripts/prepare-cora.mjs and bundled, so /graph works on a plane. The binary
// exists because the feature matrix is 15.5 MB dense and 1.27% dense in fact —
// it is stored as CSR with u16 column indices and no values array, since every
// stored value is 1.
//
// Everything below `decodeCora` is a pure transform over the decoded arrays and
// is unit-tested against a synthetic buffer the test builds itself.

import coraUrl from "./data/cora.bin?url";
import { mulberry32, shuffle } from "./random";

/** Must match the header written by scripts/prepare-cora.mjs. */
const MAGIC = 0x41524f43; // "CORA" read as u32 little-endian
const VERSION = 1;
const HEADER_BYTES = 28;

/** Topic labels, in the order the prep script assigned them (alphabetical). */
export const CORA_CLASSES = [
  "Case Based",
  "Genetic Algorithms",
  "Neural Networks",
  "Probabilistic Methods",
  "Reinforcement Learning",
  "Rule Learning",
  "Theory",
] as const;

export interface CoraGraph {
  nNodes: number;
  nFeat: number;
  nClasses: number;
  /**
   * CSR over the symmetrised citation graph. Self-loops are **not** stored:
   * message passing is `A + I` and the aggregation kernel adds `I` itself, so a
   * stored self-loop would count a node's own features twice.
   */
  rowPtr: Uint32Array; // nNodes + 1
  colIdx: Uint32Array; // nEdges (each undirected citation appears twice)
  /** Row-normalised bag-of-words, dense row-major `nNodes * nFeat`. */
  features: Float32Array;
  labels: Uint8Array;
  /** Neighbour count per node, **excluding** the self-loop. */
  degree: Uint32Array;
}

/**
 * Decode the bundled binary. Throws rather than returning something plausible if
 * the header does not match: a graph silently decoded at the wrong offset still
 * trains and still produces an accuracy curve, which is this route's documented
 * failure mode.
 */
export function decodeCora(buffer: ArrayBuffer): CoraGraph {
  if (buffer.byteLength < HEADER_BYTES) {
    throw new Error(`cora.bin is truncated: ${buffer.byteLength} bytes`);
  }
  const header = new DataView(buffer);
  const magic = header.getUint32(0, true);
  if (magic !== MAGIC) {
    throw new Error(`cora.bin has a bad magic number: 0x${magic.toString(16)}`);
  }
  const version = header.getUint32(4, true);
  if (version !== VERSION) {
    throw new Error(`cora.bin is version ${version}, expected ${VERSION}`);
  }

  const nNodes = header.getUint32(8, true);
  const nFeat = header.getUint32(12, true);
  const nClasses = header.getUint32(16, true);
  const nEdges = header.getUint32(20, true);
  const featNnz = header.getUint32(24, true);

  let at = HEADER_BYTES;
  const take = <T>(make: (b: ArrayBuffer, o: number, n: number) => T, count: number, width: number): T => {
    const array = make(buffer, at, count);
    at += count * width;
    return array;
  };

  const rowPtr = take((b, o, n) => new Uint32Array(b, o, n), nNodes + 1, 4);
  const colIdx = take((b, o, n) => new Uint32Array(b, o, n), nEdges, 4);
  const featRowPtr = take((b, o, n) => new Uint32Array(b, o, n), nNodes + 1, 4);
  const featColIdx = take((b, o, n) => new Uint16Array(b, o, n), featNnz, 2);
  const labels = take((b, o, n) => new Uint8Array(b, o, n), nNodes, 1);

  if (at !== buffer.byteLength) {
    throw new Error(`cora.bin has ${buffer.byteLength - at} trailing bytes`);
  }

  const degree = new Uint32Array(nNodes);
  for (let i = 0; i < nNodes; i++) degree[i] = rowPtr[i + 1] - rowPtr[i];

  return {
    nNodes,
    nFeat,
    nClasses,
    rowPtr,
    colIdx,
    features: densifyRowNormalised(featRowPtr, featColIdx, nNodes, nFeat),
    // A copy, so the returned graph does not alias the decoded ArrayBuffer.
    labels: labels.slice(),
    degree,
  };
}

/**
 * Expand the binary feature CSR into a dense row-normalised matrix — Kipf's
 * `preprocess_features`. Every stored value is 1, so a row's normalised value is
 * just `1 / nnz`; rows with no words stay zero rather than becoming NaN.
 *
 * Dense is deliberate: the first layer is `X · W`, with `W` only 16 columns
 * wide, so the naive matmul kernel handles it in milliseconds and a sparse
 * matmul kernel would buy nothing but a second thing to get wrong.
 */
export function densifyRowNormalised(
  rowPtr: Uint32Array,
  colIdx: Uint16Array,
  nNodes: number,
  nFeat: number,
): Float32Array {
  const dense = new Float32Array(nNodes * nFeat);
  for (let i = 0; i < nNodes; i++) {
    const start = rowPtr[i];
    const end = rowPtr[i + 1];
    if (end === start) continue;
    const value = 1 / (end - start);
    const base = i * nFeat;
    for (let e = start; e < end; e++) dense[base + colIdx[e]] = value;
  }
  return dense;
}

export interface GraphSplit {
  /** Node indices, ascending. */
  train: Uint32Array;
  val: Uint32Array;
  test: Uint32Array;
  /** The same three sets as per-node flags, for rendering. */
  trainMask: Uint8Array;
  valMask: Uint8Array;
  testMask: Uint8Array;
}

export interface SplitOptions {
  /** Labelled nodes per class. Planetoid's semi-supervised setting uses 20. */
  perClass?: number;
  valSize?: number;
  testSize?: number;
  seed?: number;
}

/**
 * The semi-supervised split: `perClass` labelled nodes per class, then a
 * validation and a test set drawn from what is left.
 *
 * This is **our** split, not the published Planetoid one — that split is a fixed
 * index list over Planetoid's node ordering, and this file's ordering comes from
 * `cora.content` instead. It is the same recipe (20 per class / 500 / 1000) and
 * lands in the same place, so a GCN here scores around 0.80 as it does in the
 * literature, but the numbers are not identical to a published table.
 *
 * It is seeded because the page compares architectures and depths against each
 * other: a split that moved between two runs would make every comparison on the
 * page meaningless.
 */
export function makeSplit(
  labels: Uint8Array,
  nClasses: number,
  options: SplitOptions = {},
): GraphSplit {
  const { perClass = 20, valSize = 500, testSize = 1000, seed = 1 } = options;
  const nNodes = labels.length;
  const rand = mulberry32(seed);

  const byClass: number[][] = Array.from({ length: nClasses }, () => []);
  for (let i = 0; i < nNodes; i++) byClass[labels[i]].push(i);

  const train: number[] = [];
  for (const members of byClass) {
    shuffle(members, rand);
    if (members.length < perClass) {
      throw new Error(
        `a class has ${members.length} nodes, fewer than the ${perClass} the split needs`,
      );
    }
    train.push(...members.slice(0, perClass));
  }

  const claimed = new Uint8Array(nNodes);
  for (const i of train) claimed[i] = 1;
  const rest: number[] = [];
  for (let i = 0; i < nNodes; i++) if (!claimed[i]) rest.push(i);
  shuffle(rest, rand);

  if (rest.length < valSize + testSize) {
    throw new Error(
      `only ${rest.length} nodes left for a ${valSize}+${testSize} val/test split`,
    );
  }
  const val = rest.slice(0, valSize);
  const test = rest.slice(valSize, valSize + testSize);

  const sorted = (xs: number[]) => Uint32Array.from(xs).sort();
  const mask = (xs: number[]) => {
    const m = new Uint8Array(nNodes);
    for (const i of xs) m[i] = 1;
    return m;
  };

  return {
    train: sorted(train),
    val: sorted(val),
    test: sorted(test),
    trainMask: mask(train),
    valMask: mask(val),
    testMask: mask(test),
  };
}

/** Fetch and decode the bundled dataset. Same-origin; nothing leaves the tab. */
export async function loadCora(): Promise<CoraGraph> {
  const response = await fetch(coraUrl);
  if (!response.ok) {
    throw new Error(`Could not read the bundled Cora dataset (${response.status})`);
  }
  return decodeCora(await response.arrayBuffer());
}
