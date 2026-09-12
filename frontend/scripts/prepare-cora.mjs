#!/usr/bin/env node
// Turn the canonical Cora release into the compact binary the /graph route
// bundles. Run it from `frontend/`:
//
//     node scripts/prepare-cora.mjs
//
// It writes src/graph/data/cora.bin and prints the header it produced, so the
// committed asset is reproducible rather than a mystery blob. Re-run it only to
// change the format — the dataset itself has not moved since 2020.
//
// Source: the LINQS release of Cora (McCallum et al.), 2708 machine-learning
// papers, 5429 citations, a 1433-word binary bag-of-words per paper and one of
// seven topic labels. Requires `curl` and `tar` on PATH.
//
// Why a binary and not JSON: the feature matrix is 2708 x 1433, which is 15.5 MB
// as dense float32 and unbundleable. It is also 1.27% dense and every stored
// value is 1, so CSR with u16 column indices and no values array is ~100 KB.
// Section 5 of the roadmap (issue #3) is the longer version of this argument.

import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const SOURCE_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz";

const here = dirname(fileURLToPath(import.meta.url));
const OUT_PATH = join(here, "..", "src", "lib", "data", "cora.bin");

/** Header layout, little-endian. Mirrored by decodeCora() in src/lib/cora.ts. */
const MAGIC = 0x41524f43; // "CORA" read as u32 little-endian
const VERSION = 1;
const HEADER_BYTES = 28;

function download(destDir) {
  const tgz = join(destDir, "cora.tgz");
  execFileSync("curl", ["-sSfL", "--max-time", "120", "-o", tgz, SOURCE_URL], {
    stdio: ["ignore", "inherit", "inherit"],
  });
  execFileSync("tar", ["xzf", tgz, "-C", destDir], { stdio: "inherit" });
  return join(destDir, "cora");
}

/**
 * Parse cora.content: `<paper id>\t<1433 binary features>\t<class name>`.
 * Node order is order of appearance, which is what every id below refers to.
 */
function parseContent(text) {
  const lines = text.split("\n").filter((l) => l.trim().length > 0);
  const index = new Map(); // paper id -> node index
  const featRows = []; // per node, the indices of its set features
  const rawLabels = [];

  let nFeat = -1;
  for (const [node, line] of lines.entries()) {
    const parts = line.split(/\s+/);
    if (nFeat === -1) nFeat = parts.length - 2;
    if (parts.length - 2 !== nFeat) {
      throw new Error(`node ${node}: expected ${nFeat} features, got ${parts.length - 2}`);
    }
    index.set(parts[0], node);
    const set = [];
    for (let f = 0; f < nFeat; f++) if (parts[f + 1] !== "0") set.push(f);
    featRows.push(set);
    rawLabels.push(parts[parts.length - 1]);
  }

  // Sorted so the class order is a property of the data, not of the file order.
  const classes = [...new Set(rawLabels)].sort();
  const classIndex = new Map(classes.map((c, i) => [c, i]));
  const labels = Uint8Array.from(rawLabels, (l) => classIndex.get(l));

  return { index, featRows, labels, classes, nFeat, nNodes: lines.length };
}

/**
 * Parse cora.cites (`<cited>\t<citing>`) into an undirected, deduplicated,
 * self-loop-free edge list, then build CSR over both directions.
 *
 * Citation direction is dropped deliberately: message passing here is A + I with
 * A symmetric, and the aggregation kernel adds the self-loop itself, so storing
 * one would double-count every node's own features.
 */
function parseCites(text, index, nNodes) {
  const undirected = new Set();
  let skipped = 0;
  let selfLoops = 0;

  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    const [a, b] = line.split(/\s+/);
    const u = index.get(a);
    const v = index.get(b);
    if (u === undefined || v === undefined) {
      skipped++;
      continue;
    }
    if (u === v) {
      selfLoops++;
      continue;
    }
    undirected.add(u < v ? `${u},${v}` : `${v},${u}`);
  }

  const degree = new Uint32Array(nNodes);
  const pairs = [];
  for (const key of undirected) {
    const [u, v] = key.split(",").map(Number);
    pairs.push([u, v]);
    degree[u]++;
    degree[v]++;
  }

  const rowPtr = new Uint32Array(nNodes + 1);
  for (let i = 0; i < nNodes; i++) rowPtr[i + 1] = rowPtr[i] + degree[i];
  const colIdx = new Uint32Array(rowPtr[nNodes]);
  const cursor = rowPtr.slice(0, nNodes);
  for (const [u, v] of pairs) {
    colIdx[cursor[u]++] = v;
    colIdx[cursor[v]++] = u;
  }
  // Sorted neighbour lists make a reverse-edge lookup a binary search, which is
  // what GAT's backward pass needs (roadmap #3 §3.1, plan phase 5).
  for (let i = 0; i < nNodes; i++) {
    colIdx.subarray(rowPtr[i], rowPtr[i + 1]).sort();
  }

  return { rowPtr, colIdx, undirectedCount: undirected.size, skipped, selfLoops };
}

function buildFeatureCsr(featRows, nNodes) {
  const rowPtr = new Uint32Array(nNodes + 1);
  for (let i = 0; i < nNodes; i++) rowPtr[i + 1] = rowPtr[i] + featRows[i].length;
  const colIdx = new Uint16Array(rowPtr[nNodes]);
  let at = 0;
  for (const row of featRows) for (const f of row) colIdx[at++] = f;
  return { rowPtr, colIdx };
}

function encode({ nNodes, nFeat, nClasses, edges, features, labels }) {
  const nEdges = edges.colIdx.length;
  const featNnz = features.colIdx.length;

  const sizes = [
    HEADER_BYTES,
    edges.rowPtr.byteLength,
    edges.colIdx.byteLength,
    features.rowPtr.byteLength,
    features.colIdx.byteLength,
    labels.byteLength,
  ];
  const buffer = new ArrayBuffer(sizes.reduce((a, b) => a + b, 0));
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  for (const [i, value] of [MAGIC, VERSION, nNodes, nFeat, nClasses, nEdges, featNnz].entries()) {
    view.setUint32(i * 4, value, true);
  }

  let at = HEADER_BYTES;
  for (const array of [edges.rowPtr, edges.colIdx, features.rowPtr, features.colIdx, labels]) {
    bytes.set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength), at);
    at += array.byteLength;
  }
  return buffer;
}

const work = mkdtempSync(join(tmpdir(), "cora-"));
try {
  const dir = download(work);
  const content = parseContent(readFileSync(join(dir, "cora.content"), "utf8"));
  const edges = parseCites(readFileSync(join(dir, "cora.cites"), "utf8"), content.index, content.nNodes);
  const features = buildFeatureCsr(content.featRows, content.nNodes);

  const buffer = encode({
    nNodes: content.nNodes,
    nFeat: content.nFeat,
    nClasses: content.classes.length,
    edges,
    features,
    labels: content.labels,
  });

  mkdirSync(dirname(OUT_PATH), { recursive: true });
  writeFileSync(OUT_PATH, Buffer.from(buffer));

  console.log(`source        ${SOURCE_URL}`);
  console.log(`nodes         ${content.nNodes}`);
  console.log(`features      ${content.nFeat}`);
  console.log(`classes       ${content.classes.length}  ${content.classes.join(", ")}`);
  console.log(`edges         ${edges.undirectedCount} undirected -> ${edges.colIdx.length} directed`);
  console.log(`  skipped     ${edges.skipped} citation(s) naming an unknown paper`);
  console.log(`  self-loops  ${edges.selfLoops} dropped (the kernel adds I itself)`);
  console.log(`feature nnz   ${features.colIdx.length}  (${((features.colIdx.length / (content.nNodes * content.nFeat)) * 100).toFixed(2)}% dense)`);
  console.log(`written       ${OUT_PATH}  ${(buffer.byteLength / 1024).toFixed(1)} KiB`);
} finally {
  rmSync(work, { recursive: true, force: true });
}
