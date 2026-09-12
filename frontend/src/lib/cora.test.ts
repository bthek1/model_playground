import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  CORA_CLASSES,
  decodeCora,
  densifyRowNormalised,
  makeSplit,
} from "./cora";

/**
 * Build a cora.bin by hand, mirroring scripts/prepare-cora.mjs. Writing the
 * encoder twice is the point: if the two ever disagree about an offset, this
 * test fails rather than the app quietly training on a shifted graph.
 */
function encode(g: {
  nNodes: number;
  nFeat: number;
  nClasses: number;
  rowPtr: number[];
  colIdx: number[];
  featRowPtr: number[];
  featColIdx: number[];
  labels: number[];
  magic?: number;
  version?: number;
}): ArrayBuffer {
  const rowPtr = Uint32Array.from(g.rowPtr);
  const colIdx = Uint32Array.from(g.colIdx);
  const featRowPtr = Uint32Array.from(g.featRowPtr);
  const featColIdx = Uint16Array.from(g.featColIdx);
  const labels = Uint8Array.from(g.labels);

  const buffer = new ArrayBuffer(
    28 +
      rowPtr.byteLength +
      colIdx.byteLength +
      featRowPtr.byteLength +
      featColIdx.byteLength +
      labels.byteLength,
  );
  const view = new DataView(buffer);
  const header = [
    g.magic ?? 0x41524f43,
    g.version ?? 1,
    g.nNodes,
    g.nFeat,
    g.nClasses,
    colIdx.length,
    featColIdx.length,
  ];
  header.forEach((value, i) => view.setUint32(i * 4, value, true));

  const bytes = new Uint8Array(buffer);
  let at = 28;
  for (const a of [rowPtr, colIdx, featRowPtr, featColIdx, labels]) {
    bytes.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), at);
    at += a.byteLength;
  }
  return buffer;
}

/** 4 nodes, edges 0–1 and 1–2, node 3 isolated; 3 features, 2 classes. */
const TOY = {
  nNodes: 4,
  nFeat: 3,
  nClasses: 2,
  rowPtr: [0, 1, 3, 4, 4],
  colIdx: [1, 0, 2, 1],
  featRowPtr: [0, 2, 3, 3, 6],
  featColIdx: [0, 2, 1, 0, 1, 2],
  labels: [0, 1, 0, 1],
};

describe("decodeCora", () => {
  it("reads every array back at the right offset", () => {
    const g = decodeCora(encode(TOY));

    expect(g.nNodes).toBe(4);
    expect(g.nFeat).toBe(3);
    expect(g.nClasses).toBe(2);
    expect([...g.rowPtr]).toEqual([0, 1, 3, 4, 4]);
    expect([...g.colIdx]).toEqual([1, 0, 2, 1]);
    expect([...g.labels]).toEqual([0, 1, 0, 1]);
    expect([...g.degree]).toEqual([1, 2, 1, 0]);
  });

  it("row-normalises the features, leaving a wordless node at zero", () => {
    const g = decodeCora(encode(TOY));
    const expected = [
      0.5, 0, 0.5, // node 0: words {0,2}
      0, 1, 0, // node 1: word {1}
      0, 0, 0, // node 2: no words — zero, not NaN
      1 / 3, 1 / 3, 1 / 3, // node 3: all three
    ];
    // Stored as float32, so 1/3 is not the double 1/3.
    expected.forEach((value, i) => expect(g.features[i]).toBeCloseTo(value, 6));
  });

  it("rejects a bad magic number rather than decoding garbage", () => {
    expect(() => decodeCora(encode({ ...TOY, magic: 0xdeadbeef }))).toThrow(
      /bad magic number/,
    );
  });

  it("rejects a format version it does not know", () => {
    expect(() => decodeCora(encode({ ...TOY, version: 2 }))).toThrow(
      /version 2, expected 1/,
    );
  });

  it("rejects a truncated file", () => {
    expect(() => decodeCora(new ArrayBuffer(12))).toThrow(/truncated/);
  });

  it("rejects trailing bytes, which mean the header disagrees with the body", () => {
    const good = encode(TOY);
    const padded = new Uint8Array(good.byteLength + 4);
    padded.set(new Uint8Array(good));
    expect(() => decodeCora(padded.buffer)).toThrow(/trailing bytes/);
  });
});

describe("densifyRowNormalised", () => {
  it("gives every set word in a row the same weight", () => {
    const dense = densifyRowNormalised(
      Uint32Array.from([0, 4]),
      Uint16Array.from([0, 1, 2, 3]),
      1,
      4,
    );
    expect([...dense]).toEqual([0.25, 0.25, 0.25, 0.25]);
  });
});

describe("makeSplit", () => {
  const labels = Uint8Array.from(
    Array.from({ length: 200 }, (_, i) => i % 4),
  );

  it("takes exactly `perClass` labelled nodes from each class", () => {
    const split = makeSplit(labels, 4, { perClass: 5, valSize: 20, testSize: 40 });
    const counts = new Array(4).fill(0);
    for (const i of split.train) counts[labels[i]]++;
    expect(counts).toEqual([5, 5, 5, 5]);
    expect(split.train).toHaveLength(20);
    expect(split.val).toHaveLength(20);
    expect(split.test).toHaveLength(40);
  });

  it("keeps the three sets disjoint", () => {
    const split = makeSplit(labels, 4, { perClass: 5, valSize: 20, testSize: 40 });
    const all = [...split.train, ...split.val, ...split.test];
    expect(new Set(all).size).toBe(all.length);
  });

  it("agrees with its own masks", () => {
    const split = makeSplit(labels, 4, { perClass: 5, valSize: 20, testSize: 40 });
    for (const [set, mask] of [
      [split.train, split.trainMask],
      [split.val, split.valMask],
      [split.test, split.testMask],
    ] as const) {
      expect(mask.reduce((a: number, b: number) => a + b, 0)).toBe(set.length);
      for (const i of set) expect(mask[i]).toBe(1);
    }
  });

  it("is reproducible for a seed, and different for another", () => {
    const opts = { perClass: 5, valSize: 20, testSize: 40 };
    const a = makeSplit(labels, 4, { ...opts, seed: 7 });
    const b = makeSplit(labels, 4, { ...opts, seed: 7 });
    const c = makeSplit(labels, 4, { ...opts, seed: 8 });
    expect([...a.test]).toEqual([...b.test]);
    expect([...a.test]).not.toEqual([...c.test]);
  });

  it("refuses a split the graph cannot supply rather than returning a short one", () => {
    expect(() => makeSplit(labels, 4, { perClass: 60 })).toThrow(/fewer than/);
    expect(() => makeSplit(labels, 4, { perClass: 5, valSize: 500 })).toThrow(
      /nodes left/,
    );
  });
});

/**
 * The committed asset itself. These numbers are the published shape of Cora, so
 * a regenerated or corrupted cora.bin fails here rather than in the browser.
 */
describe("the bundled cora.bin", () => {
  const path = join(dirname(fileURLToPath(import.meta.url)), "data", "cora.bin");
  const file = readFileSync(path);
  const graph = decodeCora(
    file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength),
  );

  it("is the published Cora", () => {
    expect(graph.nNodes).toBe(2708);
    expect(graph.nFeat).toBe(1433);
    expect(graph.nClasses).toBe(CORA_CLASSES.length);
    // 5278 undirected citations, stored once per direction. This is the count
    // PyTorch Geometric's Planetoid loader reports for the same dataset.
    expect(graph.colIdx.length).toBe(10556);
  });

  it("is symmetric, which the aggregation kernel's transpose depends on", () => {
    const has = (u: number, v: number) => {
      for (let e = graph.rowPtr[u]; e < graph.rowPtr[u + 1]; e++) {
        if (graph.colIdx[e] === v) return true;
      }
      return false;
    };
    for (let u = 0; u < graph.nNodes; u++) {
      for (let e = graph.rowPtr[u]; e < graph.rowPtr[u + 1]; e++) {
        expect(has(graph.colIdx[e], u)).toBe(true);
      }
    }
  });

  it("stores no self-loop, because the kernel adds I itself", () => {
    for (let u = 0; u < graph.nNodes; u++) {
      for (let e = graph.rowPtr[u]; e < graph.rowPtr[u + 1]; e++) {
        expect(graph.colIdx[e]).not.toBe(u);
      }
    }
  });

  it("has a monotonic rowPtr that ends at the edge count", () => {
    for (let i = 0; i < graph.nNodes; i++) {
      expect(graph.rowPtr[i + 1]).toBeGreaterThanOrEqual(graph.rowPtr[i]);
    }
    expect(graph.rowPtr[graph.nNodes]).toBe(graph.colIdx.length);
  });

  it("has row-normalised features that sum to 1 per paper", () => {
    for (let i = 0; i < graph.nNodes; i++) {
      let sum = 0;
      for (let f = 0; f < graph.nFeat; f++) sum += graph.features[i * graph.nFeat + f];
      expect(sum).toBeCloseTo(1, 5);
    }
  });

  it("supports the 20-per-class semi-supervised split", () => {
    const split = makeSplit(graph.labels, graph.nClasses);
    expect(split.train).toHaveLength(140);
    expect(split.val).toHaveLength(500);
    expect(split.test).toHaveLength(1000);
  });
});
