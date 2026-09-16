import { describe, expect, it, vi } from "vitest";

import {
  buildUnion,
  fetchProteins,
  majorityBaseline,
  makeGraphSplit,
  parseProteinsJsonl,
  PROTEINS_CLASSES,
  type ProteinRow,
} from "./proteins";

// The disjoint union is where this route's silent bug lives: an edge that leaks
// from one graph's node range into the next one's makes two proteins share a
// message-passing neighbourhood, and the model trains perfectly happily on it.
// Nothing downstream — not the loss, not the accuracy, not the gallery — can
// notice. So the union's structure is asserted directly.

/** A triangle, a path of two, and a single isolated node. */
function fixture(): ProteinRow[] {
  return [
    {
      edge_index: [
        [0, 1, 1, 2, 2, 0],
        [1, 0, 2, 1, 0, 2],
      ],
      node_feat: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      y: [1],
      num_nodes: 3,
    },
    {
      edge_index: [
        [0, 1],
        [1, 0],
      ],
      node_feat: [
        [1, 1, 0],
        [0, 1, 1],
      ],
      y: [0],
      num_nodes: 2,
    },
    {
      edge_index: [[], []],
      node_feat: [[2, 2, 2]],
      y: [0],
      num_nodes: 1,
    },
  ];
}

describe("parseProteinsJsonl", () => {
  it("reads one graph per line and tolerates blank lines", () => {
    const text = fixture()
      .map((row) => JSON.stringify(row))
      .join("\n");
    expect(parseProteinsJsonl(`${text}\n\n`)).toHaveLength(3);
  });

  it("refuses an empty file rather than returning an empty dataset", () => {
    expect(() => parseProteinsJsonl("   \n\n")).toThrow(/held no graphs/i);
  });
});

describe("buildUnion", () => {
  const union = buildUnion(fixture());

  it("concatenates the graphs block-diagonally", () => {
    expect(union.nGraphs).toBe(3);
    expect(union.nNodes).toBe(6);
    expect(Array.from(union.graphPtr)).toEqual([0, 3, 5, 6]);
    expect(Array.from(union.graphOf)).toEqual([0, 0, 0, 1, 1, 2]);
  });

  it("keeps every edge inside its own graph's node range", () => {
    // The bug this module exists to prevent. A leaked edge joins two proteins
    // into one neighbourhood and the model trains on it without complaint.
    for (let i = 0; i < union.nNodes; i++) {
      const g = union.graphOf[i];
      for (let e = union.rowPtr[i]; e < union.rowPtr[i + 1]; e++) {
        const j = union.colIdx[e];
        expect(j).toBeGreaterThanOrEqual(union.graphPtr[g]);
        expect(j).toBeLessThan(union.graphPtr[g + 1]);
        expect(union.graphOf[j]).toBe(g);
      }
    }
  });

  it("leaves a symmetric CSR with ascending, self-loop-free rows", () => {
    expect(union.rowPtr[union.nNodes]).toBe(union.colIdx.length);
    const has = (u: number, v: number) => {
      for (let e = union.rowPtr[u]; e < union.rowPtr[u + 1]; e++) {
        if (union.colIdx[e] === v) return true;
      }
      return false;
    };
    for (let u = 0; u < union.nNodes; u++) {
      for (let e = union.rowPtr[u]; e < union.rowPtr[u + 1]; e++) {
        const v = union.colIdx[e];
        expect(v).not.toBe(u);
        expect(has(v, u)).toBe(true);
        if (e > union.rowPtr[u]) expect(v).toBeGreaterThan(union.colIdx[e - 1]);
      }
    }
  });

  it("reports degrees that match the CSR it built", () => {
    for (let i = 0; i < union.nNodes; i++) {
      expect(union.degree[i]).toBe(union.rowPtr[i + 1] - union.rowPtr[i]);
    }
    // The isolated node really is isolated, and that is allowed here — unlike
    // /link-prediction, nothing was removed to make it so.
    expect(union.degree[5]).toBe(0);
  });

  it("copies the features and labels into the union", () => {
    expect(union.nFeat).toBe(3);
    expect(Array.from(union.features.slice(0, 3))).toEqual([1, 0, 0]);
    expect(Array.from(union.features.slice(15, 18))).toEqual([2, 2, 2]);
    expect(Array.from(union.labels)).toEqual([1, 0, 0]);
    expect(union.nClasses).toBe(PROTEINS_CLASSES.length);
  });

  it("refuses a stored self-loop, which the kernel would count twice", () => {
    const rows = fixture();
    rows[1].edge_index = [
      [0, 1, 1],
      [1, 0, 1],
    ];
    expect(() => buildUnion(rows)).toThrow(/self-loop/i);
  });

  it("refuses an asymmetric graph, which breaks the backward transpose", () => {
    const rows = fixture();
    rows[0].edge_index = [
      [0, 1, 2],
      [1, 2, 0],
    ];
    expect(() => buildUnion(rows)).toThrow(/reverse|symmetr/i);
  });

  it("refuses an edge that names a node the graph does not have", () => {
    const rows = fixture();
    rows[1].edge_index = [
      [0, 7],
      [7, 0],
    ];
    expect(() => buildUnion(rows)).toThrow(/outside its/i);
  });

  it("refuses a graph whose feature rows do not match its node count", () => {
    const rows = fixture();
    rows[2].node_feat = [];
    expect(() => buildUnion(rows)).toThrow(/feature rows/i);
  });
});

describe("makeGraphSplit", () => {
  // 100 graphs, 60/40, so the majority baseline is exactly 0.6 everywhere if the
  // split is stratified and something else if it is not.
  const labels = Uint8Array.from(
    Array.from({ length: 100 }, (_, i) => (i < 60 ? 0 : 1)),
  );

  it("covers every graph exactly once", () => {
    const { train, val, test } = makeGraphSplit(labels, 2);
    const all = [...train, ...val, ...test].sort((a, b) => a - b);
    expect(all).toEqual(Array.from({ length: 100 }, (_, i) => i));
  });

  it("keeps the class balance in every part", () => {
    // Unstratified, the test split's majority baseline drifts away from the one
    // the page prints, and the accuracy is then compared against the wrong null.
    const { train, val, test } = makeGraphSplit(labels, 2);
    for (const part of [train, val, test]) {
      const zeros = [...part].filter((g) => labels[g] === 0).length;
      expect(zeros / part.length).toBeCloseTo(0.6, 1);
    }
  });

  it("is the same split for the same seed, and different otherwise", () => {
    const a = makeGraphSplit(labels, 2, { seed: 5 });
    const b = makeGraphSplit(labels, 2, { seed: 5 });
    const c = makeGraphSplit(labels, 2, { seed: 6 });
    expect([...a.test]).toEqual([...b.test]);
    expect([...c.test]).not.toEqual([...a.test]);
  });

  it("refuses a split that would leave no test set", () => {
    expect(() => makeGraphSplit(labels, 2, { trainFrac: 0.95, valFrac: 0.1 })).toThrow(
      /test set/i,
    );
  });
});

describe("majorityBaseline", () => {
  const labels = Uint8Array.from([0, 0, 0, 1, 1, 0, 1, 0]);

  it("is the accuracy of always answering the commonest training class", () => {
    const train = Uint32Array.from([0, 1, 2, 3]); // three 0s, one 1
    const test = Uint32Array.from([4, 5, 6, 7]); // 1, 0, 1, 0
    const { label, accuracy } = majorityBaseline(labels, train, test, 2);
    expect(label).toBe(0);
    expect(accuracy).toBe(0.5);
  });

  it("reads the majority off the training split, not the one being scored", () => {
    // Otherwise the "baseline" is computed with knowledge of the test labels,
    // and it is no longer a baseline.
    const train = Uint32Array.from([3, 4, 6]); // all 1s
    const test = Uint32Array.from([0, 1, 2]); // all 0s
    expect(majorityBaseline(labels, train, test, 2)).toEqual({
      label: 1,
      accuracy: 0,
    });
  });

  it("is zero rather than NaN with nothing to score", () => {
    expect(
      majorityBaseline(labels, Uint32Array.from([0]), new Uint32Array(0), 2)
        .accuracy,
    ).toBe(0);
  });
});

describe("fetchProteins", () => {
  it("parses the published file", async () => {
    const text = fixture().map((r) => JSON.stringify(r)).join("\n");
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(text, { status: 200 })),
    );
    await expect(fetchProteins()).resolves.toHaveLength(3);
    vi.unstubAllGlobals();
  });

  it("says what went wrong rather than parsing an error page", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("nope", { status: 404, statusText: "Not Found" })),
    );
    await expect(fetchProteins()).rejects.toThrow(/could not be fetched.*404/i);
    vi.unstubAllGlobals();
  });
});
