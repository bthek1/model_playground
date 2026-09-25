import { describe, expect, it } from "vitest";

import { mulberry32 } from "@/lib/random";
import {
  applyBinning,
  binFeatures,
  fitBoostedClassifier,
  fitBoostedRegressor,
  fitForestClassifier,
  fitGiniTree,
  fitNewtonTree,
  leafOffset,
  predictProba,
  predictValue,
} from "./trees";

const params = { maxDepth: 6, minLeaf: 1, featureFraction: 1 };
const allRows = (n: number) => Int32Array.from({ length: n }, (_, i) => i);

describe("binFeatures", () => {
  it("gives a binary indicator exactly two bins", () => {
    // Equal-width binning would spend 63 cut points here and 62 of them split
    // nothing; quantile binning on the distinct values spends one.
    const x = Float32Array.from([0, 1, 0, 1, 1, 0]);
    const b = binFeatures(x, 6, 1);
    expect(b.thresholds[0].length).toBe(1);
    expect(Array.from(b.bins)).toEqual([0, 1, 0, 1, 1, 0]);
  });

  it("does not let one outlier collapse the rest of the column", () => {
    const values = [...Array.from({ length: 99 }, (_, i) => i), 1e6];
    const b = binFeatures(Float32Array.from(values), 100, 1);
    // The 99 ordinary values still land in many distinct bins rather than all
    // in bin 0 with the outlier alone at the top.
    const used = new Set(Array.from(b.bins).slice(0, 99));
    expect(used.size).toBeGreaterThan(20);
  });

  it("applies an existing binning to unseen rows", () => {
    const b = binFeatures(Float32Array.from([0, 10, 20, 30]), 4, 1);
    const unseen = applyBinning(b, Float32Array.from([-5, 100]), 2);
    expect(unseen[0]).toBe(0);
    expect(unseen[1]).toBe(b.thresholds[0].length);
  });
});

describe("fitGiniTree", () => {
  it("fits a separable two-column problem exactly", () => {
    // y = 1 iff x0 > 0. A depth-1 tree can say that, and a correct Gini sweep
    // finds it on the first split.
    const n = 200;
    const x = new Float32Array(n * 2);
    const y = new Uint8Array(n);
    const rand = mulberry32(3);
    for (let i = 0; i < n; i++) {
      const a = rand() * 2 - 1;
      x[i * 2] = a;
      x[i * 2 + 1] = rand();
      y[i] = a > 0 ? 1 : 0;
    }
    const binning = binFeatures(x, n, 2);
    const tree = fitGiniTree(binning, allRows(n), y, 2, params, mulberry32(1));
    expect(tree.feature[0]).toBe(0);
    let correct = 0;
    for (let i = 0; i < n; i++) {
      const off = leafOffset(tree, binning.bins, i * 2);
      const predicted = tree.value[off + 1] > tree.value[off] ? 1 : 0;
      if (predicted === y[i]) correct++;
    }
    expect(correct).toBe(n);
  });

  it("respects the depth cap", () => {
    const n = 64;
    const x = new Float32Array(n);
    const y = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      x[i] = i;
      y[i] = i % 2;
    }
    const binning = binFeatures(x, n, 1);
    const tree = fitGiniTree(
      binning,
      allRows(n),
      y,
      2,
      { ...params, maxDepth: 2 },
      mulberry32(1),
    );
    expect(tree.depth).toBeLessThanOrEqual(2);
  });

  it("never builds a leaf smaller than the minimum", () => {
    const n = 40;
    const x = Float32Array.from({ length: n }, (_, i) => i);
    const y = Uint8Array.from({ length: n }, (_, i) => (i < 3 ? 1 : 0));
    const binning = binFeatures(x, n, 1);
    const tree = fitGiniTree(
      binning,
      allRows(n),
      y,
      2,
      { ...params, minLeaf: 10 },
      mulberry32(1),
    );
    // The most informative cut sits at row 3 and is forbidden, but Gini still
    // finds gain further along — so the assertion is occupancy, not "it refused
    // to split". A leaf built from three rows is a memory of three rows.
    const occupancy = new Map<number, number>();
    for (let i = 0; i < n; i++) {
      const off = leafOffset(tree, binning.bins, i);
      occupancy.set(off, (occupancy.get(off) ?? 0) + 1);
    }
    for (const count of occupancy.values()) expect(count).toBeGreaterThanOrEqual(10);
  });

  it("makes a leaf of a pure node rather than splitting further", () => {
    const x = Float32Array.from([0, 1, 2, 3]);
    const y = new Uint8Array([1, 1, 1, 1]);
    const binning = binFeatures(x, 4, 1);
    const tree = fitGiniTree(binning, allRows(4), y, 2, params, mulberry32(1));
    expect(tree.nodes).toBe(1);
    expect(tree.value[1]).toBe(1);
  });
});

describe("fitNewtonTree", () => {
  it("picks the split a hand-computed variance reduction picks", () => {
    // y = [1,1,5,5]; the only split that reduces variance to zero is between
    // rows 1 and 2. With g = -y and h = 1 the leaf becomes the group mean and
    // the gain becomes variance reduction exactly.
    const x = Float32Array.from([0, 1, 2, 3]);
    const y = Float32Array.from([1, 1, 5, 5]);
    const g = Float32Array.from(y, (v) => -v);
    const h = new Float32Array(4).fill(1);
    const binning = binFeatures(x, 4, 1);
    const tree = fitNewtonTree(
      binning,
      allRows(4),
      g,
      h,
      { ...params, lambda: 0, maxDepth: 1 },
      mulberry32(1),
    );
    expect(tree.feature[0]).toBe(0);
    expect(tree.threshold[0]).toBeCloseTo(1);
    expect(tree.value[tree.left[0]]).toBeCloseTo(1);
    expect(tree.value[tree.right[0]]).toBeCloseTo(5);
  });

  it("gives a constant target a depth-0 tree", () => {
    const x = Float32Array.from([0, 1, 2, 3]);
    const g = new Float32Array(4).fill(-7);
    const h = new Float32Array(4).fill(1);
    const binning = binFeatures(x, 4, 1);
    const tree = fitNewtonTree(binning, allRows(4), g, h, { ...params, lambda: 0 }, mulberry32(1));
    expect(tree.nodes).toBe(1);
    expect(tree.value[0]).toBeCloseTo(7);
  });
});

describe("ensembles", () => {
  /** A problem no straight line can separate: y = 1 iff exactly one of x0, x1 is high. */
  function xor(n: number, seed = 5) {
    const rand = mulberry32(seed);
    const x = new Float32Array(n * 2);
    const y = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const a = rand();
      const b = rand();
      x[i * 2] = a;
      x[i * 2 + 1] = b;
      y[i] = (a > 0.5) !== (b > 0.5) ? 1 : 0;
    }
    return { x, y };
  }

  it("a forest beats the class prior on a problem with an interaction", () => {
    const n = 400;
    const { x, y } = xor(n);
    const binning = binFeatures(x, n, 2);
    const forest = fitForestClassifier(binning, allRows(n), y, 2, 25, params, 1);
    const probs = predictProba(forest, binning.bins, n, 2);
    let correct = 0;
    for (let i = 0; i < n; i++) {
      if ((probs[i * 2 + 1] > 0.5 ? 1 : 0) === y[i]) correct++;
    }
    expect(correct / n).toBeGreaterThan(0.9);
  });

  it("boosting's training loss falls monotonically on a fixed seed", () => {
    const n = 300;
    const { x, y } = xor(n, 9);
    const binning = binFeatures(x, n, 2);
    const losses: number[] = [];
    fitBoostedClassifier(binning, allRows(n), y, 2, 30, 0.2, params, 1, {
      onProgress: (_d, _t, loss) => {
        if (loss != null) losses.push(loss);
      },
    });
    expect(losses.length).toBe(30);
    for (let i = 1; i < losses.length; i++) {
      expect(losses[i]).toBeLessThanOrEqual(losses[i - 1] + 1e-6);
    }
  });

  it("stops when asked and keeps what it built", () => {
    const n = 200;
    const { x, y } = xor(n);
    const binning = binFeatures(x, n, 2);
    let built = 0;
    const forest = fitForestClassifier(binning, allRows(n), y, 2, 50, params, 1, {
      onProgress: () => built++,
      shouldStop: () => built >= 5,
    });
    expect(forest.trees.length).toBe(5);
  });

  it("boosted regression converges toward the target", () => {
    const n = 200;
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const rand = mulberry32(2);
    for (let i = 0; i < n; i++) {
      const v = rand() * 10;
      x[i] = v;
      y[i] = v * v;
    }
    const binning = binFeatures(x, n, 1);
    const model = fitBoostedRegressor(binning, allRows(n), y, 80, 0.2, params, 1);
    const pred = predictValue(model, binning.bins, n, 1);
    let sse = 0;
    let sst = 0;
    const mean = y.reduce((a, b) => a + b, 0) / n;
    for (let i = 0; i < n; i++) {
      sse += (pred[i] - y[i]) ** 2;
      sst += (y[i] - mean) ** 2;
    }
    expect(1 - sse / sst).toBeGreaterThan(0.98);
  });
});
