import { describe, expect, it } from "vitest";

import { buildUnion, makeGraphSplit, type ProteinRow } from "@/lib/proteins";
import { mulberry32 } from "@/lib/random";

import { AttentionPropagator } from "./gat";
import {
  archScales,
  cpuMatmulFn,
  GnnTrainer,
  isScaledArch,
  makeCpuAggregate,
  makeNeighbourSmoothness,
  prepareInput,
  scaledGatherPropagator,
  softmaxRows,
  crossEntropyLoss,
  type GnnArch,
  type GnnOps,
  type Propagator,
} from "./gnn";
import {
  fitGraphClassifier,
  graphAccuracy,
  poolBackward,
  poolForward,
  predictGraphs,
  type ReadoutMode,
} from "./graphPool";

// The readout is four lines of arithmetic with one factor that can be silently
// wrong: the `1/n_g` that makes a mean a mean. Drop it from the backward pass and
// the model still trains — every graph just gets a learning rate proportional to
// its size — and no loss curve or accuracy shows it. So the load-bearing test
// here is a finite-difference check through pool *and* encoder, per architecture
// and per mode, exactly as gnn.test.ts and linkPredictor.test.ts pin their heads.

/** Three graphs: a triangle, a pair, and a lone node. 6 nodes in total. */
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
    { edge_index: [[], []], node_feat: [[1, 1, 1]], y: [1], num_nodes: 1 },
  ];
}

const GRAPH_PTR = Uint32Array.from([0, 3, 5, 6]);

function toyOps(union: ReturnType<typeof buildUnion>, arch: GnnArch): GnnOps {
  const { rowPtr, colIdx, nNodes, degree } = union;
  let propagator: (nFeat: number) => Propagator;
  if (isScaledArch(arch)) {
    const { alpha, beta } = archScales(arch, degree);
    const gather = makeCpuAggregate(rowPtr, colIdx, nNodes, alpha, beta);
    propagator = () => scaledGatherPropagator(gather);
  } else {
    const rand = mulberry32(5);
    propagator = (nFeat) =>
      new AttentionPropagator(rowPtr, colIdx, nNodes, nFeat, rand);
  }
  return {
    matmul: cpuMatmulFn,
    propagator,
    smoothness: makeNeighbourSmoothness(rowPtr, colIdx),
  };
}

describe("poolForward", () => {
  // 6 nodes, dim 2. Node i holds [i, 10i].
  const h = Float32Array.from([0, 0, 1, 10, 2, 20, 3, 30, 4, 40, 5, 50]);

  it("averages over each graph's own nodes", () => {
    const out = poolForward(h, 2, GRAPH_PTR, "mean");
    expect(Array.from(out)).toEqual([1, 10, 3.5, 35, 5, 50]);
  });

  it("sums when asked to sum", () => {
    const out = poolForward(h, 2, GRAPH_PTR, "sum");
    expect(Array.from(out)).toEqual([3, 30, 7, 70, 5, 50]);
  });

  it("never mixes one graph's nodes into another's row", () => {
    // The disjoint union's whole premise. Zero everything but graph 1's nodes;
    // only graph 1's row may be nonzero.
    const only = new Float32Array(12);
    only[3 * 2] = 1;
    only[4 * 2] = 1;
    const out = poolForward(only, 2, GRAPH_PTR, "sum");
    expect(Array.from(out)).toEqual([0, 0, 2, 0, 0, 0]);
  });

  it("gives an empty graph a zero row rather than a NaN one", () => {
    const ptr = Uint32Array.from([0, 0, 1]);
    const out = poolForward(Float32Array.from([7]), 1, ptr, "mean");
    expect(Array.from(out)).toEqual([0, 7]);
  });
});

describe("poolBackward", () => {
  it("scatters a sum's gradient whole to every node", () => {
    const d = Float32Array.from([1, 2, 3]); // dim 1, three graphs
    expect(Array.from(poolBackward(d, 1, GRAPH_PTR, "sum"))).toEqual([
      1, 1, 1, 2, 2, 3,
    ]);
  });

  it("divides a mean's gradient by the graph's size", () => {
    // The factor that is invisible in a loss curve.
    const d = Float32Array.from([3, 2, 5]);
    expect(Array.from(poolBackward(d, 1, GRAPH_PTR, "mean"))).toEqual([
      1, 1, 1, 1, 1, 5,
    ]);
  });

  it("is the transpose of the forward pass", () => {
    // <pool(h), d> must equal <h, poolᵀ(d)> for any h and d — the definition of
    // an adjoint, and a check that does not depend on the model at all.
    const rand = mulberry32(3);
    for (const mode of ["mean", "sum"] as ReadoutMode[]) {
      const h = Float32Array.from({ length: 12 }, () => rand() * 2 - 1);
      const d = Float32Array.from({ length: 6 }, () => rand() * 2 - 1);
      const forward = poolForward(h, 2, GRAPH_PTR, mode);
      const back = poolBackward(d, 2, GRAPH_PTR, mode);
      let lhs = 0;
      for (let i = 0; i < forward.length; i++) lhs += forward[i] * d[i];
      let rhs = 0;
      for (let i = 0; i < h.length; i++) rhs += h[i] * back[i];
      // Six places, not ten: both sides are accumulated in Float32Array, whose
      // epsilon is ~1.2e-7, so the identity can only hold to f32 precision.
      expect(lhs).toBeCloseTo(rhs, 6);
    }
  });
});

describe("predictGraphs and graphAccuracy", () => {
  it("takes the argmax per graph", () => {
    const logits = Float32Array.from([1, 2, 5, 3, 0, 0]);
    expect(Array.from(predictGraphs(logits, 3, 2))).toEqual([1, 0, 0]);
  });

  it("scores only the graphs it was given", () => {
    const predicted = Uint8Array.from([1, 0, 1, 1]);
    const labels = Uint8Array.from([1, 1, 1, 0]);
    expect(graphAccuracy(predicted, labels, Uint32Array.from([0, 2]))).toBe(1);
    expect(graphAccuracy(predicted, labels, Uint32Array.from([1, 3]))).toBe(0);
    expect(graphAccuracy(predicted, labels, new Uint32Array(0))).toBe(0);
  });
});

describe("the gradient of the whole encoder + readout", () => {
  const union = buildUnion(fixture());

  async function checkArch(arch: GnnArch, mode: ReadoutMode) {
    const trainer = new GnnTrainer(
      toyOps(union, arch),
      {
        nNodes: union.nNodes,
        nFeat: union.nFeat,
        nClasses: union.nClasses,
        hidden: 5,
        layers: 2,
      },
      3,
    );
    trainer.setInput(
      prepareInput(union.features, union.nNodes, union.nFeat),
    );
    // A generic point: zero biases put preactivations exactly on ReLU's kink,
    // where a central difference reports half the true gradient.
    const jitter = mulberry32(17);
    for (const b of trainer.biases) {
      for (let i = 0; i < b.length; i++) b[i] = jitter() * 0.4 - 0.2;
    }

    const trainIdx = Uint32Array.from([0, 1, 2]);
    const lossAt = async () => {
      const pass = await trainer.forward(0, false);
      const pooled = poolForward(pass.logits, union.nClasses, union.graphPtr, mode);
      const probs = softmaxRows(pooled, union.nGraphs, union.nClasses);
      return crossEntropyLoss(probs, union.labels, trainIdx, union.nClasses);
    };

    const pass = await trainer.forward(0, false);
    const pooled = poolForward(pass.logits, union.nClasses, union.graphPtr, mode);
    const probs = softmaxRows(pooled, union.nGraphs, union.nClasses);
    const loss = crossEntropyLoss(probs, union.labels, trainIdx, union.nClasses);
    const dPooled = new Float32Array(union.nGraphs * union.nClasses);
    const scale = 1 / trainIdx.length;
    for (const g of trainIdx) {
      const row = g * union.nClasses;
      for (let c = 0; c < union.nClasses; c++) {
        dPooled[row + c] =
          (probs[row + c] - (c === union.labels[g] ? 1 : 0)) * scale;
      }
    }
    const grads = await trainer.backwardFrom(
      poolBackward(dPooled, union.nClasses, union.graphPtr, mode),
      loss,
    );

    const eps = 1e-3;
    for (const [layer, index] of [
      [0, 0],
      [0, 7],
      [1, 2],
      [1, 6],
    ] as [number, number][]) {
      const w = trainer.weights[layer];
      const original = w[index];
      w[index] = original + eps;
      const up = await lossAt();
      w[index] = original - eps;
      const down = await lossAt();
      w[index] = original;
      expect(Math.abs((up - down) / (2 * eps) - grads.dW[layer][index])).toBeLessThan(
        2e-4,
      );
    }
  }

  for (const arch of ["gcn", "sage", "gin", "gat"] as GnnArch[]) {
    for (const mode of ["mean", "sum"] as ReadoutMode[]) {
      it(`matches finite differences for ${arch} with a ${mode} readout`, () =>
        checkArch(arch, mode));
    }
  }
});

describe("fitGraphClassifier", () => {
  const union = buildUnion(fixture());
  const split = {
    train: Uint32Array.from([0, 1, 2]),
    val: Uint32Array.from([0]),
    test: Uint32Array.from([1, 2]),
  };

  const options = {
    arch: "gcn" as const,
    learningRate: 0.05,
    weightDecay: 0,
    dropout: 0,
    epochs: 40,
    readout: "mean" as ReadoutMode,
  };

  it("drives the loss down and carries the baseline through", async () => {
    const trainer = new GnnTrainer(
      toyOps(union, "gcn"),
      {
        nNodes: union.nNodes,
        nFeat: union.nFeat,
        nClasses: union.nClasses,
        hidden: 6,
        layers: 2,
      },
      2,
    );
    trainer.setInput(prepareInput(union.features, union.nNodes, union.nFeat));

    const losses: number[] = [];
    const metrics = await fitGraphClassifier(
      trainer,
      union.graphPtr,
      union.labels,
      split,
      0.5,
      options,
      { onEpoch: (m) => losses.push(m.loss) },
    );

    expect(losses).toHaveLength(40);
    expect(losses[losses.length - 1]).toBeLessThan(losses[0]);
    expect(metrics?.trainAcc).toBe(1);
    // The null model travels with the numbers rather than being recomputed by
    // the page, so the two can never come from different splits.
    expect(metrics?.baselineAcc).toBe(0.5);
  });

  it("stops between epochs and returns what it had", async () => {
    const trainer = new GnnTrainer(
      toyOps(union, "gcn"),
      {
        nNodes: union.nNodes,
        nFeat: union.nFeat,
        nClasses: union.nClasses,
        hidden: 4,
        layers: 2,
      },
      2,
    );
    trainer.setInput(prepareInput(union.features, union.nNodes, union.nFeat));
    let seen = 0;
    const metrics = await fitGraphClassifier(
      trainer,
      union.graphPtr,
      union.labels,
      split,
      0.5,
      { ...options, epochs: 100 },
      { onEpoch: () => seen++, shouldStop: () => seen >= 3 },
    );
    expect(seen).toBe(3);
    expect(metrics?.epoch).toBe(2);
  });

  it("hands the page a prediction for every graph", async () => {
    const trainer = new GnnTrainer(
      toyOps(union, "gin"),
      {
        nNodes: union.nNodes,
        nFeat: union.nFeat,
        nClasses: union.nClasses,
        hidden: 4,
        layers: 2,
      },
      1,
    );
    trainer.setInput(prepareInput(union.features, union.nNodes, union.nFeat));
    let predicted: Uint8Array | null = null;
    await fitGraphClassifier(
      trainer,
      union.graphPtr,
      union.labels,
      split,
      0.5,
      { ...options, arch: "gin", epochs: 3 },
      { onEpoch: (_m, p) => (predicted = p) },
    );
    expect(predicted).not.toBeNull();
    expect(predicted!).toHaveLength(union.nGraphs);
  });
});

describe("makeGraphSplit over the fixture", () => {
  it("still covers every graph", () => {
    const union = buildUnion(fixture());
    const split = makeGraphSplit(union.labels, union.nClasses, {
      trainFrac: 0.5,
      valFrac: 0,
    });
    const all = [...split.train, ...split.val, ...split.test].sort();
    expect(all).toHaveLength(union.nGraphs);
  });
});
