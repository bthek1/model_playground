import { describe, expect, it } from "vitest";

import { buildCsr } from "@/lib/edgeSplit";
import { mulberry32 } from "@/lib/random";

import {
  archScales,
  cpuMatmulFn,
  GnnTrainer,
  isScaledArch,
  makeCpuAggregate,
  makeNeighbourSmoothness,
  prepareInput,
  scaledGatherPropagator,
  type GnnArch,
  type GnnOps,
  type Propagator,
} from "./gnn";
import { AttentionPropagator } from "./gat";
import {
  auc,
  averagePrecision,
  bceWithLogits,
  decodeAndGrad,
  edgeScores,
  fitLinkPredictor,
  sigmoid,
} from "./linkPredictor";

// The decoder is small enough to read and still has a gradient that can be
// half-right: a pair (u, v) contributes to both rows of dOut, and scattering
// into only one produces a falling loss, a rising AUC and a worse model. So the
// load-bearing test here is a finite-difference check of the *whole* path —
// encoder and decoder together, per architecture — exactly as gnn.test.ts pins
// the classification head.

/** A 6-node graph with two triangles joined by one edge. */
function toyGraph() {
  const nNodes = 6;
  const { rowPtr, colIdx } = buildCsr(
    Uint32Array.from([0, 1, 1, 2, 0, 2, 3, 4, 4, 5, 3, 5, 2, 3]),
    nNodes,
  );
  const degree = new Uint32Array(nNodes);
  for (let u = 0; u < nNodes; u++) degree[u] = rowPtr[u + 1] - rowPtr[u];
  return { nNodes, rowPtr, colIdx, degree };
}

function toyOps(arch: GnnArch, seed = 5): GnnOps {
  const { nNodes, rowPtr, colIdx, degree } = toyGraph();
  const smoothness = makeNeighbourSmoothness(rowPtr, colIdx);
  let propagator: (nFeat: number) => Propagator;

  if (isScaledArch(arch)) {
    const { alpha, beta } = archScales(arch, degree);
    const gather = makeCpuAggregate(rowPtr, colIdx, nNodes, alpha, beta);
    propagator = () => scaledGatherPropagator(gather);
  } else {
    const rand = mulberry32(seed);
    propagator = (nFeat) =>
      new AttentionPropagator(rowPtr, colIdx, nNodes, nFeat, rand);
  }
  return { matmul: cpuMatmulFn, propagator, smoothness };
}

function toyInput(nNodes: number, nFeat: number, seed = 9) {
  const rand = mulberry32(seed);
  const x = new Float32Array(nNodes * nFeat);
  for (let i = 0; i < x.length; i++) x[i] = rand() * 2 - 1;
  return prepareInput(x, nNodes, nFeat);
}

describe("bceWithLogits", () => {
  it("agrees with the naive form where the naive form is safe", () => {
    for (const s of [-3, -0.5, 0, 0.5, 3]) {
      for (const y of [0, 1]) {
        const naive = -(
          y * Math.log(sigmoid(s)) +
          (1 - y) * Math.log(1 - sigmoid(s))
        );
        expect(bceWithLogits(s, y)).toBeCloseTo(naive, 10);
      }
    }
  });

  it("stays finite where the naive form blows up", () => {
    // σ(-800) rounds to 0, so -log(σ(s)) is Infinity and one such pair would
    // turn the epoch's mean loss into NaN — a blank chart, not an error.
    expect(Number.isFinite(bceWithLogits(-800, 1))).toBe(true);
    expect(bceWithLogits(-800, 1)).toBeCloseTo(800, 6);
    expect(Number.isFinite(bceWithLogits(800, 0))).toBe(true);
  });
});

describe("edgeScores", () => {
  it("is the dot product of the two endpoints' embeddings", () => {
    const z = Float32Array.from([1, 0, 0, 1, 1, 1]); // 3 nodes, dim 2
    const scores = edgeScores(z, 2, Uint32Array.from([0, 1, 0, 2, 1, 2]));
    expect(Array.from(scores)).toEqual([0, 1, 1]);
  });
});

describe("auc", () => {
  const f = (xs: number[]) => Float32Array.from(xs);

  it("is 1 when every positive outranks every negative", () => {
    expect(auc(f([3, 4, 5]), f([0, 1, 2]))).toBe(1);
  });

  it("is 0 when the ranking is exactly inverted", () => {
    expect(auc(f([0, 1, 2]), f([3, 4, 5]))).toBe(0);
  });

  it("is 0.5 for a scorer that cannot tell them apart", () => {
    // Ties take the average rank; without that this returns 1 or 0 and a
    // collapsed model would look perfect.
    expect(auc(f([1, 1, 1]), f([1, 1, 1]))).toBe(0.5);
  });

  it("is the probability a random positive outranks a random negative", () => {
    // 2 positives, 2 negatives, one inversion out of four pairs.
    expect(auc(f([2, 0.5]), f([1, 0]))).toBeCloseTo(0.75, 10);
  });

  it("is unchanged by a monotone transform of the scores", () => {
    // The decoder's logits and their sigmoids must score identically, which is
    // what makes it safe to rank on whichever is cheaper.
    const pos = f([2.5, -0.5, 1]);
    const neg = f([-1, 0.25, -3]);
    const through = (xs: Float32Array) => Float32Array.from(xs, sigmoid);
    expect(auc(through(pos), through(neg))).toBeCloseTo(auc(pos, neg), 12);
  });

  it("averages to about 0.5 over random scores", () => {
    const rand = mulberry32(4);
    let total = 0;
    const runs = 40;
    for (let r = 0; r < runs; r++) {
      const pos = Float32Array.from({ length: 50 }, () => rand());
      const neg = Float32Array.from({ length: 50 }, () => rand());
      total += auc(pos, neg);
    }
    expect(total / runs).toBeCloseTo(0.5, 1);
  });
});

describe("averagePrecision", () => {
  it("is 1 when every positive comes first", () => {
    expect(averagePrecision(Float32Array.from([5, 4]), Float32Array.from([1])))
      .toBe(1);
  });

  it("falls when a negative is ranked above a positive", () => {
    const ap = averagePrecision(
      Float32Array.from([5, 1]),
      Float32Array.from([4]),
    );
    // Ranks: pos(5) → 1/1, neg(4), pos(1) → 2/3. Mean = 5/6.
    expect(ap).toBeCloseTo((1 + 2 / 3) / 2, 10);
  });
});

describe("decodeAndGrad", () => {
  it("scatters each pair's gradient into both endpoints", () => {
    // The single most valuable assertion in this file after the finite
    // differences: a decoder that updates only `u` still trains.
    const z = Float32Array.from([1, 2, 3, 4]); // 2 nodes, dim 2
    const { dOut } = decodeAndGrad(
      z,
      2,
      2,
      Uint32Array.from([0, 1]),
      new Uint32Array(0),
    );
    const g = sigmoid(1 * 3 + 2 * 4) - 1;
    expect(dOut[0]).toBeCloseTo(g * 3, 6); // ∂/∂z_0 = g · z_1
    expect(dOut[1]).toBeCloseTo(g * 4, 6);
    expect(dOut[2]).toBeCloseTo(g * 1, 6); // ∂/∂z_1 = g · z_0
    expect(dOut[3]).toBeCloseTo(g * 2, 6);
  });

  it("pushes positives together and negatives apart", () => {
    const z = Float32Array.from([1, 0, 1, 0]);
    const pos = decodeAndGrad(z, 2, 2, Uint32Array.from([0, 1]), new Uint32Array(0));
    const neg = decodeAndGrad(z, 2, 2, new Uint32Array(0), Uint32Array.from([0, 1]));
    // Gradient descent moves against the gradient: a positive pair's score must
    // rise, a negative pair's must fall.
    expect(pos.dOut[0]).toBeLessThan(0);
    expect(neg.dOut[0]).toBeGreaterThan(0);
  });

  it("has no gradient and no loss with nothing to score", () => {
    const { loss, dOut } = decodeAndGrad(
      Float32Array.from([1, 1]),
      2,
      1,
      new Uint32Array(0),
      new Uint32Array(0),
    );
    expect(loss).toBe(0);
    expect(Array.from(dOut)).toEqual([0, 0]);
  });
});

describe("the gradient of the whole encoder + decoder", () => {
  const nFeat = 4;
  const dim = 3;
  const pos = Uint32Array.from([0, 1, 3, 4, 2, 3]);
  const neg = Uint32Array.from([0, 4, 1, 5]);

  // Central differences, at a generic point. Two things this needs, both learnt
  // the hard way on gnn.test.ts: the biases must not be zero (a zero bias puts
  // preactivations on ReLU's kink, where a central difference reports half the
  // true gradient), and dropout must be off so the loss is a deterministic
  // function of the parameters.
  async function checkArch(arch: GnnArch) {
    const { nNodes } = toyGraph();
    const trainer = new GnnTrainer(
      toyOps(arch),
      { nNodes, nFeat, nClasses: dim, hidden: 5, layers: 2 },
      3,
    );
    trainer.setInput(toyInput(nNodes, nFeat));
    const jitter = mulberry32(21);
    for (const b of trainer.biases) {
      for (let i = 0; i < b.length; i++) b[i] = jitter() * 0.4 - 0.2;
    }

    const lossAt = async () => {
      const pass = await trainer.forward(0, false);
      return decodeAndGrad(pass.logits, dim, nNodes, pos, neg).loss;
    };

    const pass = await trainer.forward(0, false);
    const { loss, dOut } = decodeAndGrad(pass.logits, dim, nNodes, pos, neg);
    const grads = await trainer.backwardFrom(dOut, loss);

    const eps = 1e-3;
    // A spread of coordinates in both weight matrices, not just the first.
    const probes: [number, number][] = [
      [0, 0],
      [0, 7],
      [1, 2],
      [1, 11],
    ];
    for (const [layer, index] of probes) {
      const w = trainer.weights[layer];
      const original = w[index];

      w[index] = original + eps;
      const up = await lossAt();
      w[index] = original - eps;
      const down = await lossAt();
      w[index] = original;

      const numeric = (up - down) / (2 * eps);
      const analytic = grads.dW[layer][index];
      expect(Math.abs(numeric - analytic)).toBeLessThan(2e-4);
    }
  }

  it("matches finite differences for GCN", () => checkArch("gcn"));
  it("matches finite differences for GraphSAGE", () => checkArch("sage"));
  it("matches finite differences for GIN", () => checkArch("gin"));
  it("matches finite differences for GAT", () => checkArch("gat"));
});

describe("fitLinkPredictor", () => {
  it("drives the loss down and the AUC up on a graph with real structure", async () => {
    const { nNodes, rowPtr, colIdx } = toyGraph();
    const trainer = new GnnTrainer(
      toyOps("gcn"),
      { nNodes, nFeat: 4, nClasses: 3, hidden: 6, layers: 2 },
      2,
    );

    const first: number[] = [];
    const metrics = await fitLinkPredictor(
      trainer,
      toyInput(nNodes, 4),
      {
        trainPos: Uint32Array.from([0, 1, 1, 2, 0, 2, 3, 4, 4, 5, 3, 5]),
        valPos: Uint32Array.from([2, 3]),
        valNeg: Uint32Array.from([0, 4]),
        testPos: Uint32Array.from([2, 3]),
        testNeg: Uint32Array.from([1, 5]),
      },
      {
        arch: "gcn",
        learningRate: 0.05,
        weightDecay: 0,
        dropout: 0,
        epochs: 60,
        fullRowPtr: rowPtr,
        fullColIdx: colIdx,
        seed: 3,
      },
      { onEpoch: (m) => first.push(m.loss) },
    );

    expect(metrics).not.toBeNull();
    expect(first.length).toBe(60);
    expect(first[first.length - 1]).toBeLessThan(first[0]);
    // The two triangles are the only structure there is; the encoder should end
    // up ranking a within-triangle pair above a cross-triangle one.
    expect(metrics!.trainAuc).toBeGreaterThan(0.9);
  });

  it("stops between epochs and returns what it had", async () => {
    const { nNodes, rowPtr, colIdx } = toyGraph();
    const trainer = new GnnTrainer(
      toyOps("gcn"),
      { nNodes, nFeat: 4, nClasses: 3, hidden: 4, layers: 2 },
      2,
    );
    let seen = 0;
    const metrics = await fitLinkPredictor(
      trainer,
      toyInput(nNodes, 4),
      {
        trainPos: Uint32Array.from([0, 1]),
        valPos: Uint32Array.from([2, 3]),
        valNeg: Uint32Array.from([0, 4]),
        testPos: Uint32Array.from([2, 3]),
        testNeg: Uint32Array.from([1, 5]),
      },
      {
        arch: "gcn",
        learningRate: 0.05,
        weightDecay: 0,
        dropout: 0,
        epochs: 100,
        fullRowPtr: rowPtr,
        fullColIdx: colIdx,
      },
      { onEpoch: () => seen++, shouldStop: () => seen >= 3 },
    );
    expect(seen).toBe(3);
    expect(metrics?.epoch).toBe(2);
  });
});
