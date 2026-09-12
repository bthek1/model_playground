import { describe, expect, it } from "vitest";

import { mulberry32 } from "@/lib/random";

import {
  accuracyOn,
  archScales,
  cpuMatmulFn,
  crossEntropyLoss,
  fitGnn,
  type GnnArch,
  GnnTrainer,
  type GnnInput,
  makeCpuAggregate,
  makeNeighbourSmoothness,
  deadFraction,
  predict,
  prepareInput,
  softmaxRows,
} from "./gnn";

/** Two triangles joined by a single bridge — small enough to reason about. */
const N = 6;
const EDGES: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 0],
  [3, 4],
  [4, 5],
  [5, 3],
  [2, 3],
];

function csr(nNodes: number, edges: [number, number][]) {
  const degree = new Uint32Array(nNodes);
  for (const [u, v] of edges) {
    degree[u]++;
    degree[v]++;
  }
  const rowPtr = new Uint32Array(nNodes + 1);
  for (let i = 0; i < nNodes; i++) rowPtr[i + 1] = rowPtr[i] + degree[i];
  const colIdx = new Uint32Array(rowPtr[nNodes]);
  const cursor = rowPtr.slice(0, nNodes);
  for (const [u, v] of edges) {
    colIdx[cursor[u]++] = v;
    colIdx[cursor[v]++] = u;
  }
  return { rowPtr, colIdx, degree };
}

const GRAPH = csr(N, EDGES);
const LABELS = Uint8Array.from([0, 0, 0, 1, 1, 1]);
const TRAIN = Uint32Array.from([0, 1, 3, 4]);
const NFEAT = 4;
const NCLASSES = 2;

function features(seed = 5): Float32Array {
  const rand = mulberry32(seed);
  const x = new Float32Array(N * NFEAT);
  for (let i = 0; i < x.length; i++) x[i] = rand() * 2 - 1;
  return x;
}

function makeTrainer(arch: GnnArch, layers: number, seed = 11) {
  const { alpha, beta } = archScales(arch, GRAPH.degree);
  return new GnnTrainer(
    {
      matmul: cpuMatmulFn,
      aggregate: makeCpuAggregate(GRAPH.rowPtr, GRAPH.colIdx, N, alpha, beta),
      smoothness: makeNeighbourSmoothness(GRAPH.rowPtr, GRAPH.colIdx),
    },
    { nNodes: N, nFeat: NFEAT, nClasses: NCLASSES, hidden: 5, layers },
    seed,
  );
}

function input(seed = 5): GnnInput {
  return prepareInput(features(seed), N, NFEAT);
}

async function lossOf(trainer: GnnTrainer): Promise<number> {
  const forward = await trainer.forward(0, false);
  const probs = softmaxRows(forward.logits, N, NCLASSES);
  return crossEntropyLoss(probs, LABELS, TRAIN, NCLASSES);
}

describe("archScales", () => {
  it("gives GCN the symmetric 1/sqrt(deg+1) on both sides", () => {
    const { alpha, beta } = archScales("gcn", GRAPH.degree);
    // Node 0 has neighbours 1 and 2, so deg+1 = 3.
    expect(alpha[0]).toBeCloseTo(1 / Math.sqrt(3), 6);
    expect([...alpha]).toEqual([...beta]);
  });

  it("gives GraphSAGE a mean on one side only", () => {
    const { alpha, beta } = archScales("sage", GRAPH.degree);
    expect(alpha[0]).toBeCloseTo(1 / 3, 6);
    expect([...beta]).toEqual(new Array(N).fill(1));
  });

  it("gives GIN a plain sum", () => {
    const { alpha, beta } = archScales("gin", GRAPH.degree);
    expect([...alpha]).toEqual(new Array(N).fill(1));
    expect([...beta]).toEqual(new Array(N).fill(1));
  });
});

describe("makeCpuAggregate", () => {
  const ones = new Float32Array(N).fill(1);
  const sum = makeCpuAggregate(GRAPH.rowPtr, GRAPH.colIdx, N, ones, ones);

  it("adds the self-loop the CSR does not store", async () => {
    // One feature, node i holds the value i+1. Sum aggregation over node 0
    // (neighbours 1 and 2) must be 1 + 2 + 3 = 6, not 2 + 3 = 5.
    const out = await sum(Float32Array.from([1, 2, 3, 4, 5, 6]), 1, false);
    expect(out[0]).toBe(6);
  });

  it("keeps an isolated node's own features", async () => {
    const lone = csr(2, []);
    const two = Float32Array.from([1, 1]);
    const agg = makeCpuAggregate(lone.rowPtr, lone.colIdx, 2, two, two);
    expect([...(await agg(Float32Array.from([7, 9]), 1, false))]).toEqual([7, 9]);
  });

  it("applies alpha to the destination and beta to each source", async () => {
    const alpha = Float32Array.from([2, 1, 1, 1, 1, 1]);
    const beta = Float32Array.from([1, 10, 100, 1, 1, 1]);
    const agg = makeCpuAggregate(GRAPH.rowPtr, GRAPH.colIdx, N, alpha, beta);
    const out = await agg(Float32Array.from([1, 2, 3, 4, 5, 6]), 1, false);
    // node 0: alpha_0 * (beta_0*1 + beta_1*2 + beta_2*3) = 2 * (1 + 20 + 300)
    expect(out[0]).toBe(642);
  });

  it("transposes Â when asked, which is what the backward pass needs", async () => {
    // Âᵀ exists only because A+I is symmetric. Check it the honest way: build Â
    // as a dense matrix, column by column, and compare Â[j][i] against Âᵀ[i][j].
    const alpha = Float32Array.from([0.5, 1, 2, 0.25, 1, 3]);
    const beta = Float32Array.from([1, 2, 0.5, 4, 1, 0.2]);
    const agg = makeCpuAggregate(GRAPH.rowPtr, GRAPH.colIdx, N, alpha, beta);

    const dense = async (transposed: boolean) => {
      const m: number[][] = [];
      for (let col = 0; col < N; col++) {
        const e = new Float32Array(N);
        e[col] = 1;
        m.push([...(await agg(e, 1, transposed))]); // m[col][row] = M[row][col]
      }
      return m;
    };

    const forward = await dense(false);
    const backward = await dense(true);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        expect(backward[i][j]).toBeCloseTo(forward[j][i], 6);
      }
    }
  });
});

/**
 * The load-bearing test. A wrong aggregation or a wrong backward pass still
 * produces a falling loss and a plausible accuracy curve — the failure mode this
 * whole route is exposed to — so the gradients are checked against the loss they
 * claim to be the slope of, one parameter at a time.
 */
describe("GnnTrainer gradients", () => {
  const ARCHES: GnnArch[] = ["gcn", "sage", "gin"];

  for (const arch of ARCHES) {
    for (const layers of [1, 2, 3]) {
      it(`match finite differences — ${arch}, ${layers} layer(s)`, async () => {
        const trainer = makeTrainer(arch, layers);
        trainer.setInput(input());

        const forward = await trainer.forward(0, false);
        const grads = await trainer.backward(LABELS, TRAIN, forward);

        const eps = 1e-3;
        for (let l = 0; l < layers; l++) {
          const checks = [
            { params: trainer.weights[l], grad: grads.dW[l], what: `dW[${l}]` },
            { params: trainer.biases[l], grad: grads.db[l], what: `db[${l}]` },
          ];
          for (const { params, grad, what } of checks) {
            // A handful of entries per tensor: enough to catch a transposed or
            // mis-scaled gradient, cheap enough to run on every commit.
            for (let i = 0; i < params.length; i += Math.ceil(params.length / 4)) {
              const original = params[i];
              params[i] = original + eps;
              const up = await lossOf(trainer);
              params[i] = original - eps;
              const down = await lossOf(trainer);
              params[i] = original;

              const numeric = (up - down) / (2 * eps);
              expect(
                Math.abs(numeric - grad[i]),
                `${what}[${i}]: analytic ${grad[i]}, numeric ${numeric}`,
              ).toBeLessThan(2e-3 * Math.max(1, Math.abs(grad[i])));
            }
          }
        }
      });
    }
  }

  /**
   * The same check with dropout switched on, which is the only thing that
   * exercises the masked input transpose. `dW[0] = Xdᵀ · dS`, and Xd is masked
   * through the sparsity pattern rather than by transposing 3.9 M elements — so a
   * wrong `nzT` mapping corrupts exactly this gradient and nothing else.
   *
   * Reproducible because the mask comes from the trainer's own seeded RNG: a
   * freshly constructed trainer with the same seed initialises to the same
   * weights and then draws the same mask, so a perturbed copy sees the identical
   * dropout pattern.
   */
  for (const arch of ["gcn", "sage", "gin"] as GnnArch[]) {
    for (const layers of [1, 2]) {
      it(`match finite differences under dropout — ${arch}, ${layers} layer(s)`, async () => {
        const SEED = 23;
        const DROPOUT = 0.5;
        const inp = input();

        const fresh = () => {
          const t = makeTrainer(arch, layers, SEED);
          t.setInput(inp);
          return t;
        };

        const base = fresh();
        const forward = await base.forward(DROPOUT, true);
        const grads = await base.backward(LABELS, TRAIN, forward);

        const lossAt = async (layer: number, i: number, delta: number) => {
          const t = fresh();
          t.weights[layer][i] += delta;
          const f = await t.forward(DROPOUT, true);
          const probs = softmaxRows(f.logits, N, NCLASSES);
          return crossEntropyLoss(probs, LABELS, TRAIN, NCLASSES);
        };

        // Layer 0 is the one the masked transpose feeds.
        const eps = 1e-3;
        const w = base.weights[0];
        for (let i = 0; i < w.length; i += Math.ceil(w.length / 5)) {
          const numeric =
            ((await lossAt(0, i, eps)) - (await lossAt(0, i, -eps))) / (2 * eps);
          expect(
            Math.abs(numeric - grads.dW[0][i]),
            `dW[0][${i}]: analytic ${grads.dW[0][i]}, numeric ${numeric}`,
          ).toBeLessThan(2e-3 * Math.max(1, Math.abs(grads.dW[0][i])));
        }
      });
    }
  }

  it("gives an unlabelled node no error term of its own", async () => {
    // Nodes 2 and 5 are outside TRAIN. They still take part in the forward pass
    // — that is what makes this semi-supervised — but flipping their labels must
    // not move the loss.
    const trainer = makeTrainer("gcn", 2);
    trainer.setInput(input());

    const before = await lossOf(trainer);
    const flipped = LABELS.slice();
    flipped[2] ^= 1;
    flipped[5] ^= 1;
    const forward = await trainer.forward(0, false);
    const probs = softmaxRows(forward.logits, N, NCLASSES);
    expect(crossEntropyLoss(probs, flipped, TRAIN, NCLASSES)).toBeCloseTo(before, 10);
  });
});

describe("fitGnn", () => {
  it("drives the loss down and fits the toy graph", async () => {
    const trainer = makeTrainer("gcn", 2);
    const seen: number[] = [];
    const split = {
      train: TRAIN,
      val: Uint32Array.from([2]),
      test: Uint32Array.from([5]),
    };
    const final = await fitGnn(
      trainer,
      input(),
      LABELS,
      split,
      {
        learningRate: 0.05,
        weightDecay: 0,
        dropout: 0,
        epochs: 120,
        arch: "gcn",
      },
      { onMetrics: (m) => seen.push(m.loss) },
    );

    expect(seen).toHaveLength(120);
    expect(seen[seen.length - 1]).toBeLessThan(seen[0]);
    expect(final?.trainAcc).toBe(1);
  });

  it("stops between epochs when asked", async () => {
    const trainer = makeTrainer("gcn", 2);
    let epochs = 0;
    await fitGnn(
      trainer,
      input(),
      LABELS,
      { train: TRAIN, val: TRAIN, test: TRAIN },
      { learningRate: 0.05, weightDecay: 0, dropout: 0, epochs: 50, arch: "gcn" },
      {
        onMetrics: () => epochs++,
        shouldStop: () => epochs >= 3,
      },
    );
    expect(epochs).toBe(3);
  });

  it("applies dropout only while training", async () => {
    const trainer = makeTrainer("gcn", 3);
    trainer.setInput(input());
    const a = await trainer.forward(0.5, false);
    const b = await trainer.forward(0.5, false);
    expect([...a.logits]).toEqual([...b.logits]);

    const dropped = await trainer.forward(0.9, true);
    expect([...dropped.logits]).not.toEqual([...a.logits]);
  });

  it("drops the input features too, via the sparsity pattern", async () => {
    // A 1-layer net has only layer 0, so the only thing dropout can touch is the
    // input. Before the sparse pattern existed the input was never dropped and
    // this case was a no-op.
    const trainer = makeTrainer("gcn", 1);
    trainer.setInput(input());
    const plain = await trainer.forward(0, false);
    const dropped = await trainer.forward(0.9, true);
    expect([...dropped.logits]).not.toEqual([...plain.logits]);
  });

  it("leaves the input alone when no sparsity pattern is supplied", async () => {
    const trainer = makeTrainer("gcn", 1);
    const x = features();
    const xT = new Float32Array(N * NFEAT);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < NFEAT; j++) xT[j * N + i] = x[i * NFEAT + j];
    }
    trainer.setInput({ x, xT });
    const plain = await trainer.forward(0, false);
    const dropped = await trainer.forward(1, true);
    expect([...dropped.logits]).toEqual([...plain.logits]);
  });
});

describe("prepareInput", () => {
  it("locates every nonzero in both layouts", () => {
    const x = Float32Array.from([0, 2, 0, 3, 0, 0]); // 2 rows, 3 cols
    const prepared = prepareInput(x, 2, 3);
    expect([...prepared.xT]).toEqual([0, 3, 2, 0, 0, 0]);
    expect([...prepared.pattern!.nz]).toEqual([1, 3]);
    // x[0][1] -> xT[1][0] = 1*2 + 0 = 2;  x[1][0] -> xT[0][1] = 0*2 + 1 = 1
    expect([...prepared.pattern!.nzT]).toEqual([2, 1]);
    for (let e = 0; e < 2; e++) {
      expect(prepared.xT[prepared.pattern!.nzT[e]]).toBe(x[prepared.pattern!.nz[e]]);
    }
  });
});

describe("makeNeighbourSmoothness", () => {
  // Path graph 0–1–2, so the edges are (0,1) and (1,2) only.
  const path = csr(3, [
    [0, 1],
    [1, 2],
  ]);
  const smooth = makeNeighbourSmoothness(path.rowPtr, path.colIdx);

  it("is 1 when every neighbour points the same way", () => {
    expect(smooth(Float32Array.from([1, 0, 2, 0, 0.5, 0]), 2)).toBeCloseTo(1, 5);
  });

  it("is −1 when neighbours are opposed", () => {
    expect(smooth(Float32Array.from([1, 0, -1, 0, 1, 0]), 2)).toBeCloseTo(-1, 5);
  });

  it("is 0 when neighbours are orthogonal", () => {
    expect(smooth(Float32Array.from([1, 0, 0, 1, 1, 0]), 2)).toBeCloseTo(0, 5);
  });

  it("counts each undirected edge once", () => {
    // Nodes 0 and 2 are identical and 1 is orthogonal to both, so the two edges
    // score 0 each. A double-counted edge would still give 0 — so also check a
    // graph where the two edges differ.
    const value = smooth(Float32Array.from([1, 0, 0, 1, 1, 1]), 2);
    // edge(0,1): cos = 0.  edge(1,2): cos = 1/sqrt(2).
    expect(value).toBeCloseTo((0 + 1 / Math.SQRT2) / 2, 5);
  });

  it("skips edges touching a dead row rather than scoring them zero", () => {
    // Node 2 is all-zero: edge (1,2) has no direction, so only (0,1) counts.
    expect(smooth(Float32Array.from([1, 0, 1, 0, 0, 0]), 2)).toBeCloseTo(1, 5);
    // Everything dead — there is nothing to report, not "perfectly unsmoothed".
    expect(smooth(new Float32Array(6), 2)).toBe(0);
  });
});

describe("deadFraction", () => {
  it("counts rows that are entirely zero", () => {
    expect(deadFraction(Float32Array.from([1, 0, 0, 0, 0, 3]), 3, 2)).toBeCloseTo(1 / 3);
  });

  it("counts a NaN row as dead rather than alive", () => {
    // An overflowed GIN produces Infinity, then NaN. Treating that as a live
    // direction would report a destroyed network as a healthy one.
    expect(deadFraction(Float32Array.from([NaN, NaN, 1, 0]), 2, 2)).toBe(0.5);
  });

  it("is 0 for an empty representation", () => {
    expect(deadFraction(new Float32Array(0), 0, 4)).toBe(0);
  });
});

describe("predict / accuracyOn", () => {
  it("scores only the nodes in the index set", () => {
    const logits = Float32Array.from([2, 1, 0, 3, 5, 1]);
    const pred = predict(logits, 3, 2);
    expect([...pred]).toEqual([0, 1, 0]);
    expect(accuracyOn(pred, Uint8Array.from([0, 1, 1]), Uint32Array.from([0, 1]))).toBe(1);
    expect(accuracyOn(pred, Uint8Array.from([0, 1, 1]), Uint32Array.from([2]))).toBe(0);
  });
});
