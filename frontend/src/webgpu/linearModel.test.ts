import { describe, expect, it } from "vitest";

import {
  cpuMatmul,
  LinearTrainer,
  type MatmulFn,
  type TrainDataset,
} from "./linearModel";

const matmul: MatmulFn = (a, b, m, k, n) =>
  Promise.resolve(cpuMatmul(a, b, m, k, n));

// A tiny, linearly-separable 2-class / 2-feature problem: label = (x0 > x1).
function toyDataset(n: number, seed = 1): TrainDataset {
  let s = seed;
  const rand = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  const D = 2;
  const x = new Float32Array(n * D);
  const y = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const a = rand();
    const b = rand();
    x[i * D] = a;
    x[i * D + 1] = b;
    y[i] = a > b ? 1 : 0;
  }
  return { xTrain: x, yTrain: y, xTest: x, yTest: y, trainSize: n, testSize: n };
}

describe("LinearTrainer math", () => {
  it("softmax rows are non-negative and sum to 1", () => {
    const probs = LinearTrainer.softmaxRows(
      new Float32Array([1, 2, 3, 0, 0, 0]),
      2,
      3,
    );
    for (let r = 0; r < 2; r++) {
      const sum = probs[r * 3] + probs[r * 3 + 1] + probs[r * 3 + 2];
      expect(sum).toBeCloseTo(1, 6);
    }
    expect(probs[3]).toBeCloseTo(1 / 3, 6);
  });

  it("a single SGD step reduces the batch loss", async () => {
    const data = toyDataset(32);
    const trainer = new LinearTrainer(matmul, { inputDim: 2, numClasses: 2 });
    const before = await trainer.computeGradients(
      data.xTrain,
      data.yTrain,
      data.trainSize,
    );
    await trainer.trainStep(data.xTrain, data.yTrain, data.trainSize, 0.5);
    const after = await trainer.computeGradients(
      data.xTrain,
      data.yTrain,
      data.trainSize,
    );
    expect(after.loss).toBeLessThan(before.loss);
  });

  it("learns the separable toy problem to high accuracy", async () => {
    const data = toyDataset(200);
    const trainer = new LinearTrainer(matmul, { inputDim: 2, numClasses: 2 });
    await trainer.fit(data, { learningRate: 0.5, batchSize: 32, epochs: 60 });
    const acc = await trainer.accuracy(data.xTest, data.yTest, data.testSize);
    expect(acc).toBeGreaterThan(0.9);
  });

  it("analytic gradients match finite differences", async () => {
    const D = 4;
    const C = 3;
    const B = 6;
    const data = toyDataset(1); // reuse RNG shape only
    void data;
    // Build a small random batch.
    let s = 7;
    const rand = () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff - 0.5;
    };
    const x = new Float32Array(B * D);
    for (let i = 0; i < x.length; i++) x[i] = rand();
    const y = new Uint8Array(B);
    for (let i = 0; i < B; i++) y[i] = i % C;

    const trainer = new LinearTrainer(matmul, { inputDim: D, numClasses: C });
    // Perturb weights off zero so the gradient is non-trivial.
    for (let i = 0; i < trainer.weights.length; i++) trainer.weights[i] = rand();
    for (let c = 0; c < C; c++) trainer.bias[c] = rand();

    const { dW, db } = await trainer.computeGradients(x, y, B);

    const loss = async () => {
      const logits = await trainer.forward(x, B);
      const probs = LinearTrainer.softmaxRows(logits, B, C);
      return LinearTrainer.crossEntropy(probs, y, B, C).loss;
    };
    const eps = 1e-3;

    // Check a few weight gradients.
    for (const idx of [0, 5, 11]) {
      const orig = trainer.weights[idx];
      trainer.weights[idx] = orig + eps;
      const lp = await loss();
      trainer.weights[idx] = orig - eps;
      const lm = await loss();
      trainer.weights[idx] = orig;
      const numeric = (lp - lm) / (2 * eps);
      expect(Math.abs(numeric - dW[idx])).toBeLessThan(1e-3);
    }

    // Check a bias gradient.
    const origB = trainer.bias[1];
    trainer.bias[1] = origB + eps;
    const lp = await loss();
    trainer.bias[1] = origB - eps;
    const lm = await loss();
    trainer.bias[1] = origB;
    expect(Math.abs((lp - lm) / (2 * eps) - db[1])).toBeLessThan(1e-3);
  });
});
