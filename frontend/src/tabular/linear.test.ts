import { describe, expect, it } from "vitest";

import { mulberry32 } from "@/lib/random";
import { cpuMatmul, type MatmulFn } from "@/webgpu/linearModel";

import { fitLogistic } from "./linear";

const cpu: MatmulFn = async (a, b, m, k, n) => cpuMatmul(a, b, m, k, n);

/**
 * A stand-in for the GPU path: same arithmetic, different order of operations
 * and a round-trip through a copy, which is what the real kernel adds. The
 * seam exists because a kernel that disagrees with its reference fails
 * *silently* — the loss still falls and the accuracy still looks plausible.
 */
const stubGpu: MatmulFn = async (a, b, m, k, n) => {
  const out = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) sum += a[i * k + p] * b[p * n + j];
      out[i * n + j] = sum;
    }
  }
  return Float32Array.from(out);
};

function separable(n: number, seed = 8) {
  const rand = mulberry32(seed);
  const x = new Float32Array(n * 2);
  const y = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const a = rand() * 2 - 1;
    const b = rand() * 2 - 1;
    x[i * 2] = a;
    x[i * 2 + 1] = b;
    y[i] = a + b > 0 ? 1 : 0;
  }
  return { x, y };
}

describe("fitLogistic", () => {
  it("separates a linearly separable problem", async () => {
    const n = 300;
    const { x, y } = separable(n);
    const model = await fitLogistic(cpu, x, y, n, 2, 2, {
      epochs: 40,
      learningRate: 0.5,
      batchSize: 32,
      seed: 1,
    });
    const probs = await model.predict(x, n);
    let correct = 0;
    for (let i = 0; i < n; i++) if ((probs[i * 2 + 1] > 0.5 ? 1 : 0) === y[i]) correct++;
    expect(correct / n).toBeGreaterThan(0.95);
  });

  it("gives the same answer on the CPU reference and a stub GPU matmul", async () => {
    const n = 120;
    const { x, y } = separable(n, 12);
    const opts = {
      epochs: 10,
      learningRate: 0.3,
      batchSize: 16,
      seed: 5,
    };
    const a = await fitLogistic(cpu, x, y, n, 2, 2, opts);
    const b = await fitLogistic(stubGpu, x, y, n, 2, 2, opts);
    for (let i = 0; i < a.weights.length; i++) {
      expect(a.weights[i]).toBeCloseTo(b.weights[i], 4);
    }
    for (let i = 0; i < a.bias.length; i++) {
      expect(a.bias[i]).toBeCloseTo(b.bias[i], 4);
    }
  });

  it("reports progress per epoch and stops when asked", async () => {
    const n = 64;
    const { x, y } = separable(n);
    const seen: number[] = [];
    await fitLogistic(cpu, x, y, n, 2, 2, {
      epochs: 8,
      learningRate: 0.2,
      batchSize: 16,
      seed: 2,
      onProgress: (done) => seen.push(done),
      shouldStop: () => seen.length >= 2,
    });
    expect(seen.length).toBeLessThanOrEqual(3);
    expect(seen[0]).toBe(1);
  });

  it("returns probabilities that sum to one per row", async () => {
    const n = 30;
    const { x, y } = separable(n);
    const model = await fitLogistic(cpu, x, y, n, 2, 2, {
      epochs: 3,
      learningRate: 0.1,
      batchSize: 8,
      seed: 3,
    });
    const probs = await model.predict(x, n);
    for (let i = 0; i < n; i++) {
      expect(probs[i * 2] + probs[i * 2 + 1]).toBeCloseTo(1, 5);
    }
  });
});
