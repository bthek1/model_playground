import { describe, expect, it } from "vitest";

import { gaussian, mulberry32 } from "@/lib/random";
import { cpuMatmul, LinearTrainer, type MatmulFn } from "@/webgpu/linearModel";

import { backwardMlp, fitMlp, forwardMlp, initMlp, type MlpParams } from "./mlp";

const matmul: MatmulFn = async (a, b, m, k, n) => cpuMatmul(a, b, m, k, n);

/** A deliberately *generic* point — see the comment in the gradient test. */
function genericParams(features: number, hidden: number, classes: number): MlpParams {
  const rand = mulberry32(17);
  const p = initMlp(features, hidden, classes, 17);
  for (let i = 0; i < p.b1.length; i++) p.b1[i] = gaussian(rand) * 0.5;
  for (let i = 0; i < p.b2.length; i++) p.b2[i] = gaussian(rand) * 0.5;
  return p;
}

function loss(
  p: MlpParams,
  x: Float32Array,
  y: Uint8Array,
  n: number,
): Promise<number> {
  return forwardMlp(matmul, p, x, n).then(
    ({ probs }) => LinearTrainer.crossEntropy(probs, y, n, p.classes).loss,
  );
}

describe("backwardMlp", () => {
  it("matches central finite differences at a generic point", async () => {
    // **Generic** matters here and is not decoration. Zero-initialised biases
    // put some preactivations exactly on ReLU's kink, where a central
    // difference straddles the corner and reports half the true gradient — so a
    // correct implementation fails the check and the natural next move is to
    // "fix" the code. `webgpu/gnn.test.ts` paid for that lesson; the biases
    // above are randomised for precisely this reason.
    const features = 4;
    const hidden = 5;
    const classes = 3;
    const n = 6;
    const rand = mulberry32(31);
    const x = Float32Array.from({ length: n * features }, () => gaussian(rand));
    const y = Uint8Array.from({ length: n }, () => Math.floor(rand() * classes));
    const p = genericParams(features, hidden, classes);

    const g = await backwardMlp(matmul, p, x, y, n);
    const eps = 1e-3;

    const checks: [keyof MlpParams, Float32Array][] = [
      ["w1", g.dw1],
      ["b1", g.db1],
      ["w2", g.dw2],
      ["b2", g.db2],
    ];
    for (const [name, analytic] of checks) {
      const buffer = p[name] as Float32Array;
      for (let i = 0; i < Math.min(buffer.length, 6); i++) {
        const original = buffer[i];
        buffer[i] = original + eps;
        const up = await loss(p, x, y, n);
        buffer[i] = original - eps;
        const down = await loss(p, x, y, n);
        buffer[i] = original;
        const numeric = (up - down) / (2 * eps);
        expect(Math.abs(numeric - analytic[i])).toBeLessThan(2e-3);
      }
    }
  });
});

describe("fitMlp", () => {
  it("learns an interaction a linear boundary cannot express", async () => {
    const n = 400;
    const rand = mulberry32(2);
    const x = new Float32Array(n * 2);
    const y = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const a = rand() * 2 - 1;
      const b = rand() * 2 - 1;
      x[i * 2] = a;
      x[i * 2 + 1] = b;
      y[i] = a * b > 0 ? 1 : 0;
    }
    const model = await fitMlp(matmul, x, y, n, 2, 2, {
      hidden: 16,
      epochs: 120,
      learningRate: 0.5,
      batchSize: 32,
      seed: 4,
    });
    const probs = await model.predict(x, n);
    let correct = 0;
    for (let i = 0; i < n; i++) if ((probs[i * 2 + 1] > 0.5 ? 1 : 0) === y[i]) correct++;
    expect(correct / n).toBeGreaterThan(0.85);
  });

  it("reports one progress event per epoch and honours a stop", async () => {
    const n = 40;
    const x = Float32Array.from({ length: n * 2 }, (_, i) => i % 7);
    const y = Uint8Array.from({ length: n }, (_, i) => i % 2);
    const seen: number[] = [];
    await fitMlp(matmul, x, y, n, 2, 2, {
      hidden: 4,
      epochs: 10,
      learningRate: 0.1,
      batchSize: 8,
      seed: 1,
      onProgress: (done) => seen.push(done),
      shouldStop: () => seen.length >= 3,
    });
    expect(seen).toEqual([1, 2, 3]);
  });
});
