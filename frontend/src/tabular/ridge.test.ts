import { describe, expect, it } from "vitest";

import { mulberry32 } from "@/lib/random";
import { cpuMatmul, type MatmulFn } from "@/webgpu/linearModel";

import { cholesky, choleskySolve, fitRidge, predictRidge } from "./ridge";

const cpu: MatmulFn = async (a, b, m, k, n) => cpuMatmul(a, b, m, k, n);

/** Same arithmetic, different loop order and a copy — the GPU path's shape. */
const stubGpu: MatmulFn = async (a, b, m, k, n) => {
  const out = new Float32Array(m * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < m; i++) {
      let sum = 0;
      for (let p = 0; p < k; p++) sum += a[i * k + p] * b[p * n + j];
      out[i * n + j] = sum;
    }
  }
  return Float32Array.from(out);
};

describe("cholesky", () => {
  it("factorises a positive definite matrix", () => {
    // [[4,2],[2,3]] = L Lᵀ with L = [[2,0],[1,√2]]
    const a = Float64Array.from([4, 2, 2, 3]);
    expect(cholesky(a, 2)).toBe(true);
    expect(a[0]).toBeCloseTo(2);
    expect(a[2]).toBeCloseTo(1);
    expect(a[3]).toBeCloseTo(Math.SQRT2);
    // The strict upper triangle is not part of the factor.
    expect(a[1]).toBe(0);
  });

  it("refuses a matrix that is not positive definite", () => {
    expect(cholesky(Float64Array.from([1, 2, 2, 1]), 2)).toBe(false);
    expect(cholesky(Float64Array.from([0, 0, 0, 0]), 2)).toBe(false);
  });

  it("solves the system it factorised", () => {
    const a = Float64Array.from([4, 2, 2, 3]);
    cholesky(a, 2);
    const w = choleskySolve(a, Float64Array.from([10, 11]), 2);
    // 4x + 2y = 10, 2x + 3y = 11  ->  x = 1, y = 3
    expect(w[0]).toBeCloseTo(1, 6);
    expect(w[1]).toBeCloseTo(3, 6);
  });
});

describe("fitRidge", () => {
  it("recovers a hand-computed fit on a 3x2 design", async () => {
    // y = 2·x0 + 1, exactly. With λ → 0 ridge must find it.
    const x = Float32Array.from([1, 0, 2, 0, 3, 1]);
    const y = Float32Array.from([3, 5, 7]);
    const model = await fitRidge(cpu, x, y, 3, 2, 1e-8);
    const pred = predictRidge(model, x, 3, 2);
    for (let i = 0; i < 3; i++) expect(pred[i]).toBeCloseTo(y[i], 3);
    expect(model.rankDeficient).toBe(false);
  });

  it("approaches the least-squares solution as λ falls", async () => {
    const rand = mulberry32(4);
    const n = 200;
    const x = new Float32Array(n * 2);
    const y = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const a = rand() * 4 - 2;
      const b = rand() * 4 - 2;
      x[i * 2] = a;
      x[i * 2 + 1] = b;
      y[i] = 1.5 * a - 0.75 * b + 2 + (rand() - 0.5) * 0.02;
    }
    const tiny = await fitRidge(cpu, x, y, n, 2, 1e-8);
    expect(tiny.weights[0]).toBeCloseTo(1.5, 2);
    expect(tiny.weights[1]).toBeCloseTo(-0.75, 2);
    expect(tiny.intercept).toBeCloseTo(2, 2);
    // Shrinkage is what the penalty does, so a big λ must pull them in.
    const heavy = await fitRidge(cpu, x, y, n, 2, 500);
    expect(Math.abs(heavy.weights[0])).toBeLessThan(Math.abs(tiny.weights[0]));
  });

  it("is rank-deficient at λ=0 on an exactly collinear design, and solvable above it", async () => {
    // The property that justifies the whole choice: ridge's λ > 0 is what makes
    // XᵀX positive definite, which is what makes a Cholesky solve legitimate.
    // An inverse here would return one of infinitely many answers and look fine.
    const rand = mulberry32(6);
    const n = 40;
    const x = new Float32Array(n * 2);
    const y = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const a = rand();
      x[i * 2] = a;
      x[i * 2 + 1] = a; // a duplicated column
      y[i] = 3 * a + 1;
    }
    const bare = await fitRidge(cpu, x, y, n, 2, 0);
    expect(bare.rankDeficient).toBe(true);
    // It still returns something usable, and says it fell back.
    expect(bare.lambda).toBeGreaterThan(0);

    // A penalty small next to the design's own scale: enough to make the
    // matrix positive definite, not enough to shrink the fit away.
    const penalised = await fitRidge(cpu, x, y, n, 2, 0.01);
    expect(penalised.rankDeficient).toBe(false);
    const pred = predictRidge(penalised, x, n, 2);
    for (let i = 0; i < n; i++) expect(pred[i]).toBeCloseTo(y[i], 1);
  });

  it("reports a large condition number on a nearly collinear design", async () => {
    const rand = mulberry32(11);
    const n = 60;
    const bad = new Float32Array(n * 2);
    const good = new Float32Array(n * 2);
    const y = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const a = rand();
      bad[i * 2] = a;
      bad[i * 2 + 1] = a + 1e-4 * rand();
      good[i * 2] = a;
      good[i * 2 + 1] = rand();
      y[i] = a;
    }
    const ill = await fitRidge(cpu, bad, y, n, 2, 1e-8);
    const well = await fitRidge(cpu, good, y, n, 2, 1e-8);
    expect(ill.condition).toBeGreaterThan(well.condition * 100);
  });

  it("agrees between the CPU reference and a stub GPU matmul", async () => {
    // The seam's whole purpose: a kernel that disagrees with its reference
    // fails silently — the residuals still look like residuals.
    const rand = mulberry32(21);
    const n = 120;
    const x = Float32Array.from({ length: n * 3 }, () => rand() * 2 - 1);
    const y = Float32Array.from({ length: n }, (_, i) => x[i * 3] * 2 + 0.5);
    const a = await fitRidge(cpu, x, y, n, 3, 0.5);
    const b = await fitRidge(stubGpu, x, y, n, 3, 0.5);
    for (let i = 0; i < 3; i++) expect(a.weights[i]).toBeCloseTo(b.weights[i], 4);
    expect(a.intercept).toBeCloseTo(b.intercept, 4);
  });

  it("does not penalise the intercept", async () => {
    // Shrinking the intercept would pull predictions toward zero rather than
    // toward the mean, which is a different and wrong model.
    const n = 50;
    const x = new Float32Array(n);
    const y = new Float32Array(n).fill(100);
    const model = await fitRidge(cpu, x, y, n, 1, 1000);
    expect(model.intercept).toBeCloseTo(100, 1);
  });
});
