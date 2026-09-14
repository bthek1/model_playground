import { describe, expect, it } from "vitest";

import { gaussian, mulberry32, shuffle } from "./random";

// Reproducibility is the whole contract here: the graph layout, the dataset
// split and every training run are seeded from these, and a stream that moved
// between two runs would make every head-to-head comparison in the app
// meaningless.

describe("mulberry32", () => {
  it("gives the same stream for the same seed, and a different one otherwise", () => {
    const take = (seed: number) => {
      const rand = mulberry32(seed);
      return Array.from({ length: 8 }, () => rand());
    };
    expect(take(42)).toEqual(take(42));
    expect(take(42)).not.toEqual(take(43));
  });

  it("stays in [0, 1)", () => {
    const rand = mulberry32(7);
    for (let i = 0; i < 5000; i++) {
      const v = rand();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it("covers the range rather than clustering", () => {
    const rand = mulberry32(11);
    const buckets = new Array(10).fill(0);
    for (let i = 0; i < 10_000; i++) buckets[Math.floor(rand() * 10)]++;
    // A uniform stream puts ~1000 in each; this is loose enough never to flake
    // and tight enough to catch a generator stuck in part of the range.
    for (const count of buckets) expect(count).toBeGreaterThan(700);
  });

  it("accepts seed 0 without collapsing", () => {
    const rand = mulberry32(0);
    const values = Array.from({ length: 5 }, () => rand());
    expect(new Set(values).size).toBe(5);
  });
});

describe("shuffle", () => {
  it("permutes rather than replaces", () => {
    const arr = Int32Array.from({ length: 50 }, (_, i) => i);
    shuffle(arr, mulberry32(3));
    expect([...arr].sort((a, b) => a - b)).toEqual(
      Array.from({ length: 50 }, (_, i) => i),
    );
  });

  it("actually reorders", () => {
    const arr = Int32Array.from({ length: 50 }, (_, i) => i);
    shuffle(arr, mulberry32(3));
    expect([...arr]).not.toEqual(Array.from({ length: 50 }, (_, i) => i));
  });

  it("is reproducible for a seed", () => {
    const run = () => {
      const arr = Uint32Array.from({ length: 30 }, (_, i) => i);
      shuffle(arr, mulberry32(5));
      return [...arr];
    };
    expect(run()).toEqual(run());
  });

  it("handles the arrays with nothing to do", () => {
    const empty = new Int32Array(0);
    expect(() => shuffle(empty, mulberry32(1))).not.toThrow();
    const one = Int32Array.from([9]);
    shuffle(one, mulberry32(1));
    expect([...one]).toEqual([9]);
  });

  it("works on a plain array as well as a typed one", () => {
    const arr = [1, 2, 3, 4, 5];
    shuffle(arr, mulberry32(2));
    expect([...arr].sort()).toEqual([1, 2, 3, 4, 5]);
  });
});

describe("gaussian", () => {
  it("is finite even when the RNG returns exactly 0", () => {
    // Box–Muller takes log(u); `u = 1 - rand()` is what keeps a 0 draw from
    // producing -Infinity and poisoning every weight in the layer.
    expect(Number.isFinite(gaussian(() => 0))).toBe(true);
  });

  it("is finite across the whole unit interval", () => {
    for (const v of [0, 0.25, 0.5, 0.75, 0.999999]) {
      expect(Number.isFinite(gaussian(() => v))).toBe(true);
    }
  });

  it("has roughly zero mean and unit variance", () => {
    const rand = mulberry32(17);
    const n = 20_000;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const v = gaussian(rand);
      sum += v;
      sumSq += v * v;
    }
    const mean = sum / n;
    expect(Math.abs(mean)).toBeLessThan(0.05);
    expect(Math.sqrt(sumSq / n - mean * mean)).toBeCloseTo(1, 1);
  });

  it("is reproducible for a seed", () => {
    const take = () => {
      const rand = mulberry32(23);
      return Array.from({ length: 6 }, () => gaussian(rand));
    };
    expect(take()).toEqual(take());
  });
});
