import { describe, expect, it } from "vitest";

import type { DesignColumn } from "./design";
import { permutationImportance } from "./importance";

const names = new Map([
  [0, "signal"],
  [1, "noise"],
]);
const design: DesignColumn[] = [
  { name: "signal", source: 0 },
  { name: "noise", source: 1 },
];

/** Rows where column 0 carries the answer and column 1 is decoration. */
function matrix(n: number): Float32Array {
  const x = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    x[i * 2] = i % 2;
    x[i * 2 + 1] = (i * 7) % 5;
  }
  return x;
}

/** Accuracy of "predict column 0" — perfect until column 0 is shuffled. */
function scorer(n: number) {
  return async (m: Float32Array) => {
    let hits = 0;
    for (let i = 0; i < n; i++) if (m[i * 2] === i % 2) hits++;
    return hits / n;
  };
}

describe("permutationImportance", () => {
  it("ranks a used column above one the model ignores", async () => {
    const n = 200;
    const x = matrix(n);
    const out = await permutationImportance(x, n, design, names, scorer(n), {
      seed: 1,
    });
    expect(out[0].name).toBe("signal");
    expect(out[0].drop).toBeGreaterThan(0.3);
    // The ignored column must score ~0, not merely less. A ranking where the
    // noise column scores 0.2 is a ranking that would put it above a genuine
    // but weak feature.
    expect(Math.abs(out[1].drop)).toBeLessThan(1e-9);
  });

  it("restores every column it shuffles", async () => {
    // An in-place shuffle that forgets to put the values back poisons every
    // column measured afterwards and still returns a full, ordered, plausible
    // ranking — which is why this is asserted directly rather than inferred
    // from the numbers.
    const n = 120;
    const x = matrix(n);
    const before = Float32Array.from(x);
    await permutationImportance(x, n, design, names, scorer(n), { seed: 3 });
    expect(Array.from(x)).toEqual(Array.from(before));
  });

  it("groups design columns by the dataset column the user chose", async () => {
    // A one-hot city with three levels is one thing to the user. Shuffling one
    // of its indicators on its own would also produce an impossible row.
    const wide: DesignColumn[] = [
      { name: "city = a", source: 0 },
      { name: "city = b", source: 0 },
      { name: "city = c", source: 0 },
    ];
    const n = 30;
    const x = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) x[i * 3 + (i % 3)] = 1;
    const seenRows: number[] = [];
    await permutationImportance(
      x,
      n,
      wide,
      new Map([[0, "city"]]),
      async (m) => {
        // Every row must still be exactly one-hot while permuted.
        for (let i = 0; i < n; i++) {
          seenRows.push(m[i * 3] + m[i * 3 + 1] + m[i * 3 + 2]);
        }
        return 0.5;
      },
      { seed: 2, repeats: 1 },
    );
    expect(new Set(seenRows)).toEqual(new Set([1]));
  });

  it("stops when asked", async () => {
    const n = 50;
    const x = matrix(n);
    const out = await permutationImportance(x, n, design, names, scorer(n), {
      seed: 1,
      shouldStop: () => true,
    });
    expect(out).toHaveLength(0);
  });
});
