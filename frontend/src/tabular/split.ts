// The train/test split, seeded and stratified.
//
// Small, and the most dangerous file in the module. A split that leaks fails
// *upward* — the held-out score goes up, the page looks better, and nothing
// throws. `/link-prediction` paid for that lesson on edges; here it is cheaper
// to get right and just as easy to get wrong, which is why the encoder in
// `design.ts` takes the training indices rather than the whole frame.

import { mulberry32, shuffle } from "@/lib/random";

export interface Split {
  /** Row indices used to fit. */
  train: Int32Array;
  /** Row indices never seen by the fit. */
  test: Int32Array;
}

/**
 * Split `rowCount` rows into a train and a test half.
 *
 * `strata`, when supplied, is one integer class code per row and the split is
 * stratified on it: each class contributes the same *fraction* to both halves.
 * Without that, a 95/5 dataset can hand the held-out half a class the model
 * never saw, and the resulting metrics describe a different problem.
 */
export function splitRows(
  rowCount: number,
  testFraction: number,
  seed: number,
  strata?: Float32Array | Int32Array,
): Split {
  if (rowCount < 2) throw new Error("Need at least two rows to split.");
  const frac = Math.min(0.9, Math.max(0.05, testFraction));
  const rand = mulberry32(seed);

  const groups = new Map<number, number[]>();
  for (let i = 0; i < rowCount; i++) {
    const key = strata ? strata[i] : 0;
    const bucket = groups.get(key);
    if (bucket) bucket.push(i);
    else groups.set(key, [i]);
  }

  const train: number[] = [];
  const test: number[] = [];
  // Sorted so the same seed gives the same split whatever order the classes
  // happened to appear in the file.
  for (const key of [...groups.keys()].sort((a, b) => a - b)) {
    const idx = Int32Array.from(groups.get(key) as number[]);
    shuffle(idx, rand);
    // `round`, so a class with 3 rows at 25% contributes 1 — and at least one
    // row stays on each side wherever the class has two.
    let nTest = Math.round(idx.length * frac);
    if (idx.length >= 2) nTest = Math.min(idx.length - 1, Math.max(1, nTest));
    for (let i = 0; i < idx.length; i++) {
      (i < nTest ? test : train).push(idx[i]);
    }
  }

  if (test.length === 0 || train.length === 0) {
    // Only reachable on a degenerate frame (one row per class, two classes).
    throw new Error("Not enough rows to hold any out — add rows or lower the split.");
  }

  return { train: Int32Array.from(train), test: Int32Array.from(test) };
}

/**
 * A seeded permutation of `n`, for permutation importance.
 *
 * Separate from `splitRows` so importance can draw from its own stream: sharing
 * one generator would make the shuffle for column 3 depend on how many
 * columns came before it, and the importances would move when the user toggled
 * an unrelated feature off.
 */
export function permutation(n: number, seed: number): Int32Array {
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  shuffle(idx, mulberry32(seed));
  return idx;
}
