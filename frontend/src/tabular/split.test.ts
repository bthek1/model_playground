import { describe, expect, it } from "vitest";

import { permutation, splitRows } from "./split";

describe("splitRows", () => {
  it("is reproducible from the seed alone", () => {
    const a = splitRows(200, 0.25, 7);
    const b = splitRows(200, 0.25, 7);
    expect(Array.from(a.test)).toEqual(Array.from(b.test));
    expect(Array.from(splitRows(200, 0.25, 8).test)).not.toEqual(
      Array.from(a.test),
    );
  });

  it("puts no row in both halves and loses none", () => {
    const { train, test } = splitRows(137, 0.3, 3);
    const all = [...train, ...test].sort((x, y) => x - y);
    expect(all).toEqual(Array.from({ length: 137 }, (_, i) => i));
    expect(new Set(all).size).toBe(137);
  });

  it("preserves the class ratio to within a row", () => {
    // 90/10, which is where an unstratified split can hand the held-out half a
    // class the model never saw — and the metrics then describe a different
    // problem from the one on screen.
    const strata = new Int32Array(200);
    for (let i = 0; i < 20; i++) strata[i] = 1;
    const { train, test } = splitRows(200, 0.25, 11, strata);
    const count = (rows: Int32Array) =>
      Array.from(rows).filter((r) => strata[r] === 1).length;
    expect(count(test)).toBeGreaterThanOrEqual(4);
    expect(count(test)).toBeLessThanOrEqual(6);
    expect(count(train)).toBe(20 - count(test));
  });

  it("keeps at least one row of a two-row class on each side", () => {
    const strata = new Int32Array(102);
    strata[0] = 1;
    strata[1] = 1;
    const { train, test } = splitRows(102, 0.25, 5, strata);
    expect(Array.from(test).filter((r) => strata[r] === 1)).toHaveLength(1);
    expect(Array.from(train).filter((r) => strata[r] === 1)).toHaveLength(1);
  });

  it("refuses a frame too small to split", () => {
    expect(() => splitRows(1, 0.25, 1)).toThrow(/two rows/i);
  });
});

describe("permutation", () => {
  it("is a permutation, and depends only on its own seed", () => {
    const p = permutation(50, 4);
    expect(Array.from(p).sort((a, b) => a - b)).toEqual(
      Array.from({ length: 50 }, (_, i) => i),
    );
    // Importance draws per column from `seed + column`, so a shuffle must not
    // depend on how many columns came before it — otherwise the ranking moves
    // when an unrelated feature is toggled off.
    expect(Array.from(permutation(50, 4))).toEqual(Array.from(p));
  });
});
