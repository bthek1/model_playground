import { describe, expect, it } from "vitest";

import { formatMatrix, parseMatrix } from "./matrix";

describe("parseMatrix", () => {
  it("parses whitespace- and comma-separated rows", () => {
    const spaces = parseMatrix("1 2 3\n4 5 6");
    expect(spaces.rows).toBe(2);
    expect(spaces.cols).toBe(3);
    expect(Array.from(spaces.data)).toEqual([1, 2, 3, 4, 5, 6]);

    const commas = parseMatrix("1, 2\n3, 4");
    expect(commas.rows).toBe(2);
    expect(commas.cols).toBe(2);
    expect(Array.from(commas.data)).toEqual([1, 2, 3, 4]);
  });

  it("accepts semicolons as row separators and trims blank lines", () => {
    const m = parseMatrix("  1 2 ; 3 4 \n\n");
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(2);
    expect(Array.from(m.data)).toEqual([1, 2, 3, 4]);
  });

  it("throws on empty input", () => {
    expect(() => parseMatrix("   \n  ")).toThrow(/at least one row/);
  });

  it("throws on ragged rows", () => {
    expect(() => parseMatrix("1 2 3\n4 5")).toThrow(/same length/);
  });

  it("throws on non-numeric cells", () => {
    expect(() => parseMatrix("1 x\n3 4")).toThrow(/not a number/);
  });
});

describe("formatMatrix", () => {
  it("reshapes a flat buffer and trims trailing zeros", () => {
    const grid = formatMatrix(new Float32Array([1, 2.5, 3, 4]), 2, 2);
    expect(grid).toEqual([
      ["1", "2.5"],
      ["3", "4"],
    ]);
  });

  it("rounds to the requested precision", () => {
    const grid = formatMatrix(new Float32Array([1 / 3]), 1, 1, 3);
    expect(grid).toEqual([["0.333"]]);
  });
});
