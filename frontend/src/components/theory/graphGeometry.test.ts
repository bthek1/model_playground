import { describe, expect, it } from "vitest";

import { countOpacity, fitToBox, reachedOpacity } from "./graphGeometry";

const box = { width: 200, height: 100, pad: 10 };

describe("fitToBox", () => {
  it("fits the layout's own extent, centred, keeping its aspect ratio", () => {
    // A 1×1 square in a 200×100 box: limited by height, so 80 px a side,
    // centred horizontally.
    const { x, y } = fitToBox({ x: [0.2, 0.4, 0.4], y: [0.1, 0.1, 0.3] }, box);
    expect(y[0]).toBeCloseTo(10);
    expect(y[2]).toBeCloseTo(90);
    expect(x[1] - x[0]).toBeCloseTo(80);
    expect((x[0] + x[1]) / 2).toBeCloseTo(100);
  });

  it("stretches a flat drawing to the width rather than leaving it a sliver", () => {
    const { x, y } = fitToBox({ x: [0, 0.5, 1], y: [0.49, 0.51, 0.49] }, box);
    expect(x[0]).toBeCloseTo(10);
    expect(x[2]).toBeCloseTo(190);
    // Aspect ratio kept: 0.02 of height at 180 px per unit.
    expect(y[1] - y[0]).toBeCloseTo(3.6);
  });

  it("centres a single node instead of dividing by zero", () => {
    expect(fitToBox({ x: [0.3], y: [0.7] }, box)).toEqual({ x: [100], y: [50] });
  });

  it("centres a straight line on its flat axis", () => {
    const { x, y } = fitToBox({ x: [0, 1], y: [0.5, 0.5] }, box);
    expect(x).toEqual([10, 190]);
    expect(y).toEqual([50, 50]);
  });

  it("returns nothing for no nodes", () => {
    expect(fitToBox({ x: [], y: [] }, box)).toEqual({ x: [], y: [] });
  });
});

describe("reachedOpacity", () => {
  it("is strongest at the source and fades to the frontier", () => {
    expect(reachedOpacity(0, 3)).toBeCloseTo(0.55);
    expect(reachedOpacity(3, 3)).toBeCloseTo(0.2);
    expect(reachedOpacity(1, 3)).toBeGreaterThan(reachedOpacity(2, 3));
  });

  it("is zero outside the ball and for unreachable nodes", () => {
    expect(reachedOpacity(4, 3)).toBe(0);
    expect(reachedOpacity(-1, 3)).toBe(0);
  });

  it("handles k = 0", () => {
    expect(reachedOpacity(0, 0)).toBeCloseTo(0.55);
  });
});

describe("countOpacity", () => {
  it("leaves a zero count empty, so empty means no walk", () => {
    expect(countOpacity(0, 100)).toBe(0);
  });

  it("gives the largest count full strength and every nonzero count a visible floor", () => {
    expect(countOpacity(100, 100)).toBeCloseTo(1);
    expect(countOpacity(1, 4744)).toBeGreaterThan(0.15);
  });

  it("is monotonic", () => {
    expect(countOpacity(10, 100)).toBeGreaterThan(countOpacity(9, 100));
  });
});
