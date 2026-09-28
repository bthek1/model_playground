import { describe, expect, it } from "vitest";

import { MAX_POINTS, runningMean, strideIndices } from "./series";

describe("runningMean", () => {
  it("averages what exists before the window fills, then slides", () => {
    expect(runningMean([1, 0, 1, 1, 0], 3)).toEqual([1, 0.5, 2 / 3, 2 / 3, 2 / 3]);
  });

  it("is the success rate on 0/1 returns", () => {
    const returns = [...new Array(50).fill(0), ...new Array(50).fill(1)];
    expect(runningMean(returns, 100)[99]).toBe(0.5);
  });
});

describe("strideIndices", () => {
  it("keeps every point of a short series", () => {
    expect(strideIndices(5)).toEqual([0, 1, 2, 3, 4]);
  });

  it("caps a long series and always keeps the last episode", () => {
    const idx = strideIndices(50_000);
    expect(idx).toHaveLength(MAX_POINTS);
    expect(idx[0]).toBe(0);
    expect(idx[idx.length - 1]).toBe(49_999);
    for (let i = 1; i < idx.length; i++) expect(idx[i]).toBeGreaterThan(idx[i - 1]);
  });
});
