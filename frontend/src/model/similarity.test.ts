import { describe, expect, it } from "vitest";

import { cosine, l2norm, normalize, topK } from "./similarity";

const v = (...xs: number[]) => Float32Array.from(xs);

describe("l2norm", () => {
  it("measures the vector's length", () => {
    expect(l2norm(v(3, 4))).toBeCloseTo(5, 6);
    expect(l2norm(v(0, 0, 0))).toBe(0);
  });
});

describe("normalize", () => {
  it("returns a unit vector pointing the same way", () => {
    const unit = normalize(v(3, 4));
    expect(l2norm(unit)).toBeCloseTo(1, 6);
    expect(unit[0] / unit[1]).toBeCloseTo(3 / 4, 6);
  });

  it("leaves a zero vector alone rather than producing NaNs", () => {
    // A zero vector can only come from a model that produced nothing, and a
    // page full of NaN scores hides that failure behind arithmetic noise.
    const out = normalize(v(0, 0, 0));
    expect([...out]).toEqual([0, 0, 0]);
    expect(out.some(Number.isNaN)).toBe(false);
  });
});

describe("cosine", () => {
  it("is 1 for a vector against itself, whatever its length", () => {
    expect(cosine(v(1, 2, 3), v(1, 2, 3))).toBeCloseTo(1, 6);
    expect(cosine(v(1, 2, 3), v(10, 20, 30))).toBeCloseTo(1, 6);
  });

  it("is 0 for orthogonal vectors and -1 for opposites", () => {
    expect(cosine(v(1, 0), v(0, 1))).toBeCloseTo(0, 6);
    expect(cosine(v(1, 0), v(-1, 0))).toBeCloseTo(-1, 6);
  });

  it("does not assume its inputs are normalised", () => {
    // The assumption is exactly the bug this page exists to make visible: it
    // would let an unnormalised embedding produce plausible-looking scores.
    expect(cosine(v(3, 4), v(6, 8))).toBeCloseTo(1, 6);
  });

  it("returns 0 rather than NaN against a zero vector", () => {
    expect(cosine(v(1, 2), v(0, 0))).toBe(0);
  });
});

describe("topK", () => {
  const index = [
    { id: "a", vector: normalize(v(1, 0, 0)) },
    { id: "b", vector: normalize(v(0.9, 0.1, 0)) },
    { id: "c", vector: normalize(v(0, 1, 0)) },
  ];

  it("makes a vector its own nearest neighbour with score 1", () => {
    const [first] = topK(index[0].vector, index, 3);
    expect(first.id).toBe("a");
    expect(first.score).toBeCloseTo(1, 6);
  });

  it("orders by similarity, best first, and honours k", () => {
    const out = topK(normalize(v(1, 0, 0)), index, 2);
    expect(out.map((n) => n.id)).toEqual(["a", "b"]);
  });

  it("keeps index order for ties, so a re-rank does not reshuffle", () => {
    const tied = [
      { id: "x", vector: normalize(v(1, 0)) },
      { id: "y", vector: normalize(v(1, 0)) },
    ];
    expect(topK(normalize(v(1, 0)), tied, 2).map((n) => n.id)).toEqual([
      "x",
      "y",
    ]);
  });

  it("returns nothing for an empty index rather than throwing", () => {
    // The normal state of the page for its first few seconds.
    expect(topK(v(1, 0), [], 5)).toEqual([]);
  });

  it("returns nothing for k of zero or less", () => {
    expect(topK(index[0].vector, index, 0)).toEqual([]);
    expect(topK(index[0].vector, index, -1)).toEqual([]);
  });
});
