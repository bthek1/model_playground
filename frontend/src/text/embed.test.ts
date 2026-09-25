import { describe, expect, it } from "vitest";

import { l2norm } from "@/model/similarity";

import { EmbeddingCache, toVector, truncate, truncationSteps } from "./embed";

const vec = (...xs: number[]) => new Float32Array(xs);

describe("toVector", () => {
  it("reads a pooled [1, dim] tensor", () => {
    const v = toVector({ data: [1, 2, 3], dims: [1, 3] });
    expect([...v]).toEqual([1, 2, 3]);
  });

  it("reads a bare [dim] tensor", () => {
    expect([...toVector({ data: [4, 5], dims: [2] })]).toEqual([4, 5]);
  });

  // The pooling option not reaching the model is this page's one silent
  // failure, and row 0 of a BERT hidden state is a perfectly plausible vector
  // that would rank and be wrong. So it throws rather than taking it.
  it("throws on an unpooled [1, tokens, dim] hidden state", () => {
    expect(() =>
      toVector({ data: new Array(12).fill(0), dims: [1, 4, 3] }),
    ).toThrow(/pooling option did not reach/i);
  });

  it("throws on an empty embedding rather than returning one", () => {
    expect(() => toVector({ data: [], dims: [1, 0] })).toThrow(/empty/i);
  });
});

describe("truncate", () => {
  it("renormalises, so the truncated vector is unit length", () => {
    // A vector whose tail carries most of its length: truncating without
    // renormalising would leave ‖v‖ ≈ 0.27 and scale every cosine by it.
    const v = vec(1, 1, 5, 5);
    const t = truncate(v, 2);
    expect(t.dim).toBe(2);
    expect(l2norm(t.vector)).toBeCloseTo(1, 6);
  });

  it("reports the fraction of the length the prefix held", () => {
    const v = vec(3, 4);
    // Prefix [3] has norm 3; the whole vector has norm 5.
    expect(truncate(v, 1).kept).toBeCloseTo(3 / 5, 6);
  });

  it("reports 1 when nothing was dropped", () => {
    expect(truncate(vec(1, 2, 3), 3).kept).toBeCloseTo(1, 6);
  });

  it("clamps a width beyond the vector to the whole vector", () => {
    const t = truncate(vec(1, 2), 99);
    expect(t.dim).toBe(2);
    expect(t.kept).toBeCloseTo(1, 6);
  });

  // The caller is a slider, and a slider that can produce an empty vector
  // produces NaN scores instead of an error.
  it("never returns an empty vector", () => {
    expect(truncate(vec(1, 2, 3), 0).dim).toBe(1);
    expect(truncate(vec(1, 2, 3), -4).dim).toBe(1);
  });

  it("returns zeros rather than NaNs for an all-zero vector", () => {
    const t = truncate(vec(0, 0, 0), 2);
    expect([...t.vector]).toEqual([0, 0]);
    expect(t.kept).toBe(0);
  });
});

describe("truncationSteps", () => {
  it("halves down to 64 from the model's own width", () => {
    expect(truncationSteps(768)).toEqual([768, 384, 192, 96]);
    expect(truncationSteps(384)).toEqual([384, 192, 96]);
  });

  it("offers nothing below 64", () => {
    expect(truncationSteps(64)).toEqual([64]);
    expect(truncationSteps(32)).toEqual([]);
  });
});

describe("EmbeddingCache", () => {
  it("returns the same vector for the same string and misses on a changed one", () => {
    const cache = new EmbeddingCache();
    const v = vec(1, 2);
    cache.set("hello", v);
    expect(cache.get("hello")).toBe(v);
    expect(cache.has("hello ")).toBe(false);
    expect(cache.get("hello!")).toBeUndefined();
  });

  // Asserted directly, because the failure is a *plausible number* rather than
  // an error: a 384-d MiniLM vector served for a 768-d BGE query is a confident
  // wrong answer with nothing throwing anywhere.
  it("clears completely", () => {
    const cache = new EmbeddingCache();
    cache.set("a", vec(1));
    cache.set("b", vec(2));
    expect(cache.size).toBe(2);
    cache.clear();
    expect(cache.size).toBe(0);
    expect(cache.has("a")).toBe(false);
  });

  it("evicts in insertion order once full, keeping the corpus over the queries", () => {
    const cache = new EmbeddingCache(2);
    cache.set("first", vec(1));
    cache.set("second", vec(2));
    cache.set("third", vec(3));
    expect(cache.has("first")).toBe(false);
    expect(cache.has("second")).toBe(true);
    expect(cache.has("third")).toBe(true);
    expect(cache.size).toBe(2);
  });

  it("re-setting an existing key does not evict anything", () => {
    const cache = new EmbeddingCache(2);
    cache.set("a", vec(1));
    cache.set("b", vec(2));
    cache.set("a", vec(9));
    expect(cache.size).toBe(2);
    expect(cache.get("a")?.[0]).toBe(9);
    expect(cache.has("b")).toBe(true);
  });
});
