import { describe, expect, it } from "vitest";

import { DEFAULT_RRF_K, fuse, rankOf } from "./rrf";

describe("fuse", () => {
  it("fusing a list with itself returns that list", () => {
    const list = [3, 1, 2, 0];
    expect(fuse([list, list]).map((f) => f.doc)).toEqual(list);
  });

  it("fusing one list returns that list", () => {
    expect(fuse([[5, 4, 6]]).map((f) => f.doc)).toEqual([5, 4, 6]);
  });

  // **The assumption to unlearn.** "A document both lists rank second beats one
  // that is first in one and last in the other" is the intuitive reading of RRF
  // and it is false: `1/(k+r)` is convex, so by Jensen the extreme pair wins.
  // The margin is tiny and the sign is fixed, and a page showing four rank lists
  // will display it — so it is pinned here rather than discovered on screen.
  it("rewards an extreme pair over a consistent middle, because 1/(k+r) is convex", () => {
    //   list A: 0, 1, 2
    //   list B: 2, 1, 0
    // 1 is second in both; 0 and 2 are first in one and last in the other.
    const fused = fuse([
      [0, 1, 2],
      [2, 1, 0],
    ]);
    const middle = fused.find((f) => f.doc === 1)!;
    const extreme = fused.find((f) => f.doc === 0)!;
    expect(extreme.score).toBeGreaterThan(middle.score);
    // 1/60 + 1/62 against 1/61 + 1/61 — a margin in the fifth decimal.
    expect(extreme.score - middle.score).toBeLessThan(1e-4);
    // The two extremes tie with each other, by symmetry.
    const other = fused.find((f) => f.doc === 2)!;
    expect(other.score).toBeCloseTo(extreme.score, 12);
  });

  it("closes that gap as k grows, and widens it as k falls", () => {
    const lists = [
      [0, 1, 2],
      [2, 1, 0],
    ];
    const gap = (k: number) => {
      const f = fuse(lists, k);
      return (
        f.find((x) => x.doc === 0)!.score - f.find((x) => x.doc === 1)!.score
      );
    };
    expect(gap(1)).toBeGreaterThan(gap(60));
    expect(gap(60)).toBeGreaterThan(gap(1000));
    expect(gap(1000)).toBeGreaterThan(0);
  });

  // Absence means "not in this stage's shortlist", not "ranked last" — treating
  // it as last place would let a longer list outvote a shorter one by length.
  it("contributes nothing from a list that omits a document", () => {
    const fused = fuse([[0, 1], [0]]);
    const zero = fused.find((f) => f.doc === 0)!;
    const one = fused.find((f) => f.doc === 1)!;
    expect(zero.score).toBeCloseTo(
      1 / DEFAULT_RRF_K + 1 / DEFAULT_RRF_K,
      12,
    );
    expect(one.score).toBeCloseTo(1 / (DEFAULT_RRF_K + 1), 12);
    expect(one.ranks).toEqual([1, null]);
  });

  it("records each document's rank per list, for the movement between stages", () => {
    const fused = fuse([
      [7, 8],
      [8, 7],
    ]);
    expect(fused.find((f) => f.doc === 7)!.ranks).toEqual([0, 1]);
    expect(fused.find((f) => f.doc === 8)!.ranks).toEqual([1, 0]);
  });

  it("counts a repeated document once, at its best rank", () => {
    const fused = fuse([[1, 1, 2]]);
    const one = fused.find((f) => f.doc === 1)!;
    expect(one.score).toBeCloseTo(1 / DEFAULT_RRF_K, 12);
    expect(one.ranks).toEqual([0]);
  });

  it("damps the top ranks less as k falls", () => {
    const lists = [
      [0, 1],
      [1, 0],
    ];
    // Symmetric input, so the scores tie whatever k is — what changes is the
    // magnitude, which is the thing k controls.
    const tight = fuse(lists, 1);
    const loose = fuse(lists, 1000);
    expect(tight[0].score).toBeGreaterThan(loose[0].score);
  });

  it("returns nothing for no lists, or only empty ones", () => {
    expect(fuse([])).toEqual([]);
    expect(fuse([[], []])).toEqual([]);
  });

  it("is deterministic on ties, by lowest document index", () => {
    const fused = fuse([
      [2, 0, 1],
      [1, 0, 2],
    ]);
    // 1 and 2 are each first in one list and last in the other, so by the
    // convexity above they tie *above* 0 — which is second in both. The
    // assertion here is the tie-break, not the ordering: equal scores come back
    // in document order, every time.
    expect(fused[0].score).toBeCloseTo(fused[1].score, 12);
    expect([fused[0].doc, fused[1].doc]).toEqual([1, 2]);
    expect(fused[2].doc).toBe(0);
  });
});

describe("rankOf", () => {
  it("maps a document to its position", () => {
    const ranks = rankOf([4, 9, 1]);
    expect(ranks.get(4)).toBe(0);
    expect(ranks.get(1)).toBe(2);
    expect(ranks.has(7)).toBe(false);
  });

  it("keeps the first position for a repeated document", () => {
    expect(rankOf([3, 3]).get(3)).toBe(0);
  });
});
