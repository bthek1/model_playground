import { describe, expect, it } from "vitest";

import { createSeries, maxOf, DEFAULT_CAPACITY } from "./series";

describe("createSeries", () => {
  it("defaults to a two-minute window at 1 Hz", () => {
    expect(createSeries().capacity).toBe(DEFAULT_CAPACITY);
  });

  it("rejects a nonsensical capacity rather than silently clamping", () => {
    expect(() => createSeries(0)).toThrow(/positive integer/);
    expect(() => createSeries(2.5)).toThrow(/positive integer/);
  });

  it("is empty before anything is pushed", () => {
    const s = createSeries<number>(3);
    expect(s.size()).toBe(0);
    expect(s.latest()).toBeNull();
    expect(s.toArray()).toEqual([]);
  });

  it("returns samples oldest → newest while below capacity", () => {
    const s = createSeries<number>(5);
    s.push(1);
    s.push(2);
    s.push(3);
    expect(s.toArray()).toEqual([1, 2, 3]);
    expect(s.latest()).toBe(3);
    expect(s.size()).toBe(3);
  });

  it("evicts the oldest sample once full, and never exceeds capacity", () => {
    const s = createSeries<number>(3);
    for (const n of [1, 2, 3, 4, 5]) s.push(n);
    expect(s.toArray()).toEqual([3, 4, 5]);
    expect(s.size()).toBe(3);
    expect(s.latest()).toBe(5);
  });

  it("survives many wraps without drifting", () => {
    const s = createSeries<number>(4);
    for (let i = 0; i < 1000; i++) s.push(i);
    expect(s.toArray()).toEqual([996, 997, 998, 999]);
    expect(s.size()).toBe(4);
  });

  it("hands out a copy, so a caller cannot corrupt the buffer", () => {
    const s = createSeries<number>(3);
    s.push(1);
    const first = s.toArray();
    first.push(99);
    expect(s.toArray()).toEqual([1]);
  });

  it("clear empties it", () => {
    const s = createSeries<number>(3);
    s.push(1);
    s.clear();
    expect(s.size()).toBe(0);
    expect(s.latest()).toBeNull();
  });
});

describe("maxOf", () => {
  it("returns the fallback for an empty series", () => {
    expect(maxOf([], 7)).toBe(7);
    expect(maxOf([])).toBe(0);
  });

  it("ignores non-finite values", () => {
    expect(maxOf([1, NaN, 5, Infinity, 3])).toBe(5);
  });

  it("returns the fallback when every value is non-finite", () => {
    expect(maxOf([NaN, Infinity], 2)).toBe(2);
  });

  it("finds the max below the fallback too", () => {
    expect(maxOf([1, 2], 100)).toBe(2);
  });
});
