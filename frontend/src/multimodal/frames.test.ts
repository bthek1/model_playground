import { describe, expect, it } from "vitest";

import {
  MAX_FRAMES,
  orderFrames,
  uniformFrameTimes,
} from "./frames";

describe("uniformFrameTimes", () => {
  it("spreads the requested count evenly over the clip", () => {
    expect(uniformFrameTimes(8, 4)).toEqual([1, 3, 5, 7]);
  });

  it("samples slice centres, never the endpoints", () => {
    // 0 is usually a black frame or a fade; `duration` is past the last
    // decodable frame on a good many encodes. Both look like the model
    // ignoring the video.
    const times = uniformFrameTimes(10, 5);
    expect(times[0]).toBeGreaterThan(0);
    expect(times[times.length - 1]).toBeLessThan(10);
  });

  it("returns the requested count whatever the clip's length", () => {
    // The whole difference from `/video-classification`'s fps sampling: these
    // frames share one prompt, so the cost is the count, and eight means eight
    // on a six-second clip and on a six-minute one.
    for (const duration of [3, 30, 300]) {
      expect(uniformFrameTimes(duration, 6)).toHaveLength(6);
    }
  });

  it("puts a single frame in the middle of the clip", () => {
    expect(uniformFrameTimes(10, 1)).toEqual([5]);
  });

  it("clamps to the cap, because N frames is 64N image tokens", () => {
    expect(uniformFrameTimes(60, 999)).toHaveLength(MAX_FRAMES);
  });

  it("returns at least one frame for any positive count", () => {
    expect(uniformFrameTimes(10, 0)).toHaveLength(1);
    expect(uniformFrameTimes(10, -3)).toHaveLength(1);
  });

  it("returns nothing for a clip with no duration", () => {
    expect(uniformFrameTimes(0, 4)).toEqual([]);
    expect(uniformFrameTimes(Number.NaN, 4)).toEqual([]);
  });

  it("is strictly increasing, so the filmstrip reads left to right", () => {
    const times = uniformFrameTimes(12, 8);
    for (let i = 1; i < times.length; i++) {
      expect(times[i]).toBeGreaterThan(times[i - 1]);
    }
  });
});

describe("orderFrames", () => {
  it("keeps the sampled order by default", () => {
    expect(orderFrames([1, 2, 3], false)).toEqual([1, 2, 3]);
  });

  it("reverses the list that reaches the model", () => {
    expect(orderFrames([1, 2, 3], true)).toEqual([3, 2, 1]);
  });

  it("never mutates the caller's array", () => {
    // The page keeps the sampled frames for the filmstrip, which must go on
    // showing what was sampled rather than what was last sent.
    const frames = [1, 2, 3];
    orderFrames(frames, true);
    expect(frames).toEqual([1, 2, 3]);
  });
});
