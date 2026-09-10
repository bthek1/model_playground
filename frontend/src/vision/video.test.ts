import { describe, expect, it } from "vitest";

import {
  frameTimes,
  MAX_FRAMES,
  VIDEO_SAMPLES,
  wasCapped,
} from "./video";

describe("frameTimes", () => {
  it("samples `duration x fps` frames when the cap allows it", () => {
    expect(frameTimes(10, 2, 100)).toHaveLength(20);
    expect(frameTimes(3, 1, 100)).toHaveLength(3);
    // A fractional duration rounds up: the tail of the clip is still looked at.
    expect(frameTimes(2.5, 2, 100)).toHaveLength(5);
  });

  it("offsets by half a step rather than starting at zero", () => {
    // The first frame of a clip is very often black or a fade, and a baseline
    // whose first data point is "an image of darkness" reads as a model failure
    // rather than an editing convention.
    const times = frameTimes(4, 2, 100);
    expect(times[0]).toBeCloseTo(0.25, 6);
    expect(times[1]).toBeCloseTo(0.75, 6);
  });

  it("enforces the frame cap, whatever the clip's length", () => {
    // This is the page most likely to be handed a ten-minute video, and every
    // frame is a CLIP vision pass. Without the cap a long clip is not slow, it
    // is a hung tab.
    expect(frameTimes(600, 4, MAX_FRAMES)).toHaveLength(MAX_FRAMES);
    expect(frameTimes(600, 4, 10)).toHaveLength(10);
  });

  it("never samples past the end of the clip", () => {
    const times = frameTimes(1, 4, 100);
    expect(Math.max(...times)).toBeLessThan(1);
  });

  it("gives a clip shorter than one step a single frame at its middle", () => {
    expect(frameTimes(0.2, 1, 100)).toEqual([0.1]);
  });

  it("returns nothing for a clip with no duration", () => {
    expect(frameTimes(0, 2, 100)).toEqual([]);
    expect(frameTimes(NaN, 2, 100)).toEqual([]);
    expect(frameTimes(10, 0, 100)).toEqual([]);
    expect(frameTimes(10, 2, 0)).toEqual([]);
  });
});

describe("wasCapped", () => {
  it("is true only when the cap, not the clip, decided the frame count", () => {
    // The page says so next to the filmstrip: a capped run is not a view of the
    // whole clip, and a chart that quietly covers the first minute of a
    // ten-minute video is misleading.
    expect(wasCapped(600, 4, MAX_FRAMES)).toBe(true);
    expect(wasCapped(10, 2, MAX_FRAMES)).toBe(false);
  });
});

describe("VIDEO_SAMPLES", () => {
  it("gives each clip a label set with a plausible distractor in it", () => {
    // A zero-shot score is relative to the labels given, so a clip whose list
    // has only one credible answer demonstrates nothing.
    for (const sample of VIDEO_SAMPLES) {
      expect(sample.labels.length, sample.id).toBeGreaterThanOrEqual(2);
      expect(sample.url, sample.id).toMatch(/^https:\/\/huggingface\.co\//);
    }
  });
});
