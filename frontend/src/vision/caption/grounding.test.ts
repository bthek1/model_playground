import { describe, expect, it } from "vitest";

import { toDetections } from "./grounding";

describe("toDetections", () => {
  it("pairs each box with its label, in source pixels", () => {
    // `post_process_generation` has already mapped the model's `<loc_…>` bins
    // back to pixels, so `[x1, y1, x2, y2]` is exactly what `drawBoxes` wants —
    // there is no `percentage` flag on this path and nothing to scale back.
    expect(
      toDetections({
        labels: ["a cat", "a sofa"],
        bboxes: [
          [10, 20, 110, 220],
          [0, 0, 640, 480],
        ],
      }),
    ).toEqual([
      {
        label: "a cat",
        score: 1,
        box: { xmin: 10, ymin: 20, xmax: 110, ymax: 220 },
      },
      {
        label: "a sofa",
        score: 1,
        box: { xmin: 0, ymin: 0, xmax: 640, ymax: 480 },
      },
    ]);
  });

  it("gives a box with no label a placeholder rather than an empty string", () => {
    // The upstream regex duplicates the previous label when one is missing, and
    // there may not be a previous one.
    const [only] = toDetections({ labels: ["  "], bboxes: [[1, 2, 3, 4]] });
    expect(only.label).toBe("object");
  });

  it("scores every box 1, because there is no confidence to report", () => {
    // Florence-2 writes a location out as tokens rather than ranking
    // candidates. A 0 would be filtered out by anything downstream that
    // thresholds; a 1 says "it committed to this".
    const out = toDetections({ labels: ["x"], bboxes: [[1, 2, 3, 4]] });
    expect(out[0].score).toBe(1);
  });

  it("drops a malformed box instead of losing the good ones with it", () => {
    // A truncated generation can end mid-box. Throwing here would discard
    // eleven correct boxes because the twelfth was cut off.
    expect(
      toDetections({
        labels: ["a", "b", "c"],
        bboxes: [[1, 2, 3, 4], [5, 6], [NaN, 1, 2, 3]],
      }),
    ).toHaveLength(1);
  });

  it("returns nothing for an empty or absent answer", () => {
    expect(toDetections(undefined)).toEqual([]);
    expect(toDetections({})).toEqual([]);
    expect(toDetections({ labels: [], bboxes: [] })).toEqual([]);
  });
});
