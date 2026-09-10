import { describe, expect, it, vi } from "vitest";

import {
  COCO_EDGES,
  COCO_KEYPOINTS,
  drawSkeleton,
  JOINT_CONFIDENCE,
  jointAlpha,
  type Person,
} from "./skeleton";

describe("COCO_KEYPOINTS", () => {
  it("has the 17 joints, in the order the model emits them", () => {
    // The index *is* the label — `post_process_pose_estimation` returns
    // `labels: number[]` and nothing else names them. An order that drifts
    // relabels every joint silently.
    expect(COCO_KEYPOINTS).toHaveLength(17);
    expect(COCO_KEYPOINTS[0]).toBe("nose");
    expect(COCO_KEYPOINTS[5]).toBe("left shoulder");
    expect(COCO_KEYPOINTS[16]).toBe("right ankle");
  });
});

describe("COCO_EDGES", () => {
  it("only ever names joints that exist", () => {
    for (const [a, b] of COCO_EDGES) {
      expect(COCO_KEYPOINTS[a], `edge start ${a}`).toBeDefined();
      expect(COCO_KEYPOINTS[b], `edge end ${b}`).toBeDefined();
    }
  });

  it("connects both arms and both legs to the torso", () => {
    const has = (a: number, b: number) =>
      COCO_EDGES.some(([x, y]) => (x === a && y === b) || (x === b && y === a));
    expect(has(5, 7), "left shoulder to left elbow").toBe(true);
    expect(has(7, 9), "left elbow to left wrist").toBe(true);
    expect(has(12, 14), "right hip to right knee").toBe(true);
    expect(has(14, 16), "right knee to right ankle").toBe(true);
    expect(has(5, 6), "shoulder to shoulder").toBe(true);
  });
});

describe("jointAlpha", () => {
  it("is fully opaque at and above the confidence floor", () => {
    expect(jointAlpha(JOINT_CONFIDENCE)).toBe(1);
    expect(jointAlpha(0.99)).toBe(1);
  });

  it("fades below it, but never to nothing", () => {
    // "The model put it here and does not believe it" is information; a joint
    // that vanishes is indistinguishable from one the model did not return.
    expect(jointAlpha(0)).toBeGreaterThan(0);
    expect(jointAlpha(0)).toBeLessThan(0.3);
    expect(jointAlpha(JOINT_CONFIDENCE / 2)).toBeLessThan(1);
    expect(jointAlpha(JOINT_CONFIDENCE / 2)).toBeGreaterThan(jointAlpha(0));
  });
});

/** A minimal 2D context recorder — happy-dom has no canvas implementation. */
function fakeCtx() {
  const alphas: number[] = [];
  const ctx = {
    save: vi.fn(),
    restore: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(() => alphas.push(ctx.globalAlpha)),
    arc: vi.fn(),
    fill: vi.fn(),
    lineWidth: 0,
    lineCap: "",
    strokeStyle: "",
    fillStyle: "",
    globalAlpha: 1,
  };
  return { ctx, alphas };
}

describe("drawSkeleton", () => {
  const person = (scores: Partial<Record<number, number>>): Person => ({
    box: { xmin: 0, ymin: 0, xmax: 10, ymax: 10 },
    score: 0.9,
    keypoints: COCO_KEYPOINTS.map((_, index) => ({
      x: index,
      y: index,
      score: scores[index] ?? 1,
      index,
    })),
  });

  it("draws every limb and every joint", () => {
    const { ctx } = fakeCtx();
    drawSkeleton(ctx as unknown as CanvasRenderingContext2D, person({}), "#fff");
    expect(ctx.stroke).toHaveBeenCalledTimes(COCO_EDGES.length);
    expect(ctx.arc).toHaveBeenCalledTimes(17);
  });

  it("draws a limb at its weakest end's confidence, not the average", () => {
    // An edge is only as believable as its weakest joint; averaging lets one
    // solid joint carry a guessed one into looking certain.
    const { ctx, alphas } = fakeCtx();
    drawSkeleton(
      ctx as unknown as CanvasRenderingContext2D,
      person({ 9: 0 }), // left wrist unknown
      "#fff",
    );
    // The left elbow→wrist edge is [7, 9]; every other edge is fully confident.
    const faded = alphas.filter((a) => a < 1);
    expect(faded).toHaveLength(1);
    expect(faded[0]).toBeCloseTo(jointAlpha(0), 6);
  });

  it("draws nothing for a person with no keypoints", () => {
    const { ctx } = fakeCtx();
    drawSkeleton(
      ctx as unknown as CanvasRenderingContext2D,
      { box: { xmin: 0, ymin: 0, xmax: 1, ymax: 1 }, score: 1, keypoints: [] },
      "#fff",
    );
    expect(ctx.stroke).not.toHaveBeenCalled();
    expect(ctx.arc).not.toHaveBeenCalled();
  });

  it("skips an edge whose joints were not returned", () => {
    const { ctx } = fakeCtx();
    drawSkeleton(
      ctx as unknown as CanvasRenderingContext2D,
      {
        box: { xmin: 0, ymin: 0, xmax: 1, ymax: 1 },
        score: 1,
        keypoints: [
          { x: 0, y: 0, score: 1, index: 0 },
          { x: 1, y: 1, score: 1, index: 1 },
        ],
      },
      "#fff",
    );
    // Only the nose→left-eye edge exists among these two.
    expect(ctx.stroke).toHaveBeenCalledTimes(1);
  });
});
