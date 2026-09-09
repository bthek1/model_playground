import { describe, expect, it, vi } from "vitest";

import {
  colorForLabel,
  drawBoxes,
  drawHeatmap,
  drawMasks,
  rangeOf,
  OVERLAY_COLORS,
} from "./draw";

/** A 2-D context stand-in: happy-dom has no canvas rasteriser. */
function fakeCtx(width = 4, height = 4) {
  const put = vi.fn();
  return {
    ctx: {
      save: vi.fn(),
      restore: vi.fn(),
      strokeRect: vi.fn(),
      fillRect: vi.fn(),
      fillText: vi.fn(),
      drawImage: vi.fn(),
      measureText: vi.fn(() => ({ width: 40 })),
      createImageData: (w: number, h: number) => ({
        data: new Uint8ClampedArray(w * h * 4),
        width: w,
        height: h,
      }),
      putImageData: put,
      lineWidth: 0,
      font: "",
      textBaseline: "",
      strokeStyle: "",
      fillStyle: "",
    } as unknown as CanvasRenderingContext2D,
    put,
    width,
    height,
  };
}

describe("colorForLabel", () => {
  it("is stable for the same label", () => {
    // A class that changes colour between frames of a live feed is worse than
    // no colour at all.
    expect(colorForLabel("person")).toBe(colorForLabel("person"));
  });

  it("only ever returns a palette colour", () => {
    for (const label of ["person", "car", "dog", "", "a very long label indeed"]) {
      expect(OVERLAY_COLORS).toContain(colorForLabel(label));
    }
  });
});

describe("drawBoxes", () => {
  it("draws one rect per detection, in absolute pixels", () => {
    const { ctx } = fakeCtx();
    drawBoxes(ctx, [
      { box: { xmin: 10, ymin: 20, xmax: 110, ymax: 220 }, label: "person", score: 0.91 },
      { box: { xmin: 0, ymin: 0, xmax: 50, ymax: 50 }, label: "car", score: 0.5 },
    ]);

    expect(ctx.strokeRect).toHaveBeenCalledTimes(2);
    expect(ctx.strokeRect).toHaveBeenNthCalledWith(1, 10, 20, 100, 200);
    expect(ctx.strokeRect).toHaveBeenNthCalledWith(2, 0, 0, 50, 50);
  });

  it("labels each box with its score", () => {
    const { ctx } = fakeCtx();
    drawBoxes(ctx, [
      { box: { xmin: 10, ymin: 40, xmax: 60, ymax: 90 }, label: "person", score: 0.912 },
    ]);
    expect(ctx.fillText).toHaveBeenCalledWith("person 0.91", expect.any(Number), expect.any(Number));
  });

  it("keeps the caption on-canvas for a box against the top edge", () => {
    const { ctx } = fakeCtx();
    drawBoxes(ctx, [
      { box: { xmin: 5, ymin: 0, xmax: 60, ymax: 60 }, label: "car", score: 0.7 },
    ]);
    const [, y] = (ctx.fillText as unknown as { mock: { calls: number[][] } }).mock.calls[0];
    expect(y).toBeGreaterThanOrEqual(0);
  });
});

describe("rangeOf", () => {
  it("reports the min and max actually present", () => {
    expect(rangeOf(new Float32Array([3, -1, 7]))).toEqual({ lo: -1, hi: 7 });
  });

  it("falls back to a usable range for empty data", () => {
    expect(rangeOf(new Float32Array([]))).toEqual({ lo: 0, hi: 1 });
  });
});

describe("drawHeatmap", () => {
  it("normalises to the values present, so an arbitrary scale still renders", () => {
    // A relative-depth map is on no particular scale. Without the per-map
    // rescale the canvas comes out uniformly black or uniformly white.
    const { ctx, put } = fakeCtx();
    const data = new Float32Array([100, 150, 200, 250]);
    const bounds = drawHeatmap(ctx, data, 2, 2);

    expect(bounds).toEqual({ lo: 100, hi: 250 });
    const img = put.mock.calls[0][0] as ImageData;
    const first = Array.from(img.data.slice(0, 3));
    const last = Array.from(img.data.slice(12, 15));
    expect(first).not.toEqual(last);
    // Monotonic in lightness: the far end of the ramp is brighter.
    const sum = (c: number[]) => c.reduce((a, b) => a + b, 0);
    expect(sum(last)).toBeGreaterThan(sum(first));
  });

  it("does not divide by zero on a constant map", () => {
    const { ctx, put } = fakeCtx();
    expect(() => drawHeatmap(ctx, new Float32Array([5, 5, 5, 5]), 2, 2)).not.toThrow();
    const img = put.mock.calls[0][0] as ImageData;
    expect(Array.from(img.data).every((v) => Number.isFinite(v))).toBe(true);
  });

  it("honours explicit bounds, for a scale held fixed across frames", () => {
    const { ctx } = fakeCtx();
    expect(drawHeatmap(ctx, new Float32Array([1, 2]), 2, 1, { lo: 0, hi: 10 })).toEqual({
      lo: 0,
      hi: 10,
    });
  });
});

describe("drawMasks", () => {
  it("composites every mask into one canvas rather than N images", () => {
    const scratchCtx = fakeCtx().ctx;
    const scratch = { width: 0, height: 0, getContext: () => scratchCtx } as unknown as HTMLCanvasElement;
    vi.spyOn(document, "createElement").mockReturnValueOnce(scratch);

    const { ctx } = fakeCtx();
    drawMasks(
      ctx,
      [
        { label: "sky", data: new Float32Array([1, 1, 0, 0]), width: 2, height: 2 },
        { label: "road", data: new Float32Array([0, 0, 1, 1]), width: 2, height: 2 },
      ],
      { width: 2, height: 2 },
    );

    // One blend onto the target, so the source image underneath survives:
    // `putImageData` would have replaced it, alpha and all.
    expect(ctx.drawImage).toHaveBeenCalledOnce();
    expect(ctx.putImageData).not.toHaveBeenCalled();

    const img = (scratchCtx.putImageData as unknown as { mock: { calls: ImageData[][] } }).mock.calls[0][0];
    const sky = Array.from(img.data.slice(0, 3));
    const road = Array.from(img.data.slice(8, 11));
    expect(sky).not.toEqual(road);
    expect(img.data[3]).toBeGreaterThan(0);
  });

  it("leaves uncovered pixels transparent", () => {
    const scratchCtx = fakeCtx().ctx;
    const scratch = { width: 0, height: 0, getContext: () => scratchCtx } as unknown as HTMLCanvasElement;
    vi.spyOn(document, "createElement").mockReturnValueOnce(scratch);

    const { ctx } = fakeCtx();
    drawMasks(ctx, [{ label: "sky", data: new Float32Array([1, 0]), width: 2, height: 1 }], {
      width: 2,
      height: 1,
    });

    const img = (scratchCtx.putImageData as unknown as { mock: { calls: ImageData[][] } }).mock.calls[0][0];
    expect(img.data[3]).toBeGreaterThan(0); // covered
    expect(img.data[7]).toBe(0); // not covered
  });

  it("does nothing at all with no masks", () => {
    const { ctx } = fakeCtx();
    drawMasks(ctx, [], { width: 2, height: 2 });
    expect(ctx.drawImage).not.toHaveBeenCalled();
  });
});
