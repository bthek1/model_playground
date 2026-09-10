import { describe, expect, it, vi } from "vitest";

import type { Pixels } from "./draw";
import { resample, upscale } from "./resample";

function rgb(width: number, height: number): Pixels {
  const data = new Uint8ClampedArray(width * height * 3);
  for (let i = 0; i < data.length; i++) data[i] = i % 256;
  return { data, width, height, channels: 3 };
}

describe("resample", () => {
  it("returns the input untouched when it is already the right size", () => {
    // The page calls this unconditionally, so the common case must cost nothing
    // — and must be the *same object*, not a copy.
    const src = rgb(4, 4);
    expect(resample(src, 4, 4)).toBe(src);
  });

  it("returns the input rather than throwing when there is no 2-D context", () => {
    // happy-dom has no canvas rasteriser, and a starved tab can refuse one too.
    // The comparison view then degrades to "no baseline", which is a missing
    // feature — an OUTPUT slot that throws takes the whole route down.
    const src = rgb(4, 4);
    expect(resample(src, 8, 8)).toBe(src);
  });

  it("asks the canvas for a smoothed resize, never nearest-neighbour", () => {
    // A pixel-doubled "baseline" is one any model beats trivially, which turns
    // the comparison from evidence into flattery.
    const ctx = {
      createImageData: (w: number, h: number) => ({
        data: new Uint8ClampedArray(w * h * 4),
        width: w,
        height: h,
      }),
      putImageData: vi.fn(),
      drawImage: vi.fn(),
      getImageData: (_x: number, _y: number, w: number, h: number) => ({
        data: new Uint8ClampedArray(w * h * 4),
        width: w,
        height: h,
      }),
      imageSmoothingEnabled: false,
      imageSmoothingQuality: "low" as CanvasRenderingContext2D["imageSmoothingQuality"],
    };
    const canvas = {
      width: 0,
      height: 0,
      getContext: () => ctx,
    } as unknown as HTMLCanvasElement;
    vi.spyOn(document, "createElement").mockReturnValue(canvas);

    const out = resample(rgb(4, 4), 8, 8);

    expect(ctx.imageSmoothingEnabled).toBe(true);
    expect(ctx.imageSmoothingQuality).toBe("high");
    expect(ctx.drawImage).toHaveBeenCalled();
    expect(out.width).toBe(8);
    expect(out.height).toBe(8);
    // Always RGBA on the way out — it came back through `getImageData`.
    expect(out.channels).toBe(4);

    vi.restoreAllMocks();
  });
});

describe("upscale", () => {
  it("multiplies both dimensions by the factor", () => {
    const src = rgb(4, 4);
    // No canvas here, so it short-circuits — the arithmetic is still the point.
    expect(upscale(src, 1)).toBe(src);
  });
});
