import { describe, expect, it, vi } from "vitest";

import {
  compositeOver,
  cornerFraction,
  coveredFraction,
  drawCutout,
  drawRgba,
  matteToGrey,
  WHITE,
} from "./matte";
import type { PixelBuffer } from "./draw";

/** An RGBA image whose alpha is supplied per pixel. */
function rgba(
  width: number,
  height: number,
  colour: [number, number, number],
  alpha: number | ((i: number) => number),
): PixelBuffer {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = colour[0];
    data[i * 4 + 1] = colour[1];
    data[i * 4 + 2] = colour[2];
    data[i * 4 + 3] = typeof alpha === "number" ? alpha : alpha(i);
  }
  return { data, width, height, channels: 4 };
}

/**
 * Stand in for the scratch canvas `drawRgba({ blend: true })` allocates.
 * happy-dom's `getContext("2d")` returns null, and the function then returns
 * early — so without this the blend assertions would be vacuous.
 */
function stubScratchCanvas() {
  const scratchCtx = fakeCtx().ctx;
  const scratch = {
    width: 0,
    height: 0,
    getContext: () => scratchCtx,
  } as unknown as HTMLCanvasElement;
  vi.spyOn(document, "createElement").mockReturnValueOnce(scratch);
}

/** A 2-D context stand-in: happy-dom has no canvas rasteriser. */
function fakeCtx() {
  const put = vi.fn();
  const fillRect = vi.fn();
  const drawImage = vi.fn();
  return {
    ctx: {
      fillRect,
      drawImage,
      createImageData: (w: number, h: number) => ({
        data: new Uint8ClampedArray(w * h * 4),
        width: w,
        height: h,
      }),
      putImageData: put,
      fillStyle: "",
    } as unknown as CanvasRenderingContext2D,
    put,
    fillRect,
    drawImage,
  };
}

describe("compositeOver", () => {
  it("reproduces the source where the matte is fully opaque", () => {
    const src = rgba(2, 2, [200, 100, 50], 255);
    const out = compositeOver(src, WHITE);
    expect([...out.data.slice(0, 4)]).toEqual([200, 100, 50, 255]);
  });

  it("produces the replacement background where the matte is fully clear", () => {
    const src = rgba(2, 2, [200, 100, 50], 0);
    const out = compositeOver(src, [0, 0, 255]);
    expect([...out.data.slice(0, 4)]).toEqual([0, 0, 255, 255]);
  });

  it("blends a mid-alpha pixel instead of snapping it to either side", () => {
    // The soft-matte guard, and the assertion this whole module exists for. A
    // hard threshold on the alpha — the obvious "simplification" — passes both
    // tests above and fails this one, while still producing a picture that
    // looks fine until you zoom into a strand of hair.
    const src = rgba(1, 1, [255, 255, 255], 128);
    const out = compositeOver(src, [0, 0, 0]);

    const r = out.data[0];
    expect(r).toBeGreaterThan(0);
    expect(r).toBeLessThan(255);
    // 255 * (128/255) + 0 = 128, give or take the clamp's rounding.
    expect(r).toBeCloseTo(128, -1);
  });

  it("blends each channel against its own background component", () => {
    const src = rgba(1, 1, [0, 0, 0], 128);
    const out = compositeOver(src, [255, 100, 0]);
    expect(out.data[0]).toBeCloseTo(128, -1);
    expect(out.data[1]).toBeCloseTo(50, -1);
    expect(out.data[2]).toBe(0);
  });

  it("returns an opaque image — it is a picture, not a cut-out", () => {
    const out = compositeOver(rgba(2, 2, [1, 2, 3], 40), WHITE);
    for (let i = 0; i < 4; i++) expect(out.data[i * 4 + 3]).toBe(255);
  });
});

describe("matteToGrey", () => {
  it("paints the alpha as a grey level, preserving the values between", () => {
    const src = rgba(3, 1, [255, 0, 0], (i) => [0, 128, 255][i]);
    const out = matteToGrey(src);
    expect([out.data[0], out.data[4], out.data[8]]).toEqual([0, 128, 255]);
    // Grey, so the subject's own colours cannot flatter a bad matte.
    expect(out.data[4]).toBe(out.data[5]);
    expect(out.data[5]).toBe(out.data[6]);
  });
});

describe("coveredFraction", () => {
  it("is 1 for an all-subject matte and 0 for an empty one", () => {
    // Both are what a broken preprocessing path produces, and both render as a
    // perfectly clean-looking result — the original photo, or nothing at all.
    expect(coveredFraction(rgba(4, 4, [0, 0, 0], 255))).toBe(1);
    expect(coveredFraction(rgba(4, 4, [0, 0, 0], 0))).toBe(0);
  });

  it("averages the soft values rather than counting covered pixels", () => {
    expect(coveredFraction(rgba(2, 1, [0, 0, 0], 128))).toBeCloseTo(128 / 255, 5);
  });

  it("is 0 rather than NaN for an empty image", () => {
    expect(coveredFraction({ data: [], width: 0, height: 0, channels: 4 })).toBe(0);
  });
});

describe("cornerFraction", () => {
  it("reads the corners, not the middle", () => {
    // A subject in the centre leaves the corners transparent; that is the
    // cheapest check that the matte is a cut-out and not the whole frame.
    const image = rgba(8, 8, [0, 0, 0], 0);
    for (let y = 3; y < 5; y++) {
      for (let x = 3; x < 5; x++) image.data[(y * 8 + x) * 4 + 3] = 255;
    }
    expect(cornerFraction(image, 2)).toBe(0);
    expect(coveredFraction(image)).toBeGreaterThan(0);
  });

  it("is 1 when the matte covers everything", () => {
    expect(cornerFraction(rgba(8, 8, [0, 0, 0], 255), 2)).toBe(1);
  });
});

describe("drawRgba", () => {
  it("replaces pixels by default", () => {
    const { ctx, put, drawImage } = fakeCtx();
    drawRgba(ctx, rgba(2, 2, [1, 2, 3], 255));
    expect(put).toHaveBeenCalledOnce();
    expect(drawImage).not.toHaveBeenCalled();
  });

  it("blends through a scratch canvas when asked", () => {
    // `putImageData` replaces alpha as well as colour, so writing a cut-out
    // straight onto the context would erase whatever is beneath it instead of
    // showing through. `drawImage` is the call that actually blends.
    stubScratchCanvas();
    const { ctx, drawImage } = fakeCtx();
    drawRgba(ctx, rgba(2, 2, [1, 2, 3], 128), { blend: true });
    expect(drawImage).toHaveBeenCalledOnce();
  });
});

describe("drawCutout", () => {
  it("paints a checkerboard under the cut-out, then blends it over", () => {
    // Without the checks a cut-out on a white card is indistinguishable from a
    // subject on a white background, and the user cannot tell whether the alpha
    // channel survived at all.
    stubScratchCanvas();
    const { ctx, fillRect, drawImage } = fakeCtx();
    drawCutout(ctx, rgba(24, 24, [1, 2, 3], 200));
    expect(fillRect).toHaveBeenCalled();
    expect(drawImage).toHaveBeenCalledOnce();
  });
});
