// Compositing an alpha matte — the half of `/background-removal` that is ours.
//
// The model's job ends with a per-pixel alpha. Everything a user actually wants
// from the page happens after that: the subject over transparency, the subject
// over a chosen colour, and the matte itself shown as a picture so the edge can
// be judged rather than assumed.
//
// **The one rule the whole file exists to enforce: never threshold the alpha.**
// A matting model earns its keep on the pixels that are *partly* subject — the
// strand of hair with sky behind it, the motion-blurred edge of a hand. Rounding
// those to 0 or 255 turns a matting model into a segmentation model with extra
// steps, produces the jagged cut-out that makes background removal look cheap,
// and cannot be undone downstream. So the composite is a real linear blend and
// the PNG keeps its alpha channel.
//
// Everything here is pure over pixel buffers, which is what lets the blend be
// tested at a mid-alpha pixel — the assertion that a hard threshold would fail
// and no amount of looking at a canvas in jsdom would catch.

import { drawPixels, type PixelBuffer, type Pixels } from "./draw";
import type { PlainImage } from "./serialize";

/** An RGBA buffer: what the `background-removal` pipeline hands back. */
export type RgbaImage = PlainImage;

/** A background to drop the subject onto, as 0–255 RGB. */
export type Rgb = readonly [number, number, number];

export const WHITE: Rgb = [255, 255, 255];

/**
 * Blend an RGBA image onto an opaque background colour.
 *
 * `out = src * a + bg * (1 - a)`, per channel, with `a` the matte's own value —
 * not a rounded one. A pixel at alpha 128 comes back halfway between the
 * subject and the background, which is exactly what a soft edge is.
 *
 * The result is opaque (alpha 255): it is a finished picture, not a cut-out.
 */
export function compositeOver(image: RgbaImage, background: Rgb): PixelBuffer {
  const { width, height } = image;
  const out = new Uint8ClampedArray(width * height * 4);
  const [br, bg, bb] = background;

  for (let i = 0; i < width * height; i++) {
    const o = i * 4;
    const a = image.data[o + 3] / 255;
    out[o] = image.data[o] * a + br * (1 - a);
    out[o + 1] = image.data[o + 1] * a + bg * (1 - a);
    out[o + 2] = image.data[o + 2] * a + bb * (1 - a);
    out[o + 3] = 255;
  }

  return { data: out, width, height, channels: 4 };
}

/**
 * The matte on its own, as a greyscale RGBA image — white is subject, black is
 * background, and every value between is a value the model actually produced.
 *
 * This view is the page's honesty check. A cut-out on a checkerboard hides a bad
 * matte behind the subject's own colours; the matte shown flat does not.
 */
export function matteToGrey(image: RgbaImage): PixelBuffer {
  const { width, height } = image;
  const out = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < width * height; i++) {
    const o = i * 4;
    const a = image.data[o + 3];
    out[o] = a;
    out[o + 1] = a;
    out[o + 2] = a;
    out[o + 3] = 255;
  }

  return { data: out, width, height, channels: 4 };
}

/**
 * Mean alpha over the image, 0–1 — "how much of this picture is subject".
 *
 * The number the `@slow` spec asserts a band on, because the two ways this page
 * breaks in a way nothing else notices are a matte that is entirely on and one
 * that is entirely off. Both render as a perfectly clean-looking result: all-on
 * is the original photo, all-off is an empty checkerboard, and neither throws.
 */
export function coveredFraction(image: RgbaImage): number {
  const count = image.width * image.height;
  if (count === 0) return 0;
  let sum = 0;
  for (let i = 0; i < count; i++) sum += image.data[i * 4 + 3];
  return sum / (count * 255);
}

/** Mean alpha over the four corner regions — near 0 on a working cut-out. */
export function cornerFraction(image: RgbaImage, side = 8): number {
  const { width, height } = image;
  const w = Math.min(side, width);
  const h = Math.min(side, height);
  if (w === 0 || h === 0) return 0;

  let sum = 0;
  let n = 0;
  for (const [x0, y0] of [
    [0, 0],
    [width - w, 0],
    [0, height - h],
    [width - w, height - h],
  ]) {
    for (let y = y0; y < y0 + h; y++) {
      for (let x = x0; x < x0 + w; x++) {
        sum += image.data[(y * width + x) * 4 + 3];
        n++;
      }
    }
  }
  return n === 0 ? 0 : sum / (n * 255);
}

// --- Canvas -----------------------------------------------------------------

/** Size of one checkerboard square, in source pixels. */
const CHECKER = 12;

/**
 * Paint the transparency checkerboard, then the cut-out over it.
 *
 * The checkerboard is not decoration: without it a cut-out on a white card is
 * indistinguishable from a cut-out on a white *background*, and the user cannot
 * tell whether the alpha channel exists at all. Painted here rather than as a
 * CSS background so the downloaded PNG and the preview are the same pixels
 * modulo the checks.
 */
export function drawCutout(
  ctx: CanvasRenderingContext2D,
  image: RgbaImage,
): void {
  const { width, height } = image;

  for (let y = 0; y < height; y += CHECKER) {
    for (let x = 0; x < width; x += CHECKER) {
      const dark = ((x / CHECKER) | 0) % 2 === ((y / CHECKER) | 0) % 2;
      ctx.fillStyle = dark ? "#e5e7eb" : "#f8fafc";
      ctx.fillRect(x, y, CHECKER, CHECKER);
    }
  }

  drawRgba(ctx, image, { blend: true });
}

/**
 * Paint an RGBA buffer into a canvas.
 *
 * `blend` routes through a scratch canvas and `drawImage`, because
 * `putImageData` **replaces** pixels including their alpha — writing a cut-out
 * straight onto `ctx` would erase the checkerboard underneath instead of
 * showing through it. Same trap, and the same fix, as `drawMasks` in `draw.ts`.
 */
export function drawRgba(
  ctx: CanvasRenderingContext2D,
  image: RgbaImage,
  { blend = false }: { blend?: boolean } = {},
): void {
  const { width, height } = image;
  const img = ctx.createImageData(width, height);
  const count = Math.min(image.data.length, width * height * 4);
  for (let i = 0; i < count; i++) img.data[i] = image.data[i];

  if (!blend) {
    ctx.putImageData(img, 0, 0);
    return;
  }

  const scratch = document.createElement("canvas");
  scratch.width = width;
  scratch.height = height;
  const sctx = scratch.getContext("2d");
  if (!sctx) return;
  sctx.putImageData(img, 0, 0);
  ctx.drawImage(scratch, 0, 0);
}

/**
 * Encode an RGBA buffer as a PNG blob, alpha channel intact.
 *
 * PNG rather than JPEG for exactly one reason, and it is the whole feature: JPEG
 * has no alpha, so a "download the cut-out" button that produced a JPEG would
 * hand back the subject on an arbitrary opaque background. Resolves null when
 * the canvas has no 2-D context (happy-dom, a starved tab).
 */
export async function toPngBlob(image: Pixels): Promise<Blob | null> {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  // `drawPixels` handles 1/3/4 channels and carries a 4th channel through as
  // alpha, so this is the same call for a cut-out and for an upscaled RGB frame.
  drawPixels(ctx, image);
  return new Promise((resolve) =>
    canvas.toBlob((blob) => resolve(blob), "image/png"),
  );
}
