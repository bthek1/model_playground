// The baseline a super-resolution model has to beat.
//
// A 2x image on its own proves nothing: any upscaler produces one, and the eye
// has no reference to judge it against. What shows the model doing work is the
// *same* input upscaled the boring way, side by side — so `/super-resolution`
// renders a draggable split between the model's output and a plain bicubic
// resize, and the `@slow` spec compares them by PSNR against the original.
//
// The browser already has a good bicubic-ish resampler: `drawImage` onto a
// smaller-or-larger canvas with `imageSmoothingQuality: "high"`. Reimplementing
// Catmull-Rom by hand would be slower, would differ between our version and the
// one the browser uses to display the picture, and would be one more thing to be
// wrong. This wraps it so the page has one call.

import type { Pixels } from "./draw";

/**
 * Resample a pixel buffer to `width x height` through the canvas.
 *
 * Returns the input untouched when it is already the requested size, so a page
 * that always calls this pays nothing in the common case. Returns the input
 * unchanged rather than throwing when there is no 2-D context (happy-dom, a
 * starved tab) — the comparison view degrades to "no baseline", which is a
 * missing feature, not a broken page.
 */
export function resample(source: Pixels, width: number, height: number): Pixels {
  if (source.width === width && source.height === height) return source;

  const from = document.createElement("canvas");
  from.width = source.width;
  from.height = source.height;
  const fctx = from.getContext("2d");
  if (!fctx) return source;

  const img = fctx.createImageData(source.width, source.height);
  const count = source.width * source.height;
  const c = source.channels;
  for (let i = 0; i < count; i++) {
    const s = i * c;
    const o = i * 4;
    if (c === 1) {
      img.data[o] = img.data[o + 1] = img.data[o + 2] = source.data[s];
      img.data[o + 3] = 255;
    } else {
      img.data[o] = source.data[s];
      img.data[o + 1] = source.data[s + 1];
      img.data[o + 2] = source.data[s + 2];
      img.data[o + 3] = c === 4 ? source.data[s + 3] : 255;
    }
  }
  fctx.putImageData(img, 0, 0);

  const to = document.createElement("canvas");
  to.width = width;
  to.height = height;
  const tctx = to.getContext("2d");
  if (!tctx) return source;
  // Without these two the browser falls back to nearest-neighbour on some
  // paths, and the "baseline" becomes a pixel-doubled image — which any model
  // beats trivially, turning the comparison into flattery.
  tctx.imageSmoothingEnabled = true;
  tctx.imageSmoothingQuality = "high";
  tctx.drawImage(from, 0, 0, width, height);

  const out = tctx.getImageData(0, 0, width, height);
  return { data: out.data, width, height, channels: 4 };
}

/** Upscale by a whole-number factor — the comparison baseline. */
export function upscale(source: Pixels, scale: number): Pixels {
  return resample(source, source.width * scale, source.height * scale);
}
