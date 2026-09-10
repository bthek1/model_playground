// Overlapping-tile inference, and the seam blending that makes it invisible.
//
// A super-resolution transformer on a full-resolution photo will exhaust memory:
// Swin2SR's attention is quadratic in the tile area, and a 12 MP input asks for
// a 48 MP output in one allocation. So the image is cut into tiles, each tile is
// run on its own, and the results are stitched back together.
//
// **The stitch is the whole problem.** Butt the tiles edge to edge and every
// seam is visible as a faint line, because two adjacent tiles saw different
// context and reconstructed the boundary differently. So tiles overlap, and each
// tile's contribution is feathered to zero across the overlap — a weighted
// average, accumulated per pixel and normalised at the end. Normalising by the
// *accumulated weight* rather than trusting the weights to sum to one is what
// makes the identity case exact: run tiles through a model that returns them
// unchanged and reassembly reproduces the input pixel for pixel, which is the
// test that catches a mis-placed tile, a wrong stride, or a feather that leaks.
//
// Everything here is pure arithmetic over plain buffers — no canvas, no model,
// no worker. That is deliberate: the failure mode of tiling is geometric, and
// geometry is exactly what a unit test can pin and a rendered picture cannot.
//
// ## One thing about Swin2SR specifically
//
// `Swin2SRImageProcessor` pads the input up to a multiple of `pad_size` (8) by
// reflection, and the model upscales what it was given. So a 100x100 tile comes
// back 208x208, not 200x200. Tile sizes here are multiples of 8 so this never
// fires in the interior, and {@link blendTile} reads only the top-left
// `scale x tile` region regardless — the padding is on the bottom and right, so
// cropping the top-left keeps the real reconstruction. Without that crop the
// last row and column of tiles would be 8-16 px too large and every tile after
// the first would land slightly off, which reads as a soft, doubled image rather
// than as a bug.

import type { PixelBuffer, Pixels } from "./draw";
// Type-only, so this module stays free of the Transformers.js runtime and cheap
// to unit-test: a cropped tile *is* an `ImagePayload`, and saying so here means
// the caller can post it to the worker without a cast.
import type { ImagePayload } from "./image";

/** A rectangle of the **source** image, in source pixels. */
export interface Tile {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface TilePlan {
  tiles: Tile[];
  /** Feather width in source pixels — the overlap actually used. */
  overlap: number;
}

/**
 * Tile side length, in source pixels. 192 is a multiple of both 8 (Swin2SR's
 * pad size) and its window size, and 192x192 -> 384x384 is comfortably inside a
 * tab's memory budget while still being large enough that a tile carries real
 * context rather than a postage stamp of it.
 */
export const TILE_SIDE = 192;

/** Overlap between neighbouring tiles, in source pixels. Also a multiple of 8. */
export const TILE_OVERLAP = 32;

/**
 * Start offsets along one axis. The last tile is flush with the far edge rather
 * than hanging off it, so it may overlap its neighbour by more than `overlap` —
 * harmless, because the feather is a weighted average and not a hard cut.
 */
export function tileStarts(
  length: number,
  tile: number,
  overlap: number,
): number[] {
  if (length <= tile) return [0];
  const stride = Math.max(1, tile - overlap);
  const starts: number[] = [];
  for (let s = 0; s + tile < length; s += stride) starts.push(s);
  starts.push(length - tile);
  return starts;
}

/**
 * Cover `width x height` with overlapping tiles.
 *
 * An image smaller than one tile yields exactly one tile covering all of it, and
 * that tile goes through the identical blend path as one of twenty — there is no
 * "small image" shortcut, because a shortcut is a second code path that only the
 * rare input exercises.
 */
export function planTiles(
  width: number,
  height: number,
  { tile = TILE_SIDE, overlap = TILE_OVERLAP }: { tile?: number; overlap?: number } = {},
): TilePlan {
  const tiles: Tile[] = [];
  const xs = tileStarts(width, tile, overlap);
  const ys = tileStarts(height, tile, overlap);

  for (const y of ys) {
    for (const x of xs) {
      tiles.push({
        x,
        y,
        width: Math.min(tile, width),
        height: Math.min(tile, height),
      });
    }
  }

  return { tiles, overlap };
}

/**
 * Cut a tile out of a source buffer, as its own buffer.
 *
 * Returns an `ImagePayload` rather than a bare `Pixels` because the next thing
 * that happens to a tile is always a `postMessage` to the vision worker.
 */
export function cropTile(source: Pixels, tile: Tile): ImagePayload {
  const c = source.channels;
  const out = new Uint8ClampedArray(tile.width * tile.height * c);

  for (let y = 0; y < tile.height; y++) {
    const srcRow = (tile.y + y) * source.width + tile.x;
    const dstRow = y * tile.width;
    for (let x = 0; x < tile.width; x++) {
      for (let k = 0; k < c; k++) {
        out[(dstRow + x) * c + k] = source.data[(srcRow + x) * c + k];
      }
    }
  }

  return {
    data: out,
    width: tile.width,
    height: tile.height,
    // The source is a decoded image, so its channel count is one of the four a
    // payload allows; `Pixels` merely widens it to `number` for the drawing code.
    channels: c as ImagePayload["channels"],
  };
}

/**
 * The accumulator a set of tiles is blended into: a weighted colour sum plus the
 * weight that produced it, per pixel. Kept separate from the finished image so
 * {@link finishCanvas} can divide once at the end rather than after every tile.
 */
export interface TileCanvas {
  sum: Float32Array;
  weight: Float32Array;
  width: number;
  height: number;
  channels: number;
}

export function createTileCanvas(
  width: number,
  height: number,
  channels: number,
): TileCanvas {
  return {
    sum: new Float32Array(width * height * channels),
    weight: new Float32Array(width * height),
    width,
    height,
    channels,
  };
}

/**
 * Feather weight along one axis: 1 in the middle, ramping to nearly 0 at any
 * edge that has a neighbour on the other side.
 *
 * The `+ 0.5` matters. Without it the outermost column of an interior edge gets
 * weight exactly 0, and a pixel where two tiles both weigh 0 has no colour at
 * all — a division by zero, which appears as a black seam precisely where the
 * blending was supposed to hide one.
 */
function edgeWeight(i: number, span: number, feather: number, lo: boolean, hi: boolean): number {
  const f = Math.max(1, feather);
  const left = lo ? Math.min(1, (i + 0.5) / f) : 1;
  const right = hi ? Math.min(1, (span - i - 0.5) / f) : 1;
  return Math.min(left, right);
}

/**
 * Blend one model output into the accumulator.
 *
 * `patch` is what the model returned for `tile`, at `scale x` its size — or
 * larger, if the processor padded (see the note at the top). Only the top-left
 * `scale x tile` region is read.
 */
export function blendTile(
  canvas: TileCanvas,
  patch: Pixels,
  tile: Tile,
  {
    scale,
    overlap = TILE_OVERLAP,
    source,
  }: { scale: number; overlap?: number; source: { width: number; height: number } },
): void {
  const c = canvas.channels;
  const ow = tile.width * scale;
  const oh = tile.height * scale;
  const feather = overlap * scale;

  // Which sides have a neighbour to blend against. An image edge is not
  // feathered: fading the true border to nothing would darken the frame.
  const fadeLeft = tile.x > 0;
  const fadeTop = tile.y > 0;
  const fadeRight = tile.x + tile.width < source.width;
  const fadeBottom = tile.y + tile.height < source.height;

  const dx = tile.x * scale;
  const dy = tile.y * scale;

  for (let y = 0; y < oh; y++) {
    const wy = edgeWeight(y, oh, feather, fadeTop, fadeBottom);
    const destY = dy + y;
    if (destY < 0 || destY >= canvas.height) continue;

    for (let x = 0; x < ow; x++) {
      const destX = dx + x;
      if (destX < 0 || destX >= canvas.width) continue;

      const w = wy * edgeWeight(x, ow, feather, fadeLeft, fadeRight);
      const dest = destY * canvas.width + destX;
      // `patch.width`, not `ow`: a padded output is wider than the region we
      // read, so the row stride is the patch's own.
      const src = y * patch.width + x;

      for (let k = 0; k < c; k++) {
        canvas.sum[dest * c + k] += patch.data[src * patch.channels + k] * w;
      }
      canvas.weight[dest] += w;
    }
  }
}

/** Divide the accumulated colour by the accumulated weight. */
export function finishCanvas(canvas: TileCanvas): PixelBuffer {
  const { width, height, channels: c } = canvas;
  const out = new Uint8ClampedArray(width * height * c);

  for (let i = 0; i < width * height; i++) {
    const w = canvas.weight[i];
    if (w <= 0) continue; // never reached with a real plan; not worth a NaN
    for (let k = 0; k < c; k++) {
      out[i * c + k] = Math.round(canvas.sum[i * c + k] / w);
    }
  }

  return { data: out, width, height, channels: c };
}

/**
 * Peak signal-to-noise ratio between two same-sized images, in dB.
 *
 * The number the `@slow` spec compares against a bicubic baseline, and the only
 * assertion that can prove the model ran *and* that the tiles were reassembled
 * in the right order — a shuffled reassembly produces a perfectly sharp image
 * with a terrible PSNR, which no "an image appeared" check would notice.
 */
export function psnr(a: Pixels, b: Pixels): number {
  const c = Math.min(a.channels, b.channels);
  const count = Math.min(a.width * a.height, b.width * b.height);
  if (count === 0) return 0;

  let se = 0;
  for (let i = 0; i < count; i++) {
    for (let k = 0; k < c; k++) {
      const d = a.data[i * a.channels + k] - b.data[i * b.channels + k];
      se += d * d;
    }
  }

  const mse = se / (count * c);
  if (mse === 0) return Infinity;
  return 10 * Math.log10((255 * 255) / mse);
}
