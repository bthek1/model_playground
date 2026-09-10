import { describe, expect, it } from "vitest";

import type { PixelBuffer, Pixels } from "./draw";
import {
  blendTile,
  createTileCanvas,
  cropTile,
  finishCanvas,
  planTiles,
  psnr,
  tileStarts,
  TILE_OVERLAP,
  TILE_SIDE,
} from "./tile";

/** A source image whose every pixel is a function of its position. */
function ramp(width: number, height: number, channels = 3): PixelBuffer {
  const data = new Uint8ClampedArray(width * height * channels);
  for (let i = 0; i < width * height; i++) {
    for (let k = 0; k < channels; k++) {
      // Distinct per pixel *and* per channel, so a transposed or shuffled
      // reassembly cannot accidentally match.
      data[i * channels + k] = (i * 7 + k * 53) % 256;
    }
  }
  return { data, width, height, channels };
}

/**
 * Reassemble a whole image by running every tile through `model`. The identity
 * model is the seam test; a scaling model checks the geometry at 2x.
 */
function roundTrip(
  source: Pixels,
  scale: number,
  model: (patch: Pixels) => Pixels,
  opts?: { tile?: number; overlap?: number },
): PixelBuffer {
  const plan = planTiles(source.width, source.height, opts);
  const canvas = createTileCanvas(
    source.width * scale,
    source.height * scale,
    source.channels,
  );
  for (const tile of plan.tiles) {
    blendTile(canvas, model(cropTile(source, tile)), tile, {
      scale,
      overlap: plan.overlap,
      source,
    });
  }
  return finishCanvas(canvas);
}

/** Nearest-neighbour upscale — a stand-in for the model, with known output. */
function nearest(patch: Pixels, scale: number): PixelBuffer {
  const w = patch.width * scale;
  const h = patch.height * scale;
  const c = patch.channels;
  const data = new Uint8ClampedArray(w * h * c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const src = ((y / scale) | 0) * patch.width + ((x / scale) | 0);
      for (let k = 0; k < c; k++) {
        data[(y * w + x) * c + k] = patch.data[src * c + k];
      }
    }
  }
  return { data, width: w, height: h, channels: c };
}

describe("tileStarts", () => {
  it("returns a single tile for an image no larger than one", () => {
    expect(tileStarts(100, 192, 32)).toEqual([0]);
    expect(tileStarts(192, 192, 32)).toEqual([0]);
  });

  it("steps by tile minus overlap, and lands the last tile flush", () => {
    // 500 with 192/32: 0, 160, 320, then the tail flush at 308.
    expect(tileStarts(500, 192, 32)).toEqual([0, 160, 320 - 12]);
  });

  it("never emits a duplicate or a tile that hangs off the edge", () => {
    for (const length of [1, 7, 191, 193, 200, 383, 512, 1000, 1337]) {
      const starts = tileStarts(length, 192, 32);
      expect(new Set(starts).size).toBe(starts.length);
      for (const s of starts) {
        expect(s).toBeGreaterThanOrEqual(0);
        expect(s + Math.min(192, length)).toBeLessThanOrEqual(length);
      }
    }
  });
});

describe("planTiles", () => {
  it("covers every source pixel", () => {
    const { tiles } = planTiles(500, 300);
    const seen = new Uint8Array(500 * 300);
    for (const t of tiles) {
      for (let y = t.y; y < t.y + t.height; y++) {
        for (let x = t.x; x < t.x + t.width; x++) seen[y * 500 + x] = 1;
      }
    }
    expect(seen.every((v) => v === 1)).toBe(true);
  });

  it("gives neighbouring tiles the expected overlap", () => {
    const { tiles } = planTiles(500, 192, { tile: 192, overlap: 32 });
    const xs = [...new Set(tiles.map((t) => t.x))].sort((a, b) => a - b);
    // Interior neighbours are exactly `tile - overlap` apart.
    expect(xs[1] - xs[0]).toBe(TILE_SIDE - TILE_OVERLAP);
    // Every neighbouring pair overlaps by at least the requested amount.
    for (let i = 1; i < xs.length; i++) {
      expect(192 - (xs[i] - xs[i - 1])).toBeGreaterThanOrEqual(32);
    }
  });

  it("clamps the tile to the image when the image is smaller", () => {
    const { tiles } = planTiles(64, 40);
    expect(tiles).toEqual([{ x: 0, y: 0, width: 64, height: 40 }]);
  });
});

describe("cropTile", () => {
  it("lifts the requested rectangle, row strides intact", () => {
    const src = ramp(6, 4);
    const patch = cropTile(src, { x: 2, y: 1, width: 3, height: 2 });
    expect(patch.width).toBe(3);
    expect(patch.height).toBe(2);
    // (x=2, y=1) in the source is index 1*6+2 = 8.
    expect(patch.data[0]).toBe(src.data[8 * 3]);
    // (x=2, y=2) is the start of the patch's second row: index 2*6+2 = 14.
    expect(patch.data[3 * 3]).toBe(src.data[14 * 3]);
  });
});

describe("reassembly", () => {
  it("reproduces the input exactly when the model is the identity", () => {
    // The seam test. A wrong stride, a mis-placed tile or a feather that leaks
    // past its overlap all show up here as a handful of off-by-one pixels —
    // and as a faintly visible line in a rendered picture, which is the thing
    // no rendered picture reliably reveals.
    const src = ramp(500, 300);
    const out = roundTrip(src, 1, (p) => p);

    expect(out.width).toBe(500);
    expect(out.height).toBe(300);
    expect([...out.data]).toEqual([...src.data]);
  });

  it("takes the same path for a one-tile image as for a twenty-tile one", () => {
    // No "small image" shortcut: a second code path that only the rare input
    // exercises is a second code path nobody tests.
    const small = ramp(64, 48);
    expect([...roundTrip(small, 1, (p) => p).data]).toEqual([...small.data]);

    const big = ramp(900, 700);
    expect(planTiles(900, 700).tiles.length).toBeGreaterThan(20);
    expect([...roundTrip(big, 1, (p) => p).data]).toEqual([...big.data]);
  });

  it("produces a canvas exactly `scale` times the input dimensions", () => {
    const src = ramp(300, 220);
    const out = roundTrip(src, 2, (p) => nearest(p, 2));
    expect(out.width).toBe(600);
    expect(out.height).toBe(440);
  });

  it("reassembles a 2x upscale in the right order", () => {
    // Nearest-neighbour is exactly reproducible, so a correct reassembly at 2x
    // is bit-identical to upscaling the whole image at once. A shuffled or
    // offset reassembly stays perfectly sharp and fails here.
    const src = ramp(300, 220);
    const tiled = roundTrip(src, 2, (p) => nearest(p, 2));
    const whole = nearest(src, 2);
    expect([...tiled.data]).toEqual([...whole.data]);
  });

  it("crops a padded model output instead of drifting", () => {
    // Swin2SR's processor pads the input up to a multiple of 8 and upscales what
    // it was given, so a tile can come back larger than `scale x tile`. Reading
    // the top-left region keeps the real reconstruction; not cropping shifts
    // every subsequent tile and reads as a soft, doubled image.
    const src = ramp(300, 220);
    const padded = roundTrip(src, 2, (p) => {
      const clean = nearest(p, 2);
      const w = clean.width + 16;
      const h = clean.height + 16;
      const data = new Uint8ClampedArray(w * h * clean.channels);
      for (let y = 0; y < clean.height; y++) {
        for (let x = 0; x < clean.width; x++) {
          for (let k = 0; k < clean.channels; k++) {
            data[(y * w + x) * clean.channels + k] =
              clean.data[(y * clean.width + x) * clean.channels + k];
          }
        }
      }
      return { data, width: w, height: h, channels: clean.channels };
    });
    expect([...padded.data]).toEqual([...nearest(src, 2).data]);
  });

  it("never leaves a pixel with no colour at all", () => {
    // Weight zero at a seam is a division by zero, which paints a black line
    // exactly where the feather was meant to hide one.
    const src = ramp(500, 300);
    const plan = planTiles(500, 300);
    const canvas = createTileCanvas(500, 300, 3);
    for (const tile of plan.tiles) {
      blendTile(canvas, cropTile(src, tile), tile, {
        scale: 1,
        overlap: plan.overlap,
        source: src,
      });
    }
    expect([...canvas.weight].every((w) => w > 0)).toBe(true);
  });
});

describe("psnr", () => {
  it("is infinite for identical images", () => {
    const a = ramp(8, 8);
    expect(psnr(a, a)).toBe(Infinity);
  });

  it("falls as the difference grows", () => {
    const a = ramp(8, 8);
    const near = { ...a, data: a.data.map((v) => v + 2) as Uint8ClampedArray };
    const far = { ...a, data: a.data.map((v) => v + 40) as Uint8ClampedArray };
    expect(psnr(a, near)).toBeGreaterThan(psnr(a, far));
  });

  it("is 0 rather than NaN for an empty image", () => {
    const empty = { data: new Uint8ClampedArray(), width: 0, height: 0, channels: 3 };
    expect(psnr(empty, empty)).toBe(0);
  });
});
