import { describe, expect, it } from "vitest";

import {
  DEFAULT_SUPER_RES_MODEL,
  formatDuration,
  MAX_SOURCE_SIDE,
  MS_PER_TILE,
  SUPER_RES_MODELS,
} from "./superRes";
import { planTiles } from "./tile";

describe("SUPER_RES_MODELS", () => {
  it("pins fp32 on WASM, because q8 loses to bicubic", () => {
    // **Not a style preference — a measurement.** `just fe-e2e-superres` scores
    // the model against a ground truth it builds itself:
    //   fp32  27.80 dB   bicubic 27.55 dB
    //   q8    27.39 dB   bicubic 27.55 dB
    // At q8 the model is *worse than not running it*. Dense regression puts int8
    // error straight into the picture instead of letting an argmax absorb it.
    const swin = SUPER_RES_MODELS.find((m) => m.id === DEFAULT_SUPER_RES_MODEL)!;
    expect(swin.dtypes?.wasm).toBe("fp32");
  });

  it("quotes the fp32 download it actually pins, not the q8 estimate", () => {
    // A pinned precision makes the params estimate lie, so the entry owes
    // measured bytes — and the WASM figure must be the *fp32* file (54 MB), not
    // the quantized one it no longer downloads.
    const swin = SUPER_RES_MODELS.find((m) => m.id === DEFAULT_SUPER_RES_MODEL)!;
    expect(swin.bytes?.wasm).toBeGreaterThan(50_000_000);
    expect(swin.bytes?.webgpu).toBeGreaterThan(0);
  });

  it("declares the factor it upscales by, rather than assuming 2", () => {
    // `useSuperRes` sizes the output canvas from this and `blendTile` places
    // every patch by it; a hard-coded 2 would silently break an x4 entry.
    for (const model of SUPER_RES_MODELS) {
      expect(model.scale, model.id).toBeGreaterThan(1);
      expect(model.task, model.id).toBe("image-to-image");
    }
  });
});

describe("MAX_SOURCE_SIDE", () => {
  it("keeps a full-size source inside a handful of tiles", () => {
    // Tiles grow with *area*: at 1024 a square photo is ~35 tiles, at 512 it is
    // ~9. On WASM that is the difference between a demonstration and an
    // abandoned tab — the reason this moved from 1024 down to 512.
    const { tiles } = planTiles(MAX_SOURCE_SIDE, MAX_SOURCE_SIDE);
    expect(tiles.length).toBeLessThanOrEqual(9);
  });
});

describe("MS_PER_TILE", () => {
  it("quotes the WASM cost in tens of seconds, as measured", () => {
    // The first value here was 2,500 ms — out by more than a factor of ten,
    // which had the size guard promising 23 s for a run that took 273. Measured
    // at 32 s and 40 s per tile across two real runs.
    expect(MS_PER_TILE.wasm).toBeGreaterThanOrEqual(20_000);
    expect(MS_PER_TILE.wasm).toBeLessThanOrEqual(60_000);
  });

  it("expects a GPU to be faster than a CPU", () => {
    // The WebGPU figure is still an estimate, but this ordering is not
    // negotiable — if it ever inverts, one of the two is a typo.
    expect(MS_PER_TILE.webgpu).toBeLessThan(MS_PER_TILE.wasm);
  });

  it("covers every backend a load can resolve to", () => {
    // `pickBackend()` returns one of exactly these two, and the route falls back
    // to `wasm` for anything else — an absent key would quote NaN.
    expect(MS_PER_TILE.webgpu).toBeGreaterThan(0);
    expect(MS_PER_TILE.wasm).toBeGreaterThan(0);
  });
});

describe("formatDuration", () => {
  it("counts in seconds below a minute", () => {
    expect(formatDuration(3_000)).toBe("about 3 s");
    expect(formatDuration(30_000)).toBe("about 30 s");
  });

  it("switches to minutes once seconds stop being useful", () => {
    // "about 240 s" is a number a reader has to convert; "about 4 min" is one
    // they can act on.
    expect(formatDuration(240_000)).toBe("about 4 min");
    expect(formatDuration(90_000)).toBe("about 2 min");
  });

  it("never quotes zero", () => {
    // A guard that says "about 0 s" reads as "instant" and is a lie about a run
    // that is about to start.
    expect(formatDuration(0)).toBe("about 1 s");
    expect(formatDuration(10)).toBe("about 1 s");
  });
});
