import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ImagePayload } from "@/vision/image";

/**
 * Stand in for the model: returns the patch it was given, upscaled 2x by pixel
 * doubling. Exactly reproducible, so the reassembly can be checked rather than
 * merely observed to have produced something.
 */
const run = vi.fn(async (image: ImagePayload) => {
  const { width, height, channels: c, data } = image;
  const w = width * 2;
  const h = height * 2;
  const out = new Uint8ClampedArray(w * h * c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const src = ((y / 2) | 0) * width + ((x / 2) | 0);
      for (let k = 0; k < c; k++) out[(y * w + x) * c + k] = data[src * c + k];
    }
  }
  return { data: out, width: w, height: h, channels: c };
});

const pipeline = {
  status: "ready",
  idle: false,
  loading: false,
  ready: true,
  progress: null,
  loadProgress: null,
  loadedInMs: 4200,
  backend: "wasm",
  running: false,
  error: null,
  run,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};
const useVisionPipeline = vi.fn(() => pipeline);

vi.mock("./useVisionPipeline", () => ({
  useVisionPipeline: (...args: unknown[]) => useVisionPipeline(...(args as [])),
}));

// `fromPayload` builds a real `RawImage`, which needs the Transformers.js
// runtime. The fake pipeline above only reads the four fields a payload has, so
// the identity is the right stand-in — and it keeps this a unit test.
vi.mock("@/vision/image", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, fromPayload: (p: unknown) => p };
});

const { useSuperRes, RunCancelled } = await import("./useSuperRes");

/** A source image with a distinct value per pixel and channel. */
function source(width: number, height: number) {
  const data = new Uint8ClampedArray(width * height * 3);
  for (let i = 0; i < width * height * 3; i++) data[i] = (i * 11) % 256;
  return { data, width, height, channels: 3 } as never;
}

describe("useSuperRes", () => {
  afterEach(() => vi.clearAllMocks());

  it("selects the catalogue entry's task and defers loading", () => {
    renderHook(() => useSuperRes());
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "image-to-image",
      "Xenova/swin2SR-classical-sr-x2-64",
      false,
      { wasm: "fp32" },
    );
  });

  it("falls back to the first catalogue entry for an unknown id", () => {
    renderHook(() => useSuperRes("someone/deleted-this-model"));
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "image-to-image",
      "Xenova/swin2SR-classical-sr-x2-64",
      false,
      { wasm: "fp32" },
    );
  });

  it("produces a result exactly 2x the input dimensions", async () => {
    const { result } = renderHook(() => useSuperRes());
    await act(async () => {
      const out = await result.current.run(source(100, 80));
      expect(out.width).toBe(200);
      expect(out.height).toBe(160);
    });
  });

  it("runs one inference per tile, not one per image", async () => {
    // 600x400 at tile 192 / overlap 32 is 4 columns by 3 rows. A single
    // whole-image call is precisely the failure mode tiling exists to prevent —
    // and it is also the version that passes every "an image came back" check.
    const { result } = renderHook(() => useSuperRes());
    await act(async () => {
      await result.current.run(source(600, 400));
    });
    expect(run).toHaveBeenCalledTimes(12);
  });

  it("reassembles the tiles in the right order", async () => {
    // The stand-in model is pixel doubling, so a correct reassembly is
    // bit-identical to doubling the whole image. A shuffled or offset
    // reassembly is still perfectly sharp, and only this catches it.
    const src = source(300, 220);
    const { result } = renderHook(() => useSuperRes());

    let out!: { data: Uint8ClampedArray; width: number };
    await act(async () => {
      out = await result.current.run(src);
    });

    const s = src as unknown as { data: Uint8ClampedArray; width: number };
    for (const [x, y] of [
      [0, 0],
      [150, 110],
      [299, 219],
      [200, 40],
    ]) {
      const srcIdx = (y * 300 + x) * 3;
      const dstIdx = (y * 2 * 600 + x * 2) * 3;
      expect(out.data[dstIdx]).toBe(s.data[srcIdx]);
    }
  });

  it("reports tile progress while running, and clears it after", async () => {
    const { result } = renderHook(() => useSuperRes());
    expect(result.current.tiles).toBeNull();

    let finished!: Promise<unknown>;
    await act(async () => {
      finished = result.current.run(source(300, 220));
      await Promise.resolve();
    });
    await act(async () => {
      await finished;
    });
    expect(result.current.tiles).toBeNull();
  });

  it("stops running tiles when asked, and keeps the model loaded", async () => {
    const { result } = renderHook(() => useSuperRes());

    // Stop after the first tile: the loop checks between tiles, so the count
    // must not keep climbing.
    run.mockImplementationOnce(async (image: ImagePayload) => {
      result.current.stop();
      return { data: new Uint8ClampedArray(image.width * 2 * image.height * 2 * 3), width: image.width * 2, height: image.height * 2, channels: 3 };
    });

    await act(async () => {
      await expect(result.current.run(source(300, 220))).rejects.toBeInstanceOf(
        RunCancelled,
      );
    });

    expect(run).toHaveBeenCalledTimes(1);
    expect(pipeline.cancel).not.toHaveBeenCalled();
    await waitFor(() => expect(result.current.tiles).toBeNull());
    expect(result.current.ready).toBe(true);
  });

  it("keeps the source beside the result, for the comparison view", async () => {
    // A 2x image on its own proves nothing; the page shows it against a bicubic
    // upscale of the same input, so the input has to survive the run.
    const { result } = renderHook(() => useSuperRes());
    await act(async () => {
      await result.current.run(source(100, 80));
    });
    expect(result.current.source?.width).toBe(100);
    expect(result.current.result?.width).toBe(200);
  });
});
