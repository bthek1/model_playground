import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

/** One RGBA pixel, half-covered — a soft matte, which is the whole point. */
const RESULT = {
  data: new Uint8ClampedArray([10, 20, 30, 128]),
  width: 1,
  height: 1,
  channels: 4,
};

const run = vi.fn().mockResolvedValue(RESULT);
const pipeline = {
  status: "ready",
  idle: false,
  loading: false,
  ready: true,
  progress: null,
  loadProgress: null,
  loadedInMs: 900,
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

const { useBackgroundRemoval } = await import("./useBackgroundRemoval");

const image = {} as never;

describe("useBackgroundRemoval", () => {
  afterEach(() => vi.clearAllMocks());

  it("selects the catalogue entry's task and defers loading", () => {
    renderHook(() => useBackgroundRemoval());
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "background-removal",
      "Xenova/modnet",
      false,
      undefined,
    );
  });

  it("defaults to the permissively licensed model, not the better one", () => {
    // RMBG-1.4 produces the nicer matte and is non-commercial only. A default
    // nobody downstream may legally use is a trap, so MODNet is first.
    const { result } = renderHook(() => useBackgroundRemoval());
    expect(result.current.meta.id).toBe("Xenova/modnet");
    expect(result.current.meta.licence.commercial).toBe(true);
  });

  it("exposes the selected entry's licence, so the route can state it", () => {
    const { result } = renderHook(() => useBackgroundRemoval("briaai/RMBG-1.4"));
    expect(result.current.meta.licence.commercial).toBe(false);
    expect(result.current.meta.licence.name).toBe("bria-rmbg-1.4");
  });

  it("falls back to the first catalogue entry for an unknown id", () => {
    renderHook(() => useBackgroundRemoval("someone/deleted-this-model"));
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "background-removal",
      "Xenova/modnet",
      false,
      undefined,
    );
  });

  it("keeps the RGBA result verbatim — the matte is its alpha channel", async () => {
    const { result } = renderHook(() => useBackgroundRemoval());
    await act(async () => {
      const out = await result.current.run(image);
      expect(out.data[3]).toBe(128);
    });
    expect(result.current.result?.data[3]).toBe(128);
  });
});
