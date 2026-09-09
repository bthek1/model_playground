import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const RESULT = {
  predicted_depth: { data: new Float32Array([1, 2, 3, 4]), dims: [1, 2, 2] },
};

const run = vi.fn().mockResolvedValue(RESULT);
const pipeline = {
  status: "ready",
  idle: false,
  loading: false,
  ready: true,
  progress: null,
  loadProgress: null,
  loadedInMs: 1500,
  backend: "webgpu",
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

const { useDepth, depthDims } = await import("./useDepth");

const image = {} as never;

describe("useDepth", () => {
  afterEach(() => vi.clearAllMocks());

  it("selects the catalogue entry's task and defers loading", () => {
    renderHook(() => useDepth("Xenova/depth-anything-small-hf"));
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "depth-estimation",
      "Xenova/depth-anything-small-hf",
      false,
      undefined,
    );
  });

  it("passes the catalogue's per-backend precision override through", () => {
    // Depth Pro is pinned to q8 because its fp16 export is 1.8 GB.
    renderHook(() => useDepth("onnx-community/DepthPro-ONNX"));
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "depth-estimation",
      "onnx-community/DepthPro-ONNX",
      false,
      { webgpu: "q8", wasm: "q8" },
    );
  });

  it("falls back to the first catalogue entry for an unknown id", () => {
    renderHook(() => useDepth("someone/deleted-this-model"));
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "depth-estimation",
      "onnx-community/depth-anything-v2-small",
      false,
      undefined,
    );
  });

  it("keeps the raw tensor, which is the only thing with a real range", () => {
    // The normalised `depth` image has already thrown the scale away, so a page
    // that wants a legend — or a test that wants an assertion — needs this one.
    const { result } = renderHook(() => useDepth());
    return act(async () => {
      const out = await result.current.run(image);
      expect(out.predicted_depth.data).toEqual(RESULT.predicted_depth.data);
    });
  });
});

describe("depthDims", () => {
  it("reads height and width off the last two dims, batch or not", () => {
    // `[1, h, w]` and `[h, w]` both occur. Reading them the wrong way round
    // transposes the map into diagonal streaks rather than failing.
    expect(depthDims({ data: [], dims: [1, 480, 640] })).toEqual({
      width: 640,
      height: 480,
    });
    expect(depthDims({ data: [], dims: [480, 640] })).toEqual({
      width: 640,
      height: 480,
    });
  });

  it("returns zeros rather than NaN for a shape it cannot read", () => {
    expect(depthDims({ data: [], dims: [] })).toEqual({ width: 0, height: 0 });
  });
});
