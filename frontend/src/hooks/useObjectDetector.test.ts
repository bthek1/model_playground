import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MODEL_THRESHOLD } from "@/vision/detection";

const DETECTIONS = [
  { label: "person", score: 0.91, box: { xmin: 0, ymin: 0, xmax: 10, ymax: 10 } },
  { label: "car", score: 0.42, box: { xmin: 5, ymin: 5, xmax: 20, ymax: 20 } },
  { label: "kite", score: 0.08, box: { xmin: 1, ymin: 1, xmax: 3, ymax: 3 } },
];

const run = vi.fn().mockResolvedValue(DETECTIONS);
const pipeline = {
  status: "ready",
  idle: false,
  loading: false,
  ready: true,
  progress: null,
  loadProgress: null,
  loadedInMs: 900,
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

const { useObjectDetector, aboveThreshold } = await import(
  "./useObjectDetector"
);

const image = {} as never;

describe("useObjectDetector", () => {
  afterEach(() => vi.clearAllMocks());

  it("selects the catalogue entry's task and defers loading", () => {
    renderHook(() => useObjectDetector("onnx-community/dfine_s_coco-ONNX"));
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "object-detection",
      "onnx-community/dfine_s_coco-ONNX",
      false,
      undefined,
    );
  });

  it("falls back to the first catalogue entry for an unknown id", () => {
    renderHook(() => useObjectDetector("someone/deleted-this-model"));
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "object-detection",
      "onnx-community/dfine_n_coco-ONNX",
      false,
      undefined,
    );
  });

  it("asks the pipeline for absolute pixels, never fractions", async () => {
    // A regression test for a named bug, not box-ticking: the Transformers.js
    // default returns 0-1 fractions, `drawBoxes` wants pixels, and getting it
    // backwards piles every box into the top-left corner of the canvas.
    const { result } = renderHook(() => useObjectDetector());
    await act(async () => {
      await result.current.run(image);
    });

    expect(run).toHaveBeenCalledWith(
      image,
      [{ threshold: MODEL_THRESHOLD, percentage: false }],
      undefined,
    );
  });

  it("asks the model for everything plausible, so the slider can filter for free", () => {
    // The threshold sent to the model is well below the page's default: the
    // slider re-derives the visible set, and a high model-side floor would
    // throw away the rows it needs before they ever arrive.
    expect(MODEL_THRESHOLD).toBeLessThan(0.2);
  });

  it("stores every detection the model returned, unfiltered", async () => {
    const { result } = renderHook(() => useObjectDetector());
    await act(async () => {
      await result.current.run(image);
    });
    expect(result.current.result).toHaveLength(3);
  });
});

describe("aboveThreshold", () => {
  it("filters by score without touching the model", () => {
    expect(aboveThreshold(DETECTIONS, 0.4).map((d) => d.label)).toEqual([
      "person",
      "car",
    ]);
    expect(aboveThreshold(DETECTIONS, 0.95)).toEqual([]);
    expect(run).not.toHaveBeenCalled();
  });

  it("treats a score exactly on the threshold as included", () => {
    expect(aboveThreshold(DETECTIONS, 0.42)).toHaveLength(2);
  });

  it("survives being asked before the first run", () => {
    expect(aboveThreshold(null, 0.5)).toEqual([]);
  });
});
