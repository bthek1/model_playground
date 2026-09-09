import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { TOP_K } from "@/vision/classification";

const run = vi.fn().mockResolvedValue([
  { label: "tabby", score: 0.5 },
  { label: "tiger cat", score: 0.3 },
]);
const pipeline = {
  status: "ready",
  idle: false,
  loading: false,
  ready: true,
  progress: null,
  loadProgress: null,
  loadedInMs: 1200,
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

const { useImageClassifier } = await import("./useImageClassifier");

const image = {} as never;

describe("useImageClassifier", () => {
  afterEach(() => vi.clearAllMocks());

  it("selects the catalogue entry's task and defers loading", () => {
    renderHook(() => useImageClassifier("Xenova/resnet-50"));
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "image-classification",
      "Xenova/resnet-50",
      false,
      undefined,
    );
  });

  it("falls back to the first catalogue entry for an unknown id", () => {
    renderHook(() => useImageClassifier("someone/deleted-this-model"));
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "image-classification",
      "Xenova/vit-base-patch16-224",
      false,
      undefined,
    );
  });

  it("forwards a catalogue entry's precision override", () => {
    // MobileNetV4's q8 export labels a tiger "sidewinder" — it must load at
    // fp32 on WASM or not at all. Dropping this on the floor is a silent
    // accuracy regression, not a load failure, so it gets its own guard.
    renderHook(() =>
      useImageClassifier(
        "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k",
      ),
    );
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "image-classification",
      "onnx-community/mobilenetv4_conv_small.e2400_r224_in1k",
      false,
      { wasm: "fp32" },
    );
  });

  it("asks for the top five, not the argmax", async () => {
    // The near-tie is the interesting case, and a single label hides it.
    const { result } = renderHook(() => useImageClassifier());
    await act(async () => {
      await result.current.run(image);
    });
    expect(run).toHaveBeenCalledWith(image, [{ top_k: TOP_K }]);
    expect(TOP_K).toBe(5);
  });

  it("stores the ranked predictions and returns them", async () => {
    const { result } = renderHook(() => useImageClassifier());
    expect(result.current.result).toBeNull();

    let returned: unknown;
    await act(async () => {
      returned = await result.current.run(image);
    });

    expect(returned).toEqual([
      { label: "tabby", score: 0.5 },
      { label: "tiger cat", score: 0.3 },
    ]);
    await waitFor(() =>
      expect(result.current.result).toEqual([
        { label: "tabby", score: 0.5 },
        { label: "tiger cat", score: 0.3 },
      ]),
    );
  });

  it("passes the shared contract through unchanged", () => {
    const { result } = renderHook(() => useImageClassifier());
    // No renamed fields, no `classify()` alias — page-pattern §3.
    for (const key of [
      "status", "idle", "loading", "ready", "progress", "loadProgress",
      "loadedInMs", "backend", "load", "retry", "cancel", "run", "running",
      "result", "error",
    ]) {
      expect(result.current).toHaveProperty(key);
    }
    expect(result.current).not.toHaveProperty("classify");
  });
});
