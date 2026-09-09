import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

/** 2x2 masks: sky covers three pixels, tree one. */
const MASKS = [
  {
    label: "sky",
    score: 0.9,
    mask: { data: new Uint8ClampedArray([255, 255, 255, 0]), width: 2, height: 2 },
  },
  {
    label: "tree",
    score: 0.7,
    mask: { data: new Uint8ClampedArray([0, 0, 0, 255]), width: 2, height: 2 },
  },
];

const run = vi.fn().mockResolvedValue(MASKS);
const pipeline = {
  status: "ready",
  idle: false,
  loading: false,
  ready: true,
  progress: null,
  loadProgress: null,
  loadedInMs: 800,
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

const { useSegmenter, coverageOf, visibleMasks } = await import(
  "./useSegmenter"
);

const image = {} as never;

describe("useSegmenter", () => {
  afterEach(() => vi.clearAllMocks());

  it("selects the catalogue entry's task and defers loading", () => {
    renderHook(() => useSegmenter("Xenova/face-parsing"));
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "image-segmentation",
      "Xenova/face-parsing",
      false,
      undefined,
    );
  });

  it("pins fp32 for the repo that publishes nothing else", () => {
    // `mattmdjaga/segformer_b2_clothes` ships a single fp32 `model.onnx`, so
    // asking for fp16/q8 404s at load time. Same shape of exception as
    // MobileNetV4 on the classification page.
    renderHook(() => useSegmenter("mattmdjaga/segformer_b2_clothes"));
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "image-segmentation",
      "mattmdjaga/segformer_b2_clothes",
      false,
      { webgpu: "fp32", wasm: "fp32" },
    );
  });

  it("keeps every mask rather than a composited canvas", async () => {
    // Toggles and the opacity slider are derivations over these; a page that
    // stored the finished canvas would have to re-run the model to change either.
    const { result } = renderHook(() => useSegmenter());
    await act(async () => {
      await result.current.run(image);
    });
    expect(result.current.result).toHaveLength(2);
  });
});

describe("coverageOf", () => {
  it("measures each class as a fraction of the picture, largest first", () => {
    expect(coverageOf(MASKS)).toEqual([
      { label: "sky", coverage: 0.75 },
      { label: "tree", coverage: 0.25 },
    ]);
  });

  it("reads a 0-1 probability mask on its own scale", () => {
    const probs = [
      { label: "a", score: null, mask: { data: [0.9, 0.1], width: 2, height: 1 } },
    ];
    expect(coverageOf(probs)[0].coverage).toBe(0.5);
  });
});

describe("visibleMasks", () => {
  it("drops a hidden class entirely, so it contributes no pixels", () => {
    const shown = visibleMasks(MASKS, new Set(["sky"]));
    expect(shown.map((m) => m.label)).toEqual(["tree"]);
  });

  it("paints the smallest class last, so it survives the overlap", () => {
    // `drawMasks` composites in order and later wins, so the biggest class goes
    // down first. Reversed, a 2%-coverage class is buried under the sky every
    // time — and since the masks barely overlap, it would look almost right.
    expect(visibleMasks(MASKS, new Set()).map((m) => m.label)).toEqual([
      "sky",
      "tree",
    ]);
  });

  it("gives a class the same colour on every render of the same set", () => {
    const a = visibleMasks(MASKS, new Set());
    const b = visibleMasks(MASKS, new Set());
    expect(a.map((m) => m.label)).toEqual(b.map((m) => m.label));
  });

  it("survives being asked before the first run", () => {
    expect(visibleMasks(null, new Set())).toEqual([]);
  });
});
