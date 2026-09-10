import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Detection } from "@/vision/draw";

const post = vi.fn();
const pipeState = {
  status: "ready" as const,
  idle: false,
  loading: false,
  ready: true,
  progress: null,
  loadProgress: null,
  loadedInMs: 120,
  backend: "webgpu",
  running: false,
  error: null,
  run: post,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};
const useVisionPipeline = vi.fn(() => pipeState);
vi.mock("@/hooks/useVisionPipeline", () => ({
  useVisionPipeline: (...a: unknown[]) => useVisionPipeline(...(a as [])),
}));

const { groupByQuery, useZeroShotDetector } = await import(
  "@/hooks/useZeroShotDetector"
);
const { MODEL_THRESHOLD } = await import("@/vision/zeroShotDetection");

const image = { width: 4, height: 4, channels: 3, data: [] } as never;
const DETECTIONS: Detection[] = [
  { label: "a person", score: 0.4, box: { xmin: 0, ymin: 0, xmax: 2, ymax: 4 } },
  { label: "a person", score: 0.2, box: { xmin: 2, ymin: 0, xmax: 4, ymax: 4 } },
  { label: "a car", score: 0.1, box: { xmin: 0, ymin: 2, xmax: 1, ymax: 3 } },
];

beforeEach(() => {
  vi.clearAllMocks();
  post.mockResolvedValue(DETECTIONS);
});

describe("useZeroShotDetector", () => {
  it("loads the default model, and never on mount", () => {
    renderHook(() => useZeroShotDetector());
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "zero-shot-object-detection",
      "Xenova/owlv2-base-patch16-ensemble",
      false,
      undefined,
    );
  });

  it("pins percentage: false and the low model threshold", async () => {
    // `drawBoxes` wants absolute pixels; 0–1 fractions collapse every box into
    // the top-left corner. The model is asked at a floor far below anything the
    // page shows, so the page's slider re-filters instead of re-running.
    const { result } = renderHook(() => useZeroShotDetector());
    await act(async () => {
      await result.current.run(image, ["a person"]);
    });

    expect(post).toHaveBeenCalledWith(
      image,
      [["a person"], { threshold: MODEL_THRESHOLD, percentage: false }],
      undefined,
    );
    expect(MODEL_THRESHOLD).toBeLessThan(0.05);
  });

  it("sends the queries verbatim, trimmed, with the blanks dropped", async () => {
    // No prompt template on this route: what is on screen is what the text
    // tower sees.
    const { result } = renderHook(() => useZeroShotDetector());
    await act(async () => {
      await result.current.run(image, ["  a person ", "", "a red umbrella"]);
    });
    expect(post.mock.calls[0][1][0]).toEqual(["a person", "a red umbrella"]);
  });

  it("refuses to run with no queries rather than asking for nothing", async () => {
    const { result } = renderHook(() => useZeroShotDetector());
    await expect(result.current.run(image, ["  "])).rejects.toThrow(
      /at least one query/i,
    );
    expect(post).not.toHaveBeenCalled();
  });

  it("keeps the last detections as `result`", async () => {
    const { result } = renderHook(() => useZeroShotDetector());
    await act(async () => {
      await result.current.run(image, ["a person"]);
    });
    await waitFor(() => expect(result.current.result).toEqual(DETECTIONS));
  });
});

describe("groupByQuery", () => {
  it("keeps the user's query order, and the queries that found nothing", () => {
    // The misses are the point: "which of my phrases matched nothing" is the
    // question the page exists to answer.
    const groups = groupByQuery(DETECTIONS, [
      "a car",
      "a person",
      "a purple giraffe",
    ]);
    expect(groups.map((g) => g.query)).toEqual([
      "a car",
      "a person",
      "a purple giraffe",
    ]);
    expect(groups[1].detections).toHaveLength(2);
    expect(groups[2].detections).toEqual([]);
  });

  it("surfaces a label we did not ask for under its own heading", () => {
    // Grounding DINO answers with a fragment of the query rather than the query
    // itself; dropping those would hide real detections.
    const groups = groupByQuery(
      [
        {
          label: "umbrella",
          score: 0.3,
          box: { xmin: 0, ymin: 0, xmax: 1, ymax: 1 },
        },
      ],
      ["a red umbrella"],
    );
    expect(groups.map((g) => g.query)).toEqual(["a red umbrella", "umbrella"]);
    expect(groups[0].detections).toEqual([]);
    expect(groups[1].detections).toHaveLength(1);
  });

  it("returns a group per query even with no detections at all", () => {
    expect(groupByQuery([], ["a cat"])).toEqual([
      { query: "a cat", detections: [] },
    ]);
  });
});
