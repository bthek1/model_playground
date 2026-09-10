import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { PoseResult } from "@/vision/pose/types";

const post = vi.fn();
const workerState = {
  status: "ready" as const,
  idle: false,
  loading: false,
  ready: true,
  progress: null,
  loadProgress: null,
  loadedInMs: 4200,
  backend: "webgpu",
  running: false,
  result: null,
  error: null,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
  run: post,
};
const useModelWorker = vi.fn(() => workerState);
vi.mock("@/model/useModelWorker", () => ({
  useModelWorker: (...a: unknown[]) => useModelWorker(...(a as [])),
}));
vi.mock("@/vision/pose/client", () => ({ createPoseWorker: vi.fn() }));

const { usePose } = await import("@/hooks/usePose");

// The buffer is named because the transfer test compares identity against it,
// and `image` itself is cast `as never` (the codebase's idiom for a stand-in
// `RawImage`), which leaves `image.data` with no type to dereference.
const pixels = new Uint8ClampedArray(64 * 48 * 3);
const image = {
  width: 64,
  height: 48,
  channels: 3,
  data: pixels,
} as never;

const REPLY: PoseResult = {
  people: [
    {
      box: { xmin: 0, ymin: 0, xmax: 20, ymax: 40 },
      score: 0.9,
      keypoints: [{ x: 10, y: 5, score: 0.8, index: 0 }],
    },
  ],
  detected: 3,
  detectMs: 18,
  poseMs: 120,
};

beforeEach(() => {
  vi.clearAllMocks();
  post.mockResolvedValue(REPLY);
});

describe("usePose", () => {
  it("keys the worker on the pair, and never loads on mount", () => {
    // The composite id: the user picks a *pair*, and switching either half
    // means a different worker.
    renderHook(() => usePose());
    expect(useModelWorker).toHaveBeenCalledWith(
      expect.objectContaining({
        key: "pose:dfine-n+vitpose-base",
        autoLoad: false,
        loadMessage: { model: "dfine-n+vitpose-base" },
      }),
    );
  });

  it("falls back to the default pair for an id we no longer ship", () => {
    renderHook(() => usePose("gone"));
    expect(useModelWorker).toHaveBeenCalledWith(
      expect.objectContaining({ key: "pose:dfine-n+vitpose-base" }),
    );
  });

  it("sends the threshold and the cap to the worker, not to a filter", async () => {
    // The opposite of /object-detection's slider, and deliberately: filtering
    // afterwards would mean running the pose model on people the user has
    // already excluded, and that pass is the expensive half.
    const { result } = renderHook(() => usePose());
    await act(async () => {
      await result.current.run(image, { threshold: 0.6, maxPeople: 3 });
    });

    const [payload] = post.mock.calls[0];
    expect(payload).toMatchObject({ threshold: 0.6, maxPeople: 3 });
    expect(payload.image).toMatchObject({ width: 64, height: 48, channels: 3 });
  });

  it("copies the buffer by default, and transfers a spent frame", async () => {
    // A still the page is still showing must not be detached; a webcam frame
    // the page has finished with should be.
    const { result } = renderHook(() => usePose());
    await act(async () => {
      await result.current.run(image, { threshold: 0.4, maxPeople: 5 });
    });
    expect(post.mock.calls[0][0].image.data).not.toBe(pixels);

    await act(async () => {
      await result.current.run(image, {
        threshold: 0.4,
        maxPeople: 5,
        consume: true,
      });
    });
    expect(post.mock.calls[1][0].image.data).toBe(pixels);
    // Either way there is a transfer list — the copy is the one being posted.
    expect(post.mock.calls[1][1]).toHaveLength(1);
  });

  it("keeps the last result, with both halves' timings", async () => {
    // This page's whole shape is two models, so it says what each one cost.
    const { result } = renderHook(() => usePose());
    await act(async () => {
      await result.current.run(image, { threshold: 0.4, maxPeople: 5 });
    });
    await waitFor(() => expect(result.current.result).toEqual(REPLY));
    expect(result.current.result!.detectMs).toBe(18);
    expect(result.current.result!.poseMs).toBe(120);
  });

  it("exposes the shared contract, and nothing task-shaped beyond it", () => {
    // Two models is entirely the worker's business: the LOAD state covers both
    // downloads with one aggregate bar, and the hook contract is unchanged.
    const { result } = renderHook(() => usePose());
    for (const key of [
      "status",
      "idle",
      "loading",
      "ready",
      "progress",
      "loadProgress",
      "loadedInMs",
      "backend",
      "running",
      "error",
      "result",
      "run",
      "load",
      "retry",
      "cancel",
    ]) {
      expect(result.current, key).toHaveProperty(key);
    }
  });
});
