import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const stop = vi.fn();
const openCamera = vi.fn().mockResolvedValue(stop);
const fromVideo = vi.fn((el: { videoWidth: number }) => ({
  width: el.videoWidth,
  height: 480,
  channels: 3,
}));
const downscale = vi.fn(async (img: unknown) => img);

vi.mock("@/vision/image", () => ({
  openCamera: (...a: unknown[]) => openCamera(...a),
  fromVideo: (...a: unknown[]) => fromVideo(...(a as [never])),
  downscale: (...a: unknown[]) => downscale(...(a as [never])),
}));

const { useCameraFrames } = await import("./useCameraFrames");

/** Drives the rAF loop by hand, as `useLiveFrames.test.ts` does. */
let ticks: FrameRequestCallback[] = [];

beforeEach(() => {
  vi.clearAllMocks();
  openCamera.mockResolvedValue(stop);
  ticks = [];
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    ticks.push(cb);
    return ticks.length;
  });
  vi.stubGlobal("cancelAnimationFrame", () => {});
  let clock = 0;
  vi.stubGlobal("performance", { now: () => (clock += 25) });
});
afterEach(() => vi.unstubAllGlobals());

async function pump(n: number) {
  for (let i = 0; i < n; i++) {
    const next = ticks.shift();
    if (!next) return;
    await act(async () => {
      next(performance.now());
    });
  }
}

function playing() {
  return { videoWidth: 640, videoHeight: 480 } as HTMLVideoElement;
}

describe("useCameraFrames", () => {
  it("opens nothing until the page asks for the camera", () => {
    const { result } = renderHook(() =>
      useCameraFrames({ active: false, maxSide: 640, onFrame: vi.fn() }),
    );
    act(() => result.current.videoRef(playing()));
    expect(openCamera).not.toHaveBeenCalled();
  });

  it("stops every track when the page unmounts", async () => {
    // A leaked MediaStream leaves the webcam light on after the user has
    // navigated away — the one bug here a user notices from across the room.
    const { result, unmount } = renderHook(() =>
      useCameraFrames({ active: true, maxSide: 640, onFrame: vi.fn() }),
    );
    await act(async () => {
      result.current.videoRef(playing());
    });
    expect(openCamera).toHaveBeenCalled();

    unmount();
    expect(stop).toHaveBeenCalled();
  });

  it("downscales the frame before it reaches the model", async () => {
    // Resolution is the throttle: a detector at 1280x720 costs roughly 4x the
    // same detector at 640x480.
    const onFrame = vi.fn().mockResolvedValue(undefined);
    const { result } = renderHook(() =>
      useCameraFrames({ active: true, maxSide: 512, onFrame }),
    );
    await act(async () => {
      result.current.videoRef(playing());
    });
    await pump(1);

    expect(downscale).toHaveBeenCalledWith(expect.anything(), 512);
    expect(onFrame).toHaveBeenCalledTimes(1);
  });

  it("skips a frame the stream has not sized yet", async () => {
    // The first animation frames land before the stream has dimensions;
    // capturing then yields a 0x0 canvas and one pipeline error per frame.
    const onFrame = vi.fn().mockResolvedValue(undefined);
    const { result } = renderHook(() =>
      useCameraFrames({ active: true, maxSide: 640, onFrame }),
    );
    await act(async () => {
      result.current.videoRef({
        videoWidth: 0,
        videoHeight: 0,
      } as HTMLVideoElement);
    });
    await pump(3);

    expect(fromVideo).not.toHaveBeenCalled();
    expect(onFrame).not.toHaveBeenCalled();
  });

  it("never queues: one frame in flight however many ticks fire", async () => {
    let release: (() => void) | null = null;
    const onFrame = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          release = resolve;
        }),
    );
    const { result } = renderHook(() =>
      useCameraFrames({ active: true, maxSide: 640, onFrame }),
    );
    await act(async () => {
      result.current.videoRef(playing());
    });

    await pump(5);
    expect(onFrame).toHaveBeenCalledTimes(1);

    await act(async () => {
      release?.();
    });
    await pump(1);
    expect(onFrame).toHaveBeenCalledTimes(2);
  });

  it("reports a denied permission instead of throwing", async () => {
    openCamera.mockRejectedValueOnce(new Error("Permission denied"));
    const { result } = renderHook(() =>
      useCameraFrames({ active: true, maxSide: 640, onFrame: vi.fn() }),
    );
    await act(async () => {
      result.current.videoRef(playing());
    });
    expect(result.current.error).toMatch(/permission denied/i);
  });
});
