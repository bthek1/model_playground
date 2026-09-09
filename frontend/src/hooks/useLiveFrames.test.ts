import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useCamera, useLiveFrames } from "./useLiveFrames";

/** Drives the rAF loop by hand so a test can pump exactly N frames. */
let ticks: FrameRequestCallback[] = [];

beforeEach(() => {
  ticks = [];
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    ticks.push(cb);
    return ticks.length;
  });
  vi.stubGlobal("cancelAnimationFrame", () => {});
  // A clock that always moves forward, so the hook's own frame-rate cap never
  // swallows a pumped frame just because the test ran inside one millisecond.
  let clock = 0;
  vi.stubGlobal("performance", {
    now: () => (clock += 25),
  });
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

const video = {} as HTMLVideoElement;

describe("useLiveFrames", () => {
  it("keeps exactly one frame in flight, however fast the loop runs", async () => {
    // The whole point of the hook: a naive rAF loop that posts every frame
    // builds an unbounded backlog and the overlay drifts seconds behind.
    let settle: (() => void) | undefined;
    const onFrame = vi.fn(
      () => new Promise<void>((resolve) => (settle = resolve)),
    );

    renderHook(() =>
      useLiveFrames({ video, active: true, onFrame, maxFps: 1000 }),
    );

    await pump(5);
    expect(onFrame).toHaveBeenCalledTimes(1);

    await act(async () => settle?.());
    await pump(1);
    expect(onFrame).toHaveBeenCalledTimes(2);
  });

  it("keeps running after a frame fails", async () => {
    const onFrame = vi
      .fn()
      .mockRejectedValueOnce(new Error("inference blew up"))
      .mockResolvedValue(undefined);

    renderHook(() =>
      useLiveFrames({ video, active: true, onFrame, maxFps: 1000 }),
    );

    await pump(1);
    await pump(1);
    expect(onFrame).toHaveBeenCalledTimes(2);
  });

  it("grabs nothing while inactive or without a video", async () => {
    const onFrame = vi.fn().mockResolvedValue(undefined);
    renderHook(() => useLiveFrames({ video, active: false, onFrame }));
    await pump(3);
    expect(onFrame).not.toHaveBeenCalled();

    renderHook(() => useLiveFrames({ video: null, active: true, onFrame }));
    await pump(3);
    expect(onFrame).not.toHaveBeenCalled();
  });

  it("does not restart the loop when the caller passes a new inline onFrame", async () => {
    const calls: number[] = [];
    const { rerender } = renderHook(
      ({ n }: { n: number }) =>
        useLiveFrames({
          video,
          active: true,
          maxFps: 1000,
          onFrame: async () => {
            calls.push(n);
          },
        }),
      { initialProps: { n: 1 } },
    );

    await pump(1);
    rerender({ n: 2 });
    await pump(1);

    // The second frame used the latest callback without the effect re-running —
    // restarting mid-frame is how the one-in-flight guard springs a leak.
    expect(calls).toEqual([1, 2]);
  });
});

describe("useCamera", () => {
  it("closes the camera on unmount", async () => {
    const close = vi.fn();
    const open = vi.fn().mockResolvedValue(close);
    const { unmount } = renderHook(() => useCamera(video, true, open));
    await act(async () => {});

    expect(open).toHaveBeenCalledOnce();
    unmount();
    expect(close).toHaveBeenCalledOnce();
  });

  it("closes a camera that opened after the component went away", async () => {
    let resolve: ((close: () => void) => void) | undefined;
    const open = vi.fn(() => new Promise<() => void>((r) => (resolve = r)));
    const close = vi.fn();

    const { unmount } = renderHook(() => useCamera(video, true, open));
    unmount();
    await act(async () => resolve?.(close));

    // Unmounted while the permission prompt was still up.
    expect(close).toHaveBeenCalledOnce();
  });

  it("reports a denied permission instead of throwing", async () => {
    const open = vi.fn().mockRejectedValue(new Error("Permission denied"));
    const { result } = renderHook(() => useCamera(video, true, open));
    await act(async () => {});
    expect(result.current.error).toBe("Permission denied");
  });
});
