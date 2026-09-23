import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { VideoSample } from "@/vision/video";

const sampleVideo = vi.fn();
vi.mock("@/vision/video", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, sampleVideo: (...a: unknown[]) => sampleVideo(...a) };
});

const { useVideoPick } = await import("./useVideoPick");

const SAMPLE: VideoSample = {
  id: "interview",
  label: "Interview",
  url: "https://example.test/interview.mp4",
  hint: "",
  labels: [],
};
const OTHER: VideoSample = { ...SAMPLE, id: "courtroom", label: "Courtroom", url: "https://example.test/courtroom.mp4" };

const frames = (n: number) =>
  Array.from({ length: n }, (_, i) => ({
    time: i + 0.5,
    image: { width: 8, height: 8 } as never,
  }));

beforeEach(() => {
  vi.clearAllMocks();
  sampleVideo.mockImplementation(async () => ({
    frames: frames(4),
    duration: 8,
    capped: false,
  }));
  URL.createObjectURL = vi.fn(() => "blob:clip");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => vi.clearAllMocks());

describe("useVideoPick", () => {
  it("holds a clip without decoding it", () => {
    const { result } = renderHook(() => useVideoPick());
    act(() => result.current.pickSample(SAMPLE));
    expect(result.current.picked?.url).toBe(SAMPLE.url);
    // Picking is not running, and here it is not even decoding: frame
    // extraction is a seek per frame.
    expect(sampleVideo).not.toHaveBeenCalled();
  });

  it("asks for the frame count it was given, spread over the clip", async () => {
    const { result } = renderHook(() => useVideoPick());
    act(() => result.current.pickSample(SAMPLE));
    await act(async () => {
      await result.current.take(4);
    });

    const [src, opts] = sampleVideo.mock.calls[0] as [
      string,
      { times: (d: number) => number[]; maxSide: number },
    ];
    expect(src).toBe(SAMPLE.url);
    expect(opts.maxSide).toBe(512);
    // The times are a function of the duration, which only the decoder knows.
    expect(opts.times(8)).toEqual([1, 3, 5, 7]);
  });

  it("decodes once for a repeated (clip, count) pair", async () => {
    const { result } = renderHook(() => useVideoPick());
    act(() => result.current.pickSample(SAMPLE));
    await act(async () => {
      await result.current.take(4);
      await result.current.take(4);
    });
    // Pressing GENERATE twice must cost one decode and two inferences.
    expect(sampleVideo).toHaveBeenCalledTimes(1);
  });

  it("decodes again when the frame count changes", async () => {
    const { result } = renderHook(() => useVideoPick());
    act(() => result.current.pickSample(SAMPLE));
    await act(async () => {
      await result.current.take(4);
      await result.current.take(8);
    });
    expect(sampleVideo).toHaveBeenCalledTimes(2);
  });

  it("decodes again when the clip changes, even at the same count", async () => {
    const { result } = renderHook(() => useVideoPick());
    act(() => result.current.pickSample(SAMPLE));
    await act(async () => {
      await result.current.take(4);
    });
    act(() => result.current.pickSample(OTHER));
    await act(async () => {
      await result.current.take(4);
    });
    expect(sampleVideo).toHaveBeenCalledTimes(2);
  });

  it("keeps exactly one object URL alive", async () => {
    const { result, unmount } = renderHook(() => useVideoPick());
    const file = new File([""], "a.mp4", { type: "video/mp4" });
    act(() => result.current.pickFile(file));
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);

    // A second upload revokes the first; a bundled sample revokes it too.
    act(() => result.current.pickFile(new File([""], "b.mp4")));
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(1);
    act(() => result.current.pickSample(SAMPLE));
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(2);

    // A sample URL is not ours, so unmount revokes nothing further.
    unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(2);
  });

  it("revokes its own URL on unmount", () => {
    const { result, unmount } = renderHook(() => useVideoPick());
    act(() => result.current.pickFile(new File([""], "a.mp4")));
    unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:clip");
  });

  it("reports a failed decode as an input error and returns nothing", async () => {
    sampleVideo.mockRejectedValueOnce(new Error("Could not decode that video"));
    const { result } = renderHook(() => useVideoPick());
    act(() => result.current.pickSample(SAMPLE));

    let got: unknown;
    await act(async () => {
      got = await result.current.take(4);
    });
    // Empty rather than thrown, so the caller bails without a try/catch — and
    // the reason lands in the RUN slot, not in OUTPUT.
    expect(got).toEqual([]);
    await waitFor(() =>
      expect(result.current.error).toMatch(/could not decode/i),
    );
  });

  it("does not cache a failed decode", async () => {
    sampleVideo.mockRejectedValueOnce(new Error("boom"));
    const { result } = renderHook(() => useVideoPick());
    act(() => result.current.pickSample(SAMPLE));
    await act(async () => {
      await result.current.take(4);
      await result.current.take(4);
    });
    expect(sampleVideo).toHaveBeenCalledTimes(2);
  });

  it("returns nothing when no clip is held", async () => {
    const { result } = renderHook(() => useVideoPick());
    let got: unknown;
    await act(async () => {
      got = await result.current.take(4);
    });
    expect(got).toEqual([]);
    expect(sampleVideo).not.toHaveBeenCalled();
  });
});
