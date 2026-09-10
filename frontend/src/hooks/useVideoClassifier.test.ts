import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const score = vi.fn();
const zeroShotState = {
  status: "ready" as const,
  idle: false,
  loading: false,
  ready: true,
  progress: null,
  loadProgress: null,
  loadedInMs: 900,
  backend: "webgpu",
  running: false,
  error: null,
  result: null,
  run: score,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};
const useZeroShotImage = vi.fn(() => zeroShotState);
vi.mock("@/hooks/useZeroShotImage", () => ({
  useZeroShotImage: (...a: unknown[]) => useZeroShotImage(...(a as [])),
}));

const frame = (time: number) => ({
  time,
  image: { width: 4, height: 4, channels: 3, data: [] },
});
const sampleVideo = vi.fn();
const thumbnail = vi.fn(() => "data:image/jpeg;base64,x");
vi.mock("@/vision/video", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    sampleVideo: (...a: unknown[]) => sampleVideo(...a),
    thumbnail: () => thumbnail(),
  };
});

const { useVideoClassifier } = await import("@/hooks/useVideoClassifier");

/** The shape `useZeroShotImage.run` returns: one entry per template. */
const scores = (values: number[], labels: string[]) => [
  {
    template: "a photo of a {}",
    scores: labels.map((label, i) => ({ label, score: values[i] })),
    textCached: true,
    textMs: 0,
    imageMs: 12,
  },
];

const LABELS = ["an interview", "a football match"];

beforeEach(() => {
  vi.clearAllMocks();
  sampleVideo.mockResolvedValue({
    frames: [frame(0.25), frame(0.75), frame(1.25)],
    duration: 1.5,
    capped: false,
  });
  score.mockImplementation(async () => scores([0.8, 0.2], LABELS));
});

describe("useVideoClassifier", () => {
  it("scores every frame, one request at a time", async () => {
    // Never a fan-out: two overlapping calls into one ONNX session is not a
    // guarantee worth relying on, and a fan-out makes the per-frame progress
    // meaningless.
    let inflight = 0;
    let peak = 0;
    score.mockImplementation(async () => {
      peak = Math.max(peak, ++inflight);
      await Promise.resolve();
      inflight--;
      return scores([0.8, 0.2], LABELS);
    });

    const { result } = renderHook(() => useVideoClassifier());
    await act(async () => {
      await result.current.run("blob:clip", LABELS, "a photo of a {}");
    });

    expect(score).toHaveBeenCalledTimes(3);
    expect(peak).toBe(1);
  });

  it("sends one template per frame, not the side-by-side comparison", async () => {
    // Doubling the per-frame cost for a comparison nobody can read across sixty
    // frames buys nothing — that experiment belongs to the still-image page.
    const { result } = renderHook(() => useVideoClassifier());
    await act(async () => {
      await result.current.run("blob:clip", LABELS, "a photo of a {}");
    });
    expect(score.mock.calls[0][2]).toEqual(["a photo of a {}"]);
  });

  it("keeps the raw per-frame scores, unpooled", async () => {
    // Pooling is a pure derivation, so the window slider re-derives the chart
    // without re-scoring the clip.
    score
      .mockImplementationOnce(async () => scores([0.9, 0.1], LABELS))
      .mockImplementationOnce(async () => scores([0.2, 0.8], LABELS))
      .mockImplementationOnce(async () => scores([0.7, 0.3], LABELS));

    const { result } = renderHook(() => useVideoClassifier());
    await act(async () => {
      await result.current.run("blob:clip", LABELS, "a photo of a {}");
    });

    await waitFor(() => expect(result.current.result).not.toBeNull());
    expect(result.current.result!.frames.map((f) => f.scores)).toEqual([
      [0.9, 0.1],
      [0.2, 0.8],
      [0.7, 0.3],
    ]);
    expect(result.current.result!.frames.map((f) => f.time)).toEqual([
      0.25, 0.75, 1.25,
    ]);
    expect(result.current.result!.labels).toEqual(LABELS);
  });

  it("reports sampling and scoring as separate phases", async () => {
    const seen: string[] = [];
    sampleVideo.mockImplementation(async (_src, opts) => {
      opts.onFrame?.(frame(0.25), 0, 2);
      return { frames: [frame(0.25), frame(0.75)], duration: 1, capped: false };
    });

    const { result } = renderHook(() => useVideoClassifier());
    let done!: Promise<unknown>;
    act(() => {
      done = result.current.run("blob:clip", LABELS, "{}");
    });
    await waitFor(() => {
      if (result.current.clipProgress) seen.push(result.current.clipProgress.phase);
      expect(seen.length).toBeGreaterThan(0);
    });
    await act(async () => {
      await done;
    });

    expect(result.current.clipProgress).toBeNull();
  });

  it("stops mid-clip, leaving no further requests", async () => {
    // The only page whose run is minutes long, so abandoning it has to be
    // possible — and has to leave nothing pending.
    const { result } = renderHook(() => useVideoClassifier());
    score.mockImplementation(async () => {
      act(() => result.current.stop());
      return scores([0.8, 0.2], LABELS);
    });

    await act(async () => {
      await result.current.run("blob:clip", LABELS, "{}");
    });

    // The first frame ran and set the flag; nothing after it did.
    expect(score).toHaveBeenCalledTimes(1);
    expect(result.current.clipProgress).toBeNull();
    expect(result.current.result!.frames).toHaveLength(1);
  });

  it("passes the cancel check down to the sampler too", async () => {
    // Cancelling during the *sampling* phase must stop seeking, not merely stop
    // scoring what was already decoded.
    const { result } = renderHook(() => useVideoClassifier());
    act(() => result.current.stop());
    await act(async () => {
      await result.current.run("blob:clip", LABELS, "{}");
    });
    // `run` resets the flag, so the sampler is handed a live predicate rather
    // than a stale `true`.
    expect(sampleVideo.mock.calls[0][1].cancelled()).toBe(false);
  });

  it("refuses to score with no labels", async () => {
    const { result } = renderHook(() => useVideoClassifier());
    await expect(
      result.current.run("blob:clip", ["  "], "{}"),
    ).rejects.toThrow(/at least one label/i);
    expect(sampleVideo).not.toHaveBeenCalled();
  });

  it("carries the frame cap through to the sampler", async () => {
    const { result } = renderHook(() => useVideoClassifier());
    await act(async () => {
      await result.current.run("blob:clip", LABELS, "{}", {
        fps: 4,
        maxFrames: 30,
      });
    });
    expect(sampleVideo.mock.calls[0][1]).toMatchObject({ fps: 4, maxFrames: 30 });
  });

  it("clears the progress even when sampling throws", async () => {
    sampleVideo.mockRejectedValueOnce(new Error("Could not decode that video"));
    const { result } = renderHook(() => useVideoClassifier());
    await act(async () => {
      await expect(
        result.current.run("blob:clip", LABELS, "{}"),
      ).rejects.toThrow(/could not decode/i);
    });
    expect(result.current.clipProgress).toBeNull();
    expect(result.current.running).toBe(false);
  });
});
