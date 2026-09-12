import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { AudioSample } from "@/audio/samples";

// No Web Audio in the test env: decoding and capture are the boundary, and this
// hook's whole job is what it does with what they return.
const decodeToMono = vi.fn(async () => new Float32Array([0.1, 0.2, 0.3]));
const recordMic = vi.fn(async () => new Float32Array([0.4, 0.5]));
vi.mock("@/audio/io", () => ({
  decodeToMono: (...a: unknown[]) => decodeToMono(...(a as [])),
  recordMic: (...a: unknown[]) => recordMic(...(a as [])),
}));

const { useAudioPick } = await import("./useAudioPick");

const sample: AudioSample = {
  id: "jfk",
  label: "JFK",
  url: "https://example.test/jfk.wav",
  reference: "ask not what your country can do for you",
  hint: "~11 s",
};

const okFetch = () =>
  vi.fn().mockResolvedValue({
    ok: true,
    arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)),
  });

describe("useAudioPick", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    decodeToMono.mockResolvedValue(new Float32Array([0.1, 0.2, 0.3]));
    recordMic.mockResolvedValue(new Float32Array([0.4, 0.5]));
    vi.stubGlobal("fetch", okFetch());
  });

  it("starts empty — a page with no clip has nothing to run on", () => {
    const { result } = renderHook(() => useAudioPick());
    expect(result.current.clip).toBeNull();
    expect(result.current.take()).toBeNull();
    expect(result.current.preparing).toBeNull();
  });

  it("decodes a file and holds it, named", async () => {
    const { result } = renderHook(() => useAudioPick());
    const file = new File(["x"], "noisy.wav", { type: "audio/wav" });

    act(() => result.current.pickFile(file));
    await waitFor(() => expect(result.current.clip).not.toBeNull());

    expect(result.current.clip).toMatchObject({
      name: "noisy.wav",
      sampleRate: 16_000,
      sample: null,
    });
  });

  it("fetches a bundled clip and keeps the sample it came from", async () => {
    // The route renders the sample's reference transcript beside the output, so
    // the clip has to remember which sample produced it.
    const { result } = renderHook(() => useAudioPick());

    act(() => result.current.pickSample(sample));
    await waitFor(() => expect(result.current.clip).not.toBeNull());

    expect(fetch).toHaveBeenCalledWith(sample.url);
    expect(result.current.clip?.sample).toBe(sample);
    expect(result.current.clip?.name).toBe("JFK");
  });

  it("records from the mic into the input", async () => {
    const { result } = renderHook(() => useAudioPick());

    act(() => result.current.record(6));
    await waitFor(() => expect(result.current.clip).not.toBeNull());

    expect(recordMic).toHaveBeenCalledWith(6, 16_000);
    expect(result.current.clip?.name).toMatch(/recording/i);
  });

  // **The trap this hook exists to get right.** Every audio worker takes the
  // Float32Array's buffer as a transfer, which detaches it on this side. Hand
  // over the stored array and the waveform goes blank and the second GENERATE
  // has nothing to send.
  it("hands out a copy, never the held array", async () => {
    const { result } = renderHook(() => useAudioPick());
    act(() => result.current.pickFile(new File(["x"], "a.wav")));
    await waitFor(() => expect(result.current.clip).not.toBeNull());

    const held = result.current.clip!.audio;
    const first = result.current.take()!;
    const second = result.current.take()!;

    expect(first).not.toBe(held);
    expect(second).not.toBe(first);
    expect(Array.from(first)).toEqual(Array.from(held));
    // Two runs on one clip, which is the point: re-running after a model change
    // or a parameter edit must not need a second recording.
    expect(Array.from(second)).toEqual(Array.from(held));
  });

  // /audio-to-audio is the only 48 kHz route. A 16 kHz assumption leaking in
  // from ASR discards exactly the band DeepFilterNet3 is trained to repair.
  it("decodes and records at the rate the route asked for", async () => {
    const { result } = renderHook(() => useAudioPick({ sampleRate: 48_000 }));

    act(() => result.current.pickFile(new File(["x"], "a.wav")));
    await waitFor(() => expect(result.current.clip).not.toBeNull());
    expect(decodeToMono).toHaveBeenCalledWith(expect.anything(), 48_000);
    expect(result.current.clip?.sampleRate).toBe(48_000);

    act(() => result.current.record(6));
    await waitFor(() => expect(recordMic).toHaveBeenCalled());
    expect(recordMic).toHaveBeenCalledWith(6, 48_000);
  });

  it("reports a failed decode and holds nothing", async () => {
    decodeToMono.mockRejectedValueOnce(new Error("Unsupported audio format"));
    const { result } = renderHook(() => useAudioPick());

    act(() => result.current.pickFile(new File(["x"], "a.txt")));
    await waitFor(() =>
      expect(result.current.error).toMatch(/unsupported audio format/i),
    );
    expect(result.current.clip).toBeNull();
    expect(result.current.take()).toBeNull();
  });

  it("names the clip that failed to fetch, not just its status", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 404 }));
    const { result } = renderHook(() => useAudioPick());

    act(() => result.current.pickSample(sample));
    await waitFor(() => expect(result.current.error).toMatch(/JFK/));
    expect(result.current.error).toMatch(/404/);
  });

  it("keeps the previous clip when a file dialog is dismissed", async () => {
    const { result } = renderHook(() => useAudioPick());
    act(() => result.current.pickSample(sample));
    await waitFor(() => expect(result.current.clip).not.toBeNull());

    act(() => result.current.pickFile(undefined));
    expect(result.current.clip).not.toBeNull();
    expect(decodeToMono).toHaveBeenCalledTimes(1);
  });

  it("reports which source is preparing, then clears it", async () => {
    // The button that started the work is the one that shows a spinner.
    let release!: (v: Float32Array<ArrayBuffer>) => void;
    decodeToMono.mockReturnValueOnce(
      new Promise<Float32Array<ArrayBuffer>>((r) => {
        release = r;
      }),
    );
    const { result } = renderHook(() => useAudioPick());

    act(() => result.current.pickFile(new File(["x"], "a.wav")));
    await waitFor(() => expect(result.current.preparing).toBe("file"));

    await act(async () => {
      release(new Float32Array([0.9]));
    });
    await waitFor(() => expect(result.current.preparing).toBeNull());
  });

  it("replaces the held clip rather than accumulating", async () => {
    const { result } = renderHook(() => useAudioPick());
    act(() => result.current.pickSample(sample));
    await waitFor(() => expect(result.current.clip?.name).toBe("JFK"));

    decodeToMono.mockResolvedValueOnce(new Float32Array([0.7]));
    act(() => result.current.pickFile(new File(["x"], "mine.wav")));
    await waitFor(() => expect(result.current.clip?.name).toBe("mine.wav"));
    // The sample is forgotten with it, so no stale reference transcript is left
    // sitting beside an unrelated clip.
    expect(result.current.clip?.sample).toBeNull();
  });

  it("clears the error when a later pick succeeds", async () => {
    decodeToMono.mockRejectedValueOnce(new Error("bad"));
    const { result } = renderHook(() => useAudioPick());
    act(() => result.current.pickFile(new File(["x"], "a.wav")));
    await waitFor(() => expect(result.current.error).not.toBeNull());

    act(() => result.current.pickFile(new File(["x"], "b.wav")));
    await waitFor(() => expect(result.current.clip).not.toBeNull());
    expect(result.current.error).toBeNull();
  });

  // Picking is input. The vision twin lost its `onPicked` callback for exactly
  // this reason; this hook was born without one and must stay that way.
  it("takes no run callback — choosing a clip cannot start an inference", async () => {
    const onPicked = vi.fn();
    const { result } = renderHook(() =>
      (
        useAudioPick as (o?: unknown) => ReturnType<typeof useAudioPick>
      )({ onPicked }),
    );

    act(() => result.current.pickSample(sample));
    await waitFor(() => expect(result.current.clip).not.toBeNull());
    expect(onPicked).not.toHaveBeenCalled();
  });
});
