import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AsrResult } from "@/audio/types";

// --- Mock the underlying ASR hook so we test only the live-capture logic. -----
const transcribe = vi.fn<(a: Float32Array) => Promise<AsrResult>>();
const asrState = {
  status: "ready" as const,
  loading: false,
  ready: true,
  progress: null,
  backend: "wasm" as string | null,
  result: null,
  idle: false,
  running: false,
  error: null as string | null,
  transcribe,
  load: vi.fn(),
  retry: vi.fn(),
};
// Records its arguments: `useLiveAsr` owns the capture loop but delegates the
// whole model half, including the deferred-load flag the ASR route depends on.
const useAsr = vi.fn((model: string, autoLoad?: boolean) => {
  void model;
  void autoLoad;
  return asrState;
});
vi.mock("@/hooks/useAsr", () => ({
  useAsr: (model: string, autoLoad?: boolean) => useAsr(model, autoLoad),
}));

// --- Fakes for the Web Audio / MediaRecorder globals. -------------------------
class FakeAudioBuffer {
  constructor(
    public duration: number,
    private data: Float32Array,
  ) {}
  getChannelData() {
    return this.data;
  }
}

/** Stub the decode path so decodeToMono returns `rendered`. */
function installDecodeMocks(rendered: Float32Array) {
  class FakeAudioContext {
    decodeAudioData = vi.fn().mockResolvedValue(new FakeAudioBuffer(1, new Float32Array(0)));
    close = vi.fn().mockResolvedValue(undefined);
  }
  vi.stubGlobal("AudioContext", FakeAudioContext);
  class FakeOfflineAudioContext {
    destination = {};
    createBufferSource() {
      return { buffer: null, connect: vi.fn(), start: vi.fn() };
    }
    startRendering() {
      return Promise.resolve(new FakeAudioBuffer(1, rendered));
    }
  }
  vi.stubGlobal("OfflineAudioContext", FakeOfflineAudioContext);
}

let lastRecorder: FakeMediaRecorder;

class FakeMediaRecorder {
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  state: "recording" | "inactive" = "inactive";
  start = vi.fn(() => {
    this.state = "recording";
  });
  stop = vi.fn(() => {
    this.state = "inactive";
    this.onstop?.();
  });
  /** Test helper: emit an audio chunk as the browser would per timeslice. */
  emitChunk() {
    this.ondataavailable?.({ data: new Blob([new Uint8Array([1, 2, 3])]) });
  }
}

const track = { stop: vi.fn() };
const getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => [track] });

function installMediaMocks() {
  // Wrap construction so the test can grab the instance the hook creates without
  // aliasing `this` inside the fake.
  vi.stubGlobal("MediaRecorder", function MediaRecorderMock() {
    lastRecorder = new FakeMediaRecorder();
    return lastRecorder;
  });
  Object.defineProperty(navigator, "mediaDevices", {
    value: { getUserMedia },
    configurable: true,
    writable: true,
  });
}

/** Let queued microtasks (decode → transcribe → setState) settle. */
const flush = () => act(async () => void (await new Promise((r) => setTimeout(r, 0))));

describe("useLiveAsr", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  // The model half is delegated wholesale — the capture loop is this hook's own
  // work, the load lifecycle is not. These pin the seam.
  it("forwards the model and autoLoad to useAsr", async () => {
    const { useLiveAsr } = await import("./useLiveAsr");

    renderHook(() => useLiveAsr("onnx-community/whisper-base", false));
    expect(useAsr).toHaveBeenCalledWith("onnx-community/whisper-base", false);
  });

  it("auto-loads by default, as the non-route callers expect", async () => {
    const { useLiveAsr } = await import("./useLiveAsr");

    renderHook(() => useLiveAsr("onnx-community/whisper-base"));
    expect(useAsr).toHaveBeenCalledWith("onnx-community/whisper-base", true);
  });

  it("re-exposes status, idle and the load actions for the LOAD slot", async () => {
    const { useLiveAsr } = await import("./useLiveAsr");
    const { result } = renderHook(() => useLiveAsr());

    expect(result.current.status).toBe("ready");
    expect(result.current.idle).toBe(false);

    result.current.load();
    result.current.retry();
    expect(asrState.load).toHaveBeenCalledOnce();
    expect(asrState.retry).toHaveBeenCalledOnce();
  });

  it("merges its own capture error over the model's", async () => {
    const { useLiveAsr } = await import("./useLiveAsr");
    asrState.error = "model failed";
    try {
      const { result } = renderHook(() => useLiveAsr());
      expect(result.current.error).toBe("model failed");
    } finally {
      asrState.error = null;
    }
  });

  it("starts capture, transcribes each chunk, and updates the transcript live", async () => {
    installDecodeMocks(new Float32Array([0.1, 0.2]));
    installMediaMocks();
    transcribe.mockResolvedValue({ text: "hello world" });

    const { useLiveAsr } = await import("./useLiveAsr");
    const { result } = renderHook(() => useLiveAsr());

    await act(async () => {
      await result.current.start();
    });
    expect(getUserMedia).toHaveBeenCalledWith({ audio: true });
    expect(result.current.recording).toBe(true);
    expect(lastRecorder.start).toHaveBeenCalledWith(1500);

    // A chunk arrives → decode + transcribe → text appears.
    await act(async () => lastRecorder.emitChunk());
    await flush();

    expect(transcribe).toHaveBeenCalledTimes(1);
    expect(transcribe.mock.calls[0][0]).toBeInstanceOf(Float32Array);
    await waitFor(() => expect(result.current.text).toBe("hello world"));
  });

  it("skips overlapping ticks while one transcription is in flight", async () => {
    installDecodeMocks(new Float32Array([0.1]));
    installMediaMocks();
    // A transcription that never resolves keeps the busy flag set.
    transcribe.mockReturnValue(new Promise<AsrResult>(() => {}));

    const { useLiveAsr } = await import("./useLiveAsr");
    const { result } = renderHook(() => useLiveAsr());
    await act(async () => {
      await result.current.start();
    });

    await act(async () => lastRecorder.emitChunk());
    await flush();
    await act(async () => lastRecorder.emitChunk()); // second tick while busy
    await flush();

    expect(transcribe).toHaveBeenCalledTimes(1);
  });

  it("stops capture, runs a final pass, retains the take, and releases the mic", async () => {
    const rendered = new Float32Array([0.3]);
    installDecodeMocks(rendered);
    installMediaMocks();
    transcribe.mockResolvedValue({ text: "final" });

    const { useLiveAsr } = await import("./useLiveAsr");
    const { result } = renderHook(() => useLiveAsr());
    await act(async () => {
      await result.current.start();
    });
    expect(result.current.stream).not.toBeNull(); // exposed for the live waveform
    await act(async () => lastRecorder.emitChunk());
    await flush();

    act(() => result.current.stop());
    await flush();

    expect(lastRecorder.stop).toHaveBeenCalled();
    expect(track.stop).toHaveBeenCalled();
    expect(result.current.recording).toBe(false);
    expect(result.current.stream).toBeNull();
    expect(result.current.text).toBe("final");
    // The full take is retained for visualization/playback/re-transcription…
    expect(result.current.clip).toBe(rendered);
    // …so the model must have received a copy, not the retained array.
    for (const [audio] of transcribe.mock.calls) expect(audio).not.toBe(rendered);
  });

  it("runs the final pass even while a live tick is still in flight", async () => {
    installDecodeMocks(new Float32Array([0.2]));
    installMediaMocks();
    // The in-flight tick never resolves; the final pass must not be skipped.
    transcribe.mockReturnValue(new Promise<AsrResult>(() => {}));

    const { useLiveAsr } = await import("./useLiveAsr");
    const { result } = renderHook(() => useLiveAsr());
    await act(async () => {
      await result.current.start();
    });
    await act(async () => lastRecorder.emitChunk());
    await flush();
    act(() => result.current.stop());
    await flush();

    expect(transcribe).toHaveBeenCalledTimes(2); // live tick + forced final pass
  });

  it("transcribeClip transcribes a copy, retains the clip, and sets the text", async () => {
    installMediaMocks();
    transcribe.mockResolvedValue({ text: "from a file" });

    const { useLiveAsr } = await import("./useLiveAsr");
    const { result } = renderHook(() => useLiveAsr());

    const clip = new Float32Array([0.5]);
    await act(async () => {
      await result.current.transcribeClip(clip);
    });

    expect(transcribe).toHaveBeenCalledWith(new Float32Array([0.5]), undefined);
    // The original is retained; the worker got a copy (transfer would detach it).
    expect(result.current.clip).toBe(clip);
    expect(transcribe.mock.calls[0][0]).not.toBe(clip);
    expect(result.current.text).toBe("from a file");
  });

  it("exposes timestamped segments from the model", async () => {
    installDecodeMocks(new Float32Array([0.1]));
    installMediaMocks();
    transcribe.mockResolvedValue({
      text: "hello world",
      chunks: [
        { text: " hello", timestamp: [0, 1] },
        { text: " world", timestamp: [1, 2] },
      ],
    });

    const { useLiveAsr } = await import("./useLiveAsr");
    const { result } = renderHook(() => useLiveAsr());

    await act(async () => {
      await result.current.transcribeClip(new Float32Array([0.5]));
    });

    expect(result.current.chunks).toEqual([
      { text: " hello", timestamp: [0, 1] },
      { text: " world", timestamp: [1, 2] },
    ]);
  });

  it("shifts live segment timestamps to be relative to the whole take", async () => {
    // A take longer than the 30 s window: the model only sees the tail, so its
    // timestamps restart at 0 and would rewind on screen without the offset.
    const takeSeconds = 40;
    const full = new Float32Array(takeSeconds * 16000);
    installDecodeMocks(full);
    installMediaMocks();
    transcribe.mockResolvedValue({
      text: "tail",
      chunks: [{ text: " tail", timestamp: [2, 4] }],
    });

    const { useLiveAsr } = await import("./useLiveAsr");
    const { result } = renderHook(() => useLiveAsr());

    await act(async () => {
      await result.current.start();
    });
    await act(async () => lastRecorder.emitChunk());
    await flush();

    // Window starts at 40 s − 30 s = 10 s, so [2,4] becomes [12,14].
    await waitFor(() =>
      expect(result.current.chunks).toEqual([
        { text: " tail", timestamp: [12, 14] },
      ]),
    );
  });

  it("leaves an open-ended segment end null when shifting", async () => {
    const full = new Float32Array(40 * 16000);
    installDecodeMocks(full);
    installMediaMocks();
    transcribe.mockResolvedValue({
      text: "open",
      chunks: [{ text: " open", timestamp: [5, null] }],
    });

    const { useLiveAsr } = await import("./useLiveAsr");
    const { result } = renderHook(() => useLiveAsr());

    await act(async () => {
      await result.current.start();
    });
    await act(async () => lastRecorder.emitChunk());
    await flush();

    await waitFor(() =>
      expect(result.current.chunks).toEqual([
        { text: " open", timestamp: [15, null] },
      ]),
    );
  });

  it("clears segments when a new capture starts", async () => {
    installDecodeMocks(new Float32Array([0.1]));
    installMediaMocks();
    transcribe.mockResolvedValue({
      text: "x",
      chunks: [{ text: " x", timestamp: [0, 1] }],
    });

    const { useLiveAsr } = await import("./useLiveAsr");
    const { result } = renderHook(() => useLiveAsr());

    await act(async () => {
      await result.current.transcribeClip(new Float32Array([0.5]));
    });
    expect(result.current.chunks).toHaveLength(1);

    await act(async () => {
      await result.current.start();
    });
    expect(result.current.chunks).toEqual([]);
  });

  it("clears the previous take when a new capture starts", async () => {
    installDecodeMocks(new Float32Array([0.1]));
    installMediaMocks();
    transcribe.mockResolvedValue({ text: "x" });

    const { useLiveAsr } = await import("./useLiveAsr");
    const { result } = renderHook(() => useLiveAsr());

    await act(async () => {
      await result.current.transcribeClip(new Float32Array([0.5]));
    });
    expect(result.current.clip).not.toBeNull();

    await act(async () => {
      await result.current.start();
    });
    expect(result.current.clip).toBeNull();
  });
});
