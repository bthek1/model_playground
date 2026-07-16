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
  running: false,
  error: null as string | null,
  transcribe,
};
vi.mock("@/hooks/useAsr", () => ({ useAsr: () => asrState }));

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

  it("stops capture, runs a final pass, and releases the mic", async () => {
    installDecodeMocks(new Float32Array([0.3]));
    installMediaMocks();
    transcribe.mockResolvedValue({ text: "final" });

    const { useLiveAsr } = await import("./useLiveAsr");
    const { result } = renderHook(() => useLiveAsr());
    await act(async () => {
      await result.current.start();
    });
    await act(async () => lastRecorder.emitChunk());
    await flush();

    act(() => result.current.stop());
    await flush();

    expect(lastRecorder.stop).toHaveBeenCalled();
    expect(track.stop).toHaveBeenCalled();
    expect(result.current.recording).toBe(false);
    expect(result.current.text).toBe("final");
  });

  it("transcribeClip runs a one-shot transcription and sets the text", async () => {
    installMediaMocks();
    transcribe.mockResolvedValue({ text: "from a file" });

    const { useLiveAsr } = await import("./useLiveAsr");
    const { result } = renderHook(() => useLiveAsr());

    await act(async () => {
      await result.current.transcribeClip(new Float32Array([0.5]));
    });

    expect(transcribe).toHaveBeenCalledWith(new Float32Array([0.5]), undefined);
    expect(result.current.text).toBe("from a file");
  });
});
