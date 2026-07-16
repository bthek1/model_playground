import { afterEach, describe, expect, it, vi } from "vitest";

import { decodeToMono, play, recordMic, toWavBlob } from "./io";

/** Read a Blob's bytes (happy-dom implements Blob.arrayBuffer). */
async function bytes(blob: Blob): Promise<DataView> {
  return new DataView(await blob.arrayBuffer());
}

/** A minimal stand-in for an AudioBuffer: fixed duration + one channel. */
class FakeAudioBuffer {
  constructor(
    public duration: number,
    private data: Float32Array,
  ) {}
  getChannelData() {
    return this.data;
  }
}

/**
 * Stub the Web Audio decode path (`AudioContext` + `OfflineAudioContext`). The
 * `OfflineAudioContext` returns `rendered` from `startRendering`, and its
 * constructor args are captured so tests can assert the resample target.
 */
function installDecodeMocks(rendered: Float32Array, duration: number) {
  const decodeAudioData = vi
    .fn()
    .mockResolvedValue(new FakeAudioBuffer(duration, new Float32Array(0)));
  const close = vi.fn().mockResolvedValue(undefined);
  class FakeAudioContext {
    decodeAudioData = decodeAudioData;
    close = close;
  }
  vi.stubGlobal("AudioContext", FakeAudioContext);

  const captured: { channels?: number; frames?: number; rate?: number } = {};
  const start = vi.fn();
  const connect = vi.fn();
  class FakeOfflineAudioContext {
    destination = {};
    constructor(channels: number, frames: number, rate: number) {
      captured.channels = channels;
      captured.frames = frames;
      captured.rate = rate;
    }
    createBufferSource() {
      return { buffer: null, connect, start };
    }
    startRendering() {
      return Promise.resolve(new FakeAudioBuffer(duration, rendered));
    }
  }
  vi.stubGlobal("OfflineAudioContext", FakeOfflineAudioContext);

  return { captured, decodeAudioData, close, start };
}

function str(view: DataView, offset: number, length: number): string {
  let s = "";
  for (let i = 0; i < length; i++) s += String.fromCharCode(view.getUint8(offset + i));
  return s;
}

describe("toWavBlob", () => {
  it("emits a well-formed 16-bit PCM WAV header", async () => {
    const samples = new Float32Array([0, 0.5, -0.5, 1, -1]);
    const blob = toWavBlob(samples, 16000);

    expect(blob.type).toBe("audio/wav");
    expect(blob.size).toBe(44 + samples.length * 2);

    const view = await bytes(blob);
    expect(str(view, 0, 4)).toBe("RIFF");
    expect(str(view, 8, 4)).toBe("WAVE");
    expect(str(view, 12, 4)).toBe("fmt ");
    expect(str(view, 36, 4)).toBe("data");
    expect(view.getUint16(20, true)).toBe(1); // PCM
    expect(view.getUint16(22, true)).toBe(1); // mono
    expect(view.getUint32(24, true)).toBe(16000); // sample rate
    expect(view.getUint16(34, true)).toBe(16); // bits per sample
    expect(view.getUint32(40, true)).toBe(samples.length * 2); // data size
  });

  it("clamps and quantizes samples to full-scale int16", async () => {
    const view = await bytes(toWavBlob(new Float32Array([1, -1, 2, 0]), 8000));
    expect(view.getInt16(44, true)).toBe(0x7fff); // +1 → positive full-scale
    expect(view.getInt16(46, true)).toBe(-0x8000); // -1 → negative full-scale
    expect(view.getInt16(48, true)).toBe(0x7fff); // 2 clamps to +1 → positive full-scale
    expect(view.getInt16(50, true)).toBe(0); // 0 → 0
  });
});

describe("decodeToMono", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("resamples through an OfflineAudioContext at the target rate", async () => {
    const rendered = new Float32Array([0.1, 0.2, 0.3]);
    const { captured, decodeAudioData, close } = installDecodeMocks(rendered, 2);

    const out = await decodeToMono(new ArrayBuffer(8), 16000);

    expect(out).toBe(rendered); // returns the rendered mono channel
    expect(decodeAudioData).toHaveBeenCalledTimes(1);
    expect(close).toHaveBeenCalledTimes(1); // the scratch AudioContext is closed
    // 1 channel, ceil(duration × rate) frames, at the target rate.
    expect(captured.channels).toBe(1);
    expect(captured.frames).toBe(2 * 16000);
    expect(captured.rate).toBe(16000);
  });

  it("renders at least one frame for a near-zero-length clip", async () => {
    const { captured } = installDecodeMocks(new Float32Array([0]), 0);
    await decodeToMono(new ArrayBuffer(8), 16000);
    expect(captured.frames).toBe(1); // Math.max(1, …) guards against 0 frames
  });
});

describe("play", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("copies samples into a mono AudioBuffer and starts playback", () => {
    const channel = new Float32Array(3);
    const createBuffer = vi.fn(() => ({ getChannelData: () => channel }));
    const start = vi.fn();
    const connect = vi.fn();
    class FakeAudioContext {
      createBuffer = createBuffer;
      destination = {};
      createBufferSource() {
        return { buffer: null, connect, start };
      }
    }
    vi.stubGlobal("AudioContext", FakeAudioContext);

    const samples = new Float32Array([0.4, 0.5, 0.6]);
    play(samples, 22050);

    expect(createBuffer).toHaveBeenCalledWith(1, 3, 22050);
    expect(Array.from(channel)).toEqual([
      expect.closeTo(0.4),
      expect.closeTo(0.5),
      expect.closeTo(0.6),
    ]);
    expect(connect).toHaveBeenCalledTimes(1);
    expect(start).toHaveBeenCalledTimes(1);
  });
});

describe("recordMic", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("captures a mic clip, decodes it, and stops the tracks", async () => {
    const rendered = new Float32Array([0.7, 0.8]);
    installDecodeMocks(rendered, 1);

    const track = { stop: vi.fn() };
    const stream = { getTracks: () => [track] };
    Object.defineProperty(navigator, "mediaDevices", {
      value: { getUserMedia: vi.fn().mockResolvedValue(stream) },
      configurable: true,
      writable: true,
    });

    // A MediaRecorder that, on stop(), emits one chunk then fires onstop.
    class FakeMediaRecorder {
      ondataavailable: ((e: { data: Blob }) => void) | null = null;
      onstop: (() => void) | null = null;
      start = vi.fn();
      stop = vi.fn(() => {
        this.ondataavailable?.({ data: new Blob([new Uint8Array([1, 2, 3])]) });
        this.onstop?.();
      });
    }
    vi.stubGlobal("MediaRecorder", FakeMediaRecorder);

    // seconds = 0 → the internal setTimeout fires on the next tick, keeping the
    // test fast without fake timers.
    const out = await recordMic(0, 16000);

    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({ audio: true });
    expect(out).toBe(rendered);
    expect(track.stop).toHaveBeenCalledTimes(1); // mic released
  });
});
