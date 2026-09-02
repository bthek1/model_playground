import { afterEach, describe, expect, it, vi } from "vitest";

import { createTtsHandler, type TtsSynthesizer } from "./ttsEngine";
import type { TtsResponse } from "./tts";

// pickBackend (used when a `load` omits opts) touches navigator.gpu; keep it
// absent so the engine deterministically picks the wasm backend.
function clearGpu() {
  Object.defineProperty(navigator, "gpu", {
    value: undefined,
    configurable: true,
    writable: true,
  });
}

describe("createTtsHandler", () => {
  afterEach(() => clearGpu());

  it("loads a model and posts progress then ready", async () => {
    clearGpu();
    const posted: TtsResponse[] = [];
    const synth = vi.fn() as unknown as TtsSynthesizer;
    const factory = vi.fn(async (_model: string, opts) => {
      opts.progress_callback?.({ status: "download", file: "model.onnx", progress: 55 });
      return synth;
    });

    const handle = createTtsHandler((m) => posted.push(m), factory, {
      warmup: false,
    });
    await handle({ type: "load", model: "onnx-community/Kokoro-82M-v1.0-ONNX" });

    expect(factory).toHaveBeenCalledWith(
      "onnx-community/Kokoro-82M-v1.0-ONNX",
      expect.objectContaining({ device: "wasm", dtype: "q8" }),
    );
    expect(posted).toEqual([
      { type: "progress", progress: { status: "download", file: "model.onnx", progress: 55 } },
      { type: "ready", model: "onnx-community/Kokoro-82M-v1.0-ONNX", backend: "wasm" },
    ]);
  });

  it("synthesises text and transfers the audio buffer back", async () => {
    const posted: Array<{ message: TtsResponse; transfer?: Transferable[] }> = [];
    const audio = new Float32Array([0.1, -0.2, 0.3]);
    const synth = vi
      .fn()
      .mockResolvedValue({ audio, sampleRate: 24000 }) as unknown as TtsSynthesizer;
    const handle = createTtsHandler(
      (message, transfer) => posted.push({ message, transfer }),
      async () => synth,
    );

    await handle({ type: "load", model: "m", opts: { device: "wasm", dtype: "q8" } });
    await handle({ type: "run", id: 4, text: "hello", opts: { voice: "af_heart" } });

    expect(synth).toHaveBeenCalledWith("hello", { voice: "af_heart" });
    const last = posted[posted.length - 1];
    expect(last.message).toEqual({
      type: "result",
      id: 4,
      result: { audio, sampleRate: 24000 },
    });
    expect(last.transfer).toEqual([audio.buffer]); // zero-copy back to main
  });

  it("synthesises a short warm-up phrase before reporting ready", async () => {
    clearGpu();
    const posted: TtsResponse[] = [];
    const synth = vi
      .fn()
      .mockResolvedValue({ audio: new Float32Array(4), sampleRate: 24000 }) as unknown as TtsSynthesizer;
    const handle = createTtsHandler((m) => posted.push(m), async () => synth);

    await handle({ type: "load", model: "m", opts: { device: "wasm", dtype: "q8" } });

    expect(synth).toHaveBeenCalledTimes(1);
    // A tiny token budget: MusicGen would otherwise generate 256 tokens (~5 s of
    // music) on every load just to warm up.
    expect(synth).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ maxNewTokens: expect.any(Number) }),
    );
    const [, warmOpts] = (synth as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(warmOpts.maxNewTokens).toBeLessThanOrEqual(32);
    expect(posted).toEqual([
      { type: "progress", progress: { status: "warmup" } },
      { type: "ready", model: "m", backend: "wasm" },
    ]);
  });

  it("still reports ready when the warm-up synthesis throws", async () => {
    clearGpu();
    const posted: TtsResponse[] = [];
    const synth = vi
      .fn()
      .mockRejectedValue(new Error("no voice")) as unknown as TtsSynthesizer;
    const handle = createTtsHandler((m) => posted.push(m), async () => synth);

    await handle({ type: "load", model: "m", opts: { device: "wasm", dtype: "q8" } });

    expect(posted[posted.length - 1]).toEqual({
      type: "ready",
      model: "m",
      backend: "wasm",
    });
  });

  it("errors a run when no model is loaded", async () => {
    const posted: TtsResponse[] = [];
    const handle = createTtsHandler((m) => posted.push(m), async () => vi.fn() as never);

    await handle({ type: "run", id: 9, text: "hi" });

    expect(posted[posted.length - 1]).toEqual({
      type: "error",
      id: 9,
      error: "No TTS model loaded",
    });
  });

  it("reports a load failure as an id-less error", async () => {
    clearGpu();
    const posted: TtsResponse[] = [];
    const handle = createTtsHandler(
      (m) => posted.push(m),
      async () => {
        throw new Error("download failed");
      },
    );

    await handle({ type: "load", model: "m", opts: { device: "wasm", dtype: "q8" } });

    expect(posted[posted.length - 1]).toEqual({ type: "error", error: "download failed" });
  });

  it("disposes the previous model before loading a new one", async () => {
    const dispose = vi.fn().mockResolvedValue(undefined);
    const first = Object.assign(vi.fn(), { dispose }) as unknown as TtsSynthesizer;
    const second = vi.fn() as unknown as TtsSynthesizer;
    const factory = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(second);
    const handle = createTtsHandler(() => {}, factory, { warmup: false });

    await handle({ type: "load", model: "a", opts: { device: "wasm", dtype: "q8" } });
    await handle({ type: "load", model: "b", opts: { device: "wasm", dtype: "q8" } });

    expect(dispose).toHaveBeenCalledTimes(1);
    expect(second).not.toBe(first); // the new model is the live one
  });

  it("loads the next model even when disposing the previous one fails", async () => {
    const dispose = vi.fn().mockRejectedValue(new Error("device lost"));
    const first = Object.assign(vi.fn(), { dispose }) as unknown as TtsSynthesizer;
    const audio = new Float32Array([0.5]);
    const second = vi
      .fn()
      .mockResolvedValue({ audio, sampleRate: 16000 }) as unknown as TtsSynthesizer;
    const posted: TtsResponse[] = [];
    const factory = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(second);
    const handle = createTtsHandler((m) => posted.push(m), factory, { warmup: false });

    await handle({ type: "load", model: "a", opts: { device: "wasm", dtype: "q8" } });
    await handle({ type: "load", model: "b", opts: { device: "wasm", dtype: "q8" } });
    await handle({ type: "run", id: 1, text: "hi" });

    expect(second).toHaveBeenCalledWith("hi", undefined);
    expect(posted[posted.length - 1]).toEqual({
      type: "result",
      id: 1,
      result: { audio, sampleRate: 16000 },
    });
  });
});
