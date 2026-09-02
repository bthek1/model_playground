import { describe, expect, it, vi } from "vitest";

import golden from "./__fixtures__/deepFilterNetFeatures.json";
import { ERB_BANDS } from "./aux";
import { DF_LOOKAHEAD, DF_ORDER } from "./deepFilterNet";
import { createEnhanceHandler, type SessionFactory } from "./enhanceEngine";
import { DF_BINS } from "./features";
import type { DeepFilterSession } from "./session";
import type { EnhanceResponse } from "./types";

/** A session that passes audio straight through, so the DSP is not on trial here. */
function fakeSession(overrides: Partial<DeepFilterSession> = {}): DeepFilterSession {
  return {
    aux: {
      fwd: new Float32Array(0),
      inv: new Float32Array(0),
      window: Float32Array.from(golden.window),
      widths: Int32Array.from(golden.widths),
    },
    backend: "wasm",
    dispose: vi.fn(async () => {}),
    infer: async (_erb, _spec, frames) => {
      const dfCoefs = new Float32Array(DF_ORDER * frames * DF_BINS * 2);
      const tap = DF_ORDER - 1 - DF_LOOKAHEAD;
      for (let t = 0; t < frames; t++) {
        for (let f = 0; f < DF_BINS; f++) dfCoefs[((tap * frames + t) * DF_BINS + f) * 2] = 1;
      }
      return { erbMask: new Float32Array(frames * ERB_BANDS).fill(1), dfCoefs };
    },
    ...overrides,
  };
}

describe("createEnhanceHandler", () => {
  it("loads a session and posts progress, warm-up, then ready", async () => {
    const posted: EnhanceResponse[] = [];
    const factory: SessionFactory = async (_repo, onProgress) => {
      onProgress({ status: "progress", file: "deepfilter.onnx", progress: 40 });
      return fakeSession();
    };

    const handle = createEnhanceHandler((m) => posted.push(m), factory);
    await handle({ type: "load", model: "soniqo/DeepFilterNet3-ONNX" });

    expect(posted.map((p) => p.type)).toEqual(["progress", "progress", "ready"]);
    expect(posted[1]).toEqual({ type: "progress", progress: { status: "warmup" } });
    expect(posted[2]).toEqual({
      type: "ready",
      model: "soniqo/DeepFilterNet3-ONNX",
      backend: "wasm",
    });
  });

  it("forwards a forced backend to the factory", async () => {
    const factory = vi.fn(async () => fakeSession({ backend: "webgpu" }));
    const handle = createEnhanceHandler(() => {}, factory, { warmup: false });
    await handle({ type: "load", model: "m", backend: "webgpu" });
    expect(factory).toHaveBeenCalledWith("m", expect.any(Function), "webgpu");
  });

  it("keeps only one session live, disposing the previous one first", async () => {
    const first = fakeSession();
    const second = fakeSession();
    const factory = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(second);
    const handle = createEnhanceHandler(() => {}, factory, { warmup: false });

    await handle({ type: "load", model: "a" });
    await handle({ type: "load", model: "b" });
    expect(first.dispose).toHaveBeenCalledTimes(1);
    expect(second.dispose).not.toHaveBeenCalled();
  });

  it("still loads when the warm-up inference throws", async () => {
    const posted: EnhanceResponse[] = [];
    const session = fakeSession({
      infer: async () => {
        throw new Error("shader compilation failed");
      },
    });
    const handle = createEnhanceHandler((m) => posted.push(m), async () => session);
    await handle({ type: "load", model: "m" });
    expect(posted[posted.length - 1]?.type).toBe("ready");
  });

  it("reports a load failure with no id — the model never became usable", async () => {
    const posted: EnhanceResponse[] = [];
    const handle = createEnhanceHandler(
      (m) => posted.push(m),
      async () => {
        throw new Error("404 on deepfilter.onnx");
      },
    );
    await handle({ type: "load", model: "m" });
    expect(posted).toEqual([{ type: "error", error: "404 on deepfilter.onnx" }]);
  });

  it("enhances a clip and transfers the result buffer", async () => {
    const posted: EnhanceResponse[] = [];
    const transfers: Transferable[][] = [];
    const handle = createEnhanceHandler(
      (m, t) => {
        posted.push(m);
        if (t) transfers.push(t);
      },
      async () => fakeSession(),
      { warmup: false },
    );
    await handle({ type: "load", model: "m" });

    const audio = new Float32Array(4800);
    for (let i = 0; i < audio.length; i++) audio[i] = Math.sin(i / 20) * 0.3;
    await handle({ type: "run", id: 7, audio });

    const result = posted[posted.length - 1];
    expect(result).toMatchObject({ type: "result", id: 7 });
    if (result?.type !== "result") throw new Error("expected a result");
    expect(result.result.sampleRate).toBe(48000);
    expect(result.result.audio).toHaveLength(4800);
    expect(transfers[transfers.length - 1]).toEqual([result.result.audio.buffer]);
  });

  it("reports a run failure against its id, leaving the model loaded", async () => {
    const posted: EnhanceResponse[] = [];
    const handle = createEnhanceHandler((m) => posted.push(m), async () =>
      fakeSession({
        infer: async () => {
          throw new Error("session run failed");
        },
      }),
    { warmup: false });

    await handle({ type: "load", model: "m" });
    await handle({ type: "run", id: 3, audio: new Float32Array(4800) });
    expect(posted[posted.length - 1]).toEqual({ type: "error", id: 3, error: "session run failed" });
  });

  it("rejects a run before any load, against its id", async () => {
    const posted: EnhanceResponse[] = [];
    const handle = createEnhanceHandler((m) => posted.push(m), async () => fakeSession());
    await handle({ type: "run", id: 1, audio: new Float32Array(480) });
    expect(posted).toEqual([{ type: "error", id: 1, error: "No model loaded" }]);
  });
});
