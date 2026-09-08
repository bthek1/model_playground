import { describe, expect, it, vi } from "vitest";

import type { VadSession } from "./session";
import type { VadResponse } from "./types";
import { FRAME_SAMPLES } from "./vad";
import { createVadHandler, type SessionFactory } from "./vadEngine";

function fakeSession(overrides: Partial<VadSession> = {}): VadSession {
  return {
    backend: "wasm",
    dispose: vi.fn(async () => {}),
    probabilities: async (audio) =>
      new Float32Array(Math.ceil(audio.length / FRAME_SAMPLES)).fill(0.7),
    ...overrides,
  };
}

/** `Array.prototype.at` is outside this project's ES2020 lib target. */
function last<T>(items: readonly T[]): T | undefined {
  return items[items.length - 1];
}

describe("createVadHandler", () => {
  it("loads a session and posts progress, warm-up, then ready", async () => {
    const posted: VadResponse[] = [];
    const factory: SessionFactory = async (_model, onProgress) => {
      onProgress({ status: "progress", file: "onnx/model.onnx", progress: 40 });
      return fakeSession();
    };

    const handle = createVadHandler((m) => posted.push(m), factory);
    await handle({ type: "load", model: "onnx-community/silero-vad" });

    expect(posted.map((p) => p.type)).toEqual(["progress", "progress", "ready"]);
    expect(posted[1]).toEqual({
      type: "progress",
      progress: { status: "warmup" },
    });
    expect(posted[2]).toEqual({
      type: "ready",
      model: "onnx-community/silero-vad",
      backend: "wasm",
    });
  });

  it("warms up before ready, so the first clip does not pay graph init", async () => {
    const probabilities = vi.fn(async () => new Float32Array(3));
    const handle = createVadHandler(
      () => {},
      async () => fakeSession({ probabilities }),
    );
    await handle({ type: "load", model: "m" });
    expect(probabilities).toHaveBeenCalledTimes(1);
  });

  it("does not fail the load when warm-up throws", async () => {
    const posted: VadResponse[] = [];
    const handle = createVadHandler(
      (m) => posted.push(m),
      async () =>
        fakeSession({
          probabilities: vi.fn(async () => {
            throw new Error("graph init hiccup");
          }),
        }),
    );
    await handle({ type: "load", model: "m" });
    expect(last(posted)?.type).toBe("ready");
  });

  it("disposes the previous session before opening the next, reference first", async () => {
    const order: string[] = [];
    const dispose = vi.fn(async () => void order.push("dispose"));
    const factory = vi.fn(async () => {
      order.push("create");
      return fakeSession({ dispose });
    });

    const handle = createVadHandler(() => {}, factory, { warmup: false });
    await handle({ type: "load", model: "a" });
    await handle({ type: "load", model: "b" });

    expect(order).toEqual(["create", "dispose", "create"]);
  });

  it("survives a failing teardown rather than leaving a stale model live", async () => {
    const posted: VadResponse[] = [];
    let n = 0;
    const handle = createVadHandler(
      (m) => posted.push(m),
      async () =>
        fakeSession({
          dispose: vi.fn(async () => {
            throw new Error(`release failed ${++n}`);
          }),
        }),
      { warmup: false },
    );
    await handle({ type: "load", model: "a" });
    await handle({ type: "load", model: "b" });
    expect(posted.filter((p) => p.type === "error")).toHaveLength(0);
    expect(last(posted)).toMatchObject({ type: "ready", model: "b" });
  });

  it("returns probabilities with the frame geometry the page needs", async () => {
    const posted: VadResponse[] = [];
    const handle = createVadHandler(
      (m) => posted.push(m),
      async () => fakeSession(),
      { warmup: false },
    );
    await handle({ type: "load", model: "m" });
    await handle({
      type: "run",
      id: 1,
      audio: new Float32Array(FRAME_SAMPLES * 4),
    });

    const result = last(posted);
    expect(result).toMatchObject({ type: "result", id: 1 });
    expect(result).toHaveProperty("result.frameSamples", FRAME_SAMPLES);
    expect(result).toHaveProperty("result.sampleRate", 16000);
    // The page draws the timeline against the clip, so it needs the clip length.
    expect(result).toHaveProperty("result.samples", FRAME_SAMPLES * 4);
  });

  it("reports a run before any load as a run error, keeping the id", async () => {
    // `id` is the discriminator: with it the page stays `ready` and shows the
    // failure in OUTPUT; without it the load machine would drop to `error`.
    const posted: VadResponse[] = [];
    const handle = createVadHandler(
      (m) => posted.push(m),
      async () => fakeSession(),
    );
    await handle({ type: "run", id: 7, audio: new Float32Array(FRAME_SAMPLES) });
    expect(last(posted)).toEqual({
      type: "error",
      id: 7,
      error: "No model loaded",
    });
  });

  it("reports a failed load as a load error, with no id", async () => {
    const posted: VadResponse[] = [];
    const handle = createVadHandler(
      (m) => posted.push(m),
      async () => {
        throw new Error("onnx/model.onnx: 404 Not Found");
      },
    );
    await handle({ type: "load", model: "onnx-community/nope" });
    expect(last(posted)).toEqual({
      type: "error",
      error: "onnx/model.onnx: 404 Not Found",
    });
    expect(last(posted)).not.toHaveProperty("id");
  });
});
