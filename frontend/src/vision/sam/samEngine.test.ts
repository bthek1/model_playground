import { afterEach, describe, expect, it, vi } from "vitest";

import type { ImagePayload } from "../image";
import { createSamHandler } from "./samEngine";
import type { SamMask, SamResponse } from "./types";

// `pickBackend` (used when a `load` omits opts) touches navigator.gpu; keep it
// absent so the engine deterministically picks the wasm backend.
function clearGpu() {
  Object.defineProperty(navigator, "gpu", {
    value: undefined,
    configurable: true,
    writable: true,
  });
}

function image(side = 4): ImagePayload {
  return {
    data: new Uint8ClampedArray(side * side * 3),
    width: side,
    height: side,
    channels: 3,
  };
}

function mask(score = 0.9): SamMask {
  return { data: new Uint8Array(4), width: 2, height: 2, score };
}

const LOAD = { type: "load", model: "Xenova/slimsam-77-uniform" } as const;
const WASM = { device: "wasm", dtype: "q8" } as const;

/** `Array.prototype.at` is ES2022; the app's lib is ES2020. */
const last = <T,>(items: T[]): T => items[items.length - 1];

describe("createSamHandler", () => {
  afterEach(() => clearGpu());

  it("loads both graphs and posts progress then ready", async () => {
    clearGpu();
    const posted: SamResponse[] = [];
    const factory = vi.fn(async (_model: string, opts) => {
      opts.progress_callback?.({ status: "download", file: "vision_encoder.onnx" });
      return { encode: vi.fn(async () => ({})), decode: vi.fn(async () => [mask()]) };
    });

    const handle = createSamHandler((m) => posted.push(m), factory, {
      warmup: false,
    });
    await handle(LOAD);

    expect(factory).toHaveBeenCalledWith(
      "Xenova/slimsam-77-uniform",
      expect.objectContaining({ device: "wasm", dtype: "q8" }),
    );
    expect(last(posted)).toEqual({
      type: "ready",
      model: "Xenova/slimsam-77-uniform",
      backend: "wasm",
    });
  });

  it("warms both graphs, and leaves nothing encoded behind", async () => {
    // Both compile separately, and a first *click* that pays for the decoder's
    // compile is the one a user reads as "this is slow". The warm-up must not
    // leave its 64x64 grey square encoded — the next real click would decode
    // against it.
    const posted: SamResponse[] = [];
    const encode = vi.fn(async () => ({}));
    const decode = vi.fn(async () => [mask()]);
    const handle = createSamHandler((m) => posted.push(m), async () => ({
      encode,
      decode,
    }));

    await handle({ ...LOAD, opts: WASM });
    expect(encode).toHaveBeenCalledTimes(1);
    expect(decode).toHaveBeenCalledTimes(1);
    expect(posted).toContainEqual({
      type: "progress",
      progress: { status: "warmup" },
    });

    // A real encode after the warm-up is not served from the warm-up's cache.
    await handle({ type: "run", id: 1, kind: "encode", token: "a", image: image() });
    expect(encode).toHaveBeenCalledTimes(2);
  });

  it("reaches ready even when the warm-up throws", async () => {
    const posted: SamResponse[] = [];
    const handle = createSamHandler((m) => posted.push(m), async () => ({
      encode: vi.fn().mockRejectedValue(new Error("shader compile blew up")),
      decode: vi.fn(async () => [mask()]),
    }));

    await handle({ ...LOAD, opts: WASM });
    expect(last(posted)).toMatchObject({ type: "ready" });
  });

  it("answers an encode with its timing and the image's size", async () => {
    const posted: SamResponse[] = [];
    const handle = createSamHandler((m) => posted.push(m), async () => ({
      encode: vi.fn(async () => ({})),
      decode: vi.fn(async () => [mask()]),
    }), { warmup: false });

    await handle({ ...LOAD, opts: WASM });
    await handle({ type: "run", id: 7, kind: "encode", token: "a", image: image(8) });

    expect(last(posted)).toMatchObject({
      type: "result",
      id: 7,
      result: { kind: "encode", cached: false, width: 8, height: 8 },
    });
  });

  it("answers a decode with the candidates, and transfers their buffers", async () => {
    // Three 640x480 masks is 921 KB per click; copying them on every click is
    // exactly the cost this route claims not to pay.
    const posted: { message: SamResponse; transfer?: Transferable[] }[] = [];
    const masks = [mask(0.9), mask(0.5)];
    const handle = createSamHandler(
      (message, transfer) => posted.push({ message, transfer }),
      async () => ({
        encode: vi.fn(async () => ({})),
        decode: vi.fn(async () => masks),
      }),
      { warmup: false },
    );

    await handle({ ...LOAD, opts: WASM });
    await handle({ type: "run", id: 1, kind: "encode", token: "a", image: image() });
    await handle({
      type: "run",
      id: 2,
      kind: "decode",
      points: [{ x: 1, y: 1, positive: true }],
    });

    const reply = last(posted);
    expect(reply.message).toMatchObject({
      type: "result",
      id: 2,
      result: { kind: "decode" },
    });
    expect(reply.transfer).toEqual(masks.map((m) => m.data.buffer));
  });

  it("puts a decode before any encode in Machine B, with the model still loaded", async () => {
    // A per-request failure, correlated by id: `status` must stay `ready` so a
    // different click still works. This is why the encode is a run request
    // rather than a progress event — a progress event has no failure path.
    const posted: SamResponse[] = [];
    const handle = createSamHandler((m) => posted.push(m), async () => ({
      encode: vi.fn(async () => ({})),
      decode: vi.fn(async () => [mask()]),
    }), { warmup: false });

    await handle({ ...LOAD, opts: WASM });
    await handle({
      type: "run",
      id: 3,
      kind: "decode",
      points: [{ x: 1, y: 1, positive: true }],
    });

    expect(last(posted)).toEqual({
      type: "error",
      id: 3,
      error: expect.stringMatching(/encode an image/i),
    });
  });

  it("disposes the previous model on reload, nulling the reference first", async () => {
    const posted: SamResponse[] = [];
    const dispose = vi.fn().mockRejectedValue(new Error("teardown blew up"));
    let made = 0;
    const handle = createSamHandler(
      (m) => posted.push(m),
      async () => ({
        encode: vi.fn(async () => ({ n: ++made })),
        decode: vi.fn(async () => [mask()]),
        dispose,
      }),
      { warmup: false },
    );

    await handle({ ...LOAD, opts: WASM });
    await handle({ ...LOAD, model: "onnx-community/sam2.1-hiera-tiny-ONNX", opts: WASM });

    // A teardown that throws must not fail the load, and must not leave the
    // old model live.
    expect(dispose).toHaveBeenCalled();
    expect(last(posted)).toMatchObject({ type: "ready" });
  });

  it("drops a cached embedding when a new model is loaded", async () => {
    // Vectors from another checkpoint are not decodable by this one.
    const posted: SamResponse[] = [];
    const handle = createSamHandler((m) => posted.push(m), async () => ({
      encode: vi.fn(async () => ({})),
      decode: vi.fn(async () => [mask()]),
    }), { warmup: false });

    await handle({ ...LOAD, opts: WASM });
    await handle({ type: "run", id: 1, kind: "encode", token: "a", image: image() });
    await handle({ ...LOAD, opts: WASM });
    await handle({
      type: "run",
      id: 2,
      kind: "decode",
      points: [{ x: 1, y: 1, positive: true }],
    });

    expect(last(posted)).toMatchObject({ type: "error", id: 2 });
  });

  it("refuses to run at all before a model is loaded", async () => {
    const posted: SamResponse[] = [];
    const handle = createSamHandler((m) => posted.push(m), async () => ({
      encode: vi.fn(async () => ({})),
      decode: vi.fn(async () => [mask()]),
    }));

    await handle({ type: "run", id: 1, kind: "encode", token: "a", image: image() });
    expect(last(posted)).toEqual({
      type: "error",
      id: 1,
      error: "No model loaded",
    });
  });
});
