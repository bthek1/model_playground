import { afterEach, describe, expect, it, vi } from "vitest";

import type { ImagePayload } from "../image";
import { createCaptionHandler, type Captioner } from "./engine";
import type { CaptionResponse } from "./types";

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

const LOAD = {
  type: "load",
  model: "onnx-community/Florence-2-base-ft",
  family: "florence2",
} as const;
const WASM = { device: "wasm", dtype: "q8" } as const;

/** `Array.prototype.at` is ES2022; the app's lib is ES2020. */
const last = <T,>(items: T[]): T => items[items.length - 1];

function captioner(over: Partial<Captioner> = {}): Captioner {
  return {
    generate: vi.fn(async () => ({ kind: "text" as const, text: "a cat" })),
    ...over,
  };
}

describe("createCaptionHandler", () => {
  afterEach(() => clearGpu());

  it("passes the family through to the factory, and posts ready", async () => {
    clearGpu();
    const posted: CaptionResponse[] = [];
    const factory = vi.fn(async (_model: string, opts) => {
      opts.progress_callback?.({ status: "download", file: "encoder_model.onnx" });
      return captioner();
    });

    const handle = createCaptionHandler((m) => posted.push(m), factory, {
      warmup: false,
    });
    await handle(LOAD);

    expect(factory).toHaveBeenCalledWith(
      "onnx-community/Florence-2-base-ft",
      expect.objectContaining({ device: "wasm", dtype: "q8", family: "florence2" }),
    );
    expect(last(posted)).toEqual({
      type: "ready",
      model: "onnx-community/Florence-2-base-ft",
      backend: "wasm",
    });
  });

  it("warms up with a token budget, not a whole caption", async () => {
    // Every other engine warms up with one forward pass. This one has an
    // autoregressive decoder, so a full generation would add seconds to a load
    // that is already the longest in the category.
    const generate = vi.fn(async () => ({ kind: "text" as const, text: "" }));
    const posted: CaptionResponse[] = [];
    const handle = createCaptionHandler(
      (m) => posted.push(m),
      async () => captioner({ generate }),
    );

    await handle({ ...LOAD, opts: WASM });
    expect(generate).toHaveBeenCalledTimes(1);
    const [, mode, tokens] = generate.mock.calls[0] as unknown as [
      unknown,
      string,
      number,
    ];
    expect(mode).toBe("<CAPTION>");
    expect(tokens).toBeLessThanOrEqual(4);
    expect(posted).toContainEqual({
      type: "progress",
      progress: { status: "warmup" },
    });
  });

  it("reaches ready even when the warm-up throws", async () => {
    const posted: CaptionResponse[] = [];
    const handle = createCaptionHandler(
      (m) => posted.push(m),
      async () =>
        captioner({
          generate: vi.fn().mockRejectedValue(new Error("shader compile blew up")),
        }),
    );
    await handle({ ...LOAD, opts: WASM });
    expect(last(posted)).toMatchObject({ type: "ready" });
  });

  it("returns a text answer tagged with the mode that produced it", async () => {
    const posted: CaptionResponse[] = [];
    const handle = createCaptionHandler(
      (m) => posted.push(m),
      async () =>
        captioner({
          generate: vi.fn(async () => ({ kind: "text" as const, text: "two cats" })),
        }),
      { warmup: false },
    );

    await handle({ ...LOAD, opts: WASM });
    await handle({
      type: "run",
      id: 5,
      image: image(),
      mode: "<OCR>",
      maxNewTokens: 64,
    });

    expect(last(posted)).toMatchObject({
      type: "result",
      id: 5,
      result: { kind: "text", mode: "<OCR>", text: "two cats" },
    });
  });

  it("returns boxes as boxes, so grounding never renders as prose", async () => {
    const detections = [
      { label: "a cat", score: 1, box: { xmin: 1, ymin: 2, xmax: 3, ymax: 4 } },
    ];
    const posted: CaptionResponse[] = [];
    const handle = createCaptionHandler(
      (m) => posted.push(m),
      async () =>
        captioner({
          generate: vi.fn(async () => ({ kind: "boxes" as const, detections })),
        }),
      { warmup: false },
    );

    await handle({ ...LOAD, opts: WASM });
    await handle({
      type: "run",
      id: 1,
      image: image(),
      mode: "<OD>",
      maxNewTokens: 512,
    });

    expect(last(posted)).toMatchObject({
      type: "result",
      result: { kind: "boxes", mode: "<OD>", detections },
    });
  });

  it("forwards the token budget the caller asked for", async () => {
    // A caption is a sentence and a detailed one is a paragraph; the cap is
    // what keeps the cheap mode cheap.
    const generate = vi.fn(async () => ({ kind: "text" as const, text: "x" }));
    const handle = createCaptionHandler(
      () => {},
      async () => captioner({ generate }),
      { warmup: false },
    );
    await handle({ ...LOAD, opts: WASM });
    await handle({
      type: "run",
      id: 1,
      image: image(),
      mode: "<DETAILED_CAPTION>",
      maxNewTokens: 256,
    });
    expect(generate).toHaveBeenLastCalledWith(
      expect.anything(),
      "<DETAILED_CAPTION>",
      256,
    );
  });

  it("puts a generation failure in Machine B, with the model still loaded", async () => {
    const posted: CaptionResponse[] = [];
    const handle = createCaptionHandler(
      (m) => posted.push(m),
      async () =>
        captioner({
          generate: vi.fn().mockRejectedValue(new Error("Non-zero status code")),
        }),
      { warmup: false },
    );

    await handle({ ...LOAD, opts: WASM });
    await handle({
      type: "run",
      id: 2,
      image: image(),
      mode: "<CAPTION>",
      maxNewTokens: 64,
    });

    expect(last(posted)).toEqual({
      type: "error",
      id: 2,
      error: "Non-zero status code",
    });
  });

  it("disposes the previous model on reload, nulling the reference first", async () => {
    const dispose = vi.fn().mockRejectedValue(new Error("teardown blew up"));
    const posted: CaptionResponse[] = [];
    const handle = createCaptionHandler(
      (m) => posted.push(m),
      async () => captioner({ dispose }),
      { warmup: false },
    );

    await handle({ ...LOAD, opts: WASM });
    await handle({
      ...LOAD,
      model: "Xenova/vit-gpt2-image-captioning",
      family: "vision-encoder-decoder",
      opts: WASM,
    });

    expect(dispose).toHaveBeenCalled();
    expect(last(posted)).toMatchObject({ type: "ready" });
  });

  it("refuses to run before a model is loaded", async () => {
    const posted: CaptionResponse[] = [];
    const handle = createCaptionHandler(
      (m) => posted.push(m),
      async () => captioner(),
    );
    await handle({
      type: "run",
      id: 1,
      image: image(),
      mode: "<CAPTION>",
      maxNewTokens: 64,
    });
    expect(last(posted)).toEqual({
      type: "error",
      id: 1,
      error: "No model loaded",
    });
  });
});
