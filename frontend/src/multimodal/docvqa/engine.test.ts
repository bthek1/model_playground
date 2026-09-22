import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  createDocVqaHandler,
  type DocAnswerer,
  type DocAnswererOpts,
} from "./engine";
import type { DocVqaResponse } from "./types";

/** `src` targets ES2020, which has no `Array.prototype.at`. */
const last = <T,>(items: T[]): T | undefined => items[items.length - 1];

const pickBackend = vi.fn(async () => "wasm" as const);
vi.mock("@/model/backend", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, pickBackend: () => pickBackend() };
});

const image = {
  data: new Uint8ClampedArray(3),
  width: 1,
  height: 1,
  channels: 3,
} as const;

function fakeAnswerer(overrides: Partial<DocAnswerer> = {}) {
  const answer = vi.fn<DocAnswerer["answer"]>(async () => "us-001");
  const dispose = vi.fn(async () => {});
  return { answer, dispose, ...overrides } as DocAnswerer & {
    answer: typeof answer;
    dispose: typeof dispose;
  };
}

function harness(
  factoryImpl?: (model: string) => Promise<DocAnswerer>,
  opts: { warmup?: boolean } = {},
) {
  const posted: DocVqaResponse[] = [];
  const models: Array<ReturnType<typeof fakeAnswerer>> = [];
  const factory = vi.fn(async (model: string, o: DocAnswererOpts) => {
    void o;
    if (factoryImpl) return factoryImpl(model);
    const m = fakeAnswerer();
    models.push(m);
    return m;
  });
  const handle = createDocVqaHandler((m) => posted.push(m), factory, opts);
  return { handle, posted, factory, models };
}

const load = { type: "load" as const, model: "donut" };
const run = {
  type: "run" as const,
  id: 3,
  image,
  question: "What is the invoice number?",
  maxNewTokens: 128,
};

beforeEach(() => {
  vi.clearAllMocks();
  pickBackend.mockResolvedValue("wasm");
});

describe("createDocVqaHandler — load", () => {
  it("takes the shared dtype rather than a VLM one", async () => {
    // Donut is an encoder plus a short extractive decode, not a chat decoder:
    // `loadOpts()` is right here, and `vlmLoadOpts()` would be an inherited
    // decision rather than a made one.
    const { handle, factory, posted } = harness(undefined, { warmup: false });
    await handle(load);
    expect(factory.mock.calls[0][1]).toMatchObject({
      device: "wasm",
      dtype: "q8",
    });
    expect(last(posted)).toEqual({
      type: "ready",
      model: "donut",
      backend: "wasm",
    });
  });

  it("warms up before ready, cheaply, and says so", async () => {
    const { handle, posted, models } = harness();
    await handle(load);

    const warmAt = posted.findIndex(
      (m) => m.type === "progress" && m.progress.status === "warmup",
    );
    const readyAt = posted.findIndex((m) => m.type === "ready");
    expect(warmAt).toBeGreaterThanOrEqual(0);
    expect(warmAt).toBeLessThan(readyAt);
    // A couple of tokens: this is a ~219-411 MB load already.
    expect(models[0].answer.mock.calls[0][2]).toBeLessThanOrEqual(4);
  });

  it("does not fail the load when the warm-up throws", async () => {
    const broken = fakeAnswerer({
      answer: vi.fn().mockRejectedValue(new Error("shader compile")),
    });
    const { handle, posted } = harness(async () => broken);
    await handle(load);
    expect(last(posted)).toMatchObject({ type: "ready" });
  });

  it("reports a load failure with no id, so it reaches Machine A", async () => {
    const { handle, posted } = harness(async () => {
      throw new Error("404 not found");
    });
    await handle(load);
    expect(last(posted)).toEqual({ type: "error", error: "404 not found" });
  });

  it("keeps one model live, and a throwing dispose does not block the next load", async () => {
    const stubborn = fakeAnswerer({
      dispose: vi.fn().mockRejectedValue(new Error("device lost")),
    });
    let first = true;
    const { handle, posted } = harness(async () => {
      if (first) {
        first = false;
        return stubborn;
      }
      return fakeAnswerer();
    }, { warmup: false });

    await handle(load);
    await handle({ ...load, model: "donut-2" });
    expect(last(posted)).toMatchObject({ type: "ready", model: "donut-2" });
  });
});

describe("createDocVqaHandler — run", () => {
  it("returns the answer with the question it was asked", async () => {
    const { handle, posted } = harness(undefined, { warmup: false });
    await handle(load);
    await handle(run);

    expect(last(posted)).toMatchObject({
      type: "result",
      id: 3,
      result: { answer: "us-001", question: "What is the invoice number?" },
    });
  });

  it("passes a null answer through instead of flattening it to a string", async () => {
    // The pipeline returns `{ answer: null }` when its `<s_answer>` regex
    // misses. "Found nothing" and "found an empty span" are different things to
    // say on screen, so the engine must not collapse them here.
    const silent = fakeAnswerer({ answer: vi.fn(async () => null) });
    const { handle, posted } = harness(async () => silent, { warmup: false });
    await handle(load);
    await handle(run);

    const out = last(posted);
    expect(out).toMatchObject({ type: "result", id: 3 });
    expect(out?.type === "result" && out.result.answer).toBeNull();
  });

  it("rejects a run with no model loaded, carrying the id", async () => {
    const { handle, posted } = harness(undefined, { warmup: false });
    await handle(run);
    expect(last(posted)).toEqual({
      type: "error",
      id: 3,
      error: "No model loaded",
    });
  });

  it("reports an inference failure against the id, leaving the model loaded", async () => {
    const flaky = fakeAnswerer({
      answer: vi.fn().mockRejectedValue(new Error("Non-zero status code")),
    });
    const { handle, posted } = harness(async () => flaky, { warmup: false });
    await handle(load);
    await handle(run);
    expect(last(posted)).toEqual({
      type: "error",
      id: 3,
      error: "Non-zero status code",
    });
  });
});

describe("the DocVQA catalogue", () => {
  it("declares both graphs, so the size check looks at the right files", async () => {
    // The repo publishes three *alternative* decoders; only the merged one is
    // loaded, and a naive sum over the repo quotes a 4 GB model.
    const { DOCVQA_MODELS } = await import("./types");
    for (const m of DOCVQA_MODELS) {
      expect(m.graphs).toEqual(["encoder_model", "decoder_model_merged"]);
      expect(m.bytes.webgpu).toBeGreaterThan(0);
      expect(m.bytes.wasm).toBeGreaterThan(0);
    }
  });

  it("gates no backend, because neither was measured", async () => {
    // An encoder plus a short extractive decode is a real CPU path, unlike the
    // chat decoders in `multimodal/types.ts`. Declaring `backends` without a
    // measurement would be the guess this catalogue avoids.
    const { DOCVQA_MODELS } = await import("./types");
    for (const m of DOCVQA_MODELS) expect(m.backends).toBeUndefined();
  });

  it("caps the source at the processor's own dimension, not below it", async () => {
    // `do_resize` + `do_thumbnail` + `do_pad` at a fixed 2560x1920, and
    // `thumbnail()` never upscales — so anything smaller is padded, costing
    // legibility at identical compute. The cap is a memory bound, not
    // preprocessing, and it must not drop below what the processor uses.
    const { MAX_SOURCE_SIDE } = await import("./types");
    expect(MAX_SOURCE_SIDE).toBe(2560);
  });
});
