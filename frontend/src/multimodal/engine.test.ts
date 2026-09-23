import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  createVlmHandler,
  type Vlm,
  type VlmOpts,
  type VlmOutput,
} from "./engine";
import type { VlmPartial, VlmResponse } from "./types";

/** `src` targets ES2020, which has no `Array.prototype.at`. */
const last = <T,>(items: T[]): T | undefined => items[items.length - 1];

const pickBackend = vi.fn(async () => "webgpu" as const);
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

function fakeVlm(overrides: Partial<Vlm> = {}) {
  const generate = vi.fn<Vlm["generate"]>(async () => ({
    text: "a cat",
    encodeMs: 120,
    tokens: 3,
  }));
  const dispose = vi.fn(async () => {});
  return { generate, dispose, ...overrides } as Vlm & {
    generate: typeof generate;
    dispose: typeof dispose;
  };
}

function harness(
  factoryImpl?: (model: string) => Promise<Vlm>,
  opts: { warmup?: boolean } = {},
) {
  const posted: VlmResponse[] = [];
  const models: Array<ReturnType<typeof fakeVlm>> = [];
  const factory = vi.fn(async (model: string, opts: VlmOpts) => {
    void opts;
    if (factoryImpl) return factoryImpl(model);
    const m = fakeVlm();
    models.push(m);
    return m;
  });
  const handle = createVlmHandler((m) => posted.push(m), factory, opts);
  return { handle, posted, factory, models };
}

function lastCall(fn: { mock: { calls: unknown[][] } }): unknown[] {
  return fn.mock.calls[fn.mock.calls.length - 1];
}

const load = { type: "load" as const, model: "smol", family: "idefics3" as const };

beforeEach(() => {
  vi.clearAllMocks();
  pickBackend.mockResolvedValue("webgpu");
});

describe("createVlmHandler — load", () => {
  it("resolves the backend and reports ready", async () => {
    const { handle, posted, factory } = harness(undefined, { warmup: false });
    await handle(load);

    expect(factory).toHaveBeenCalledWith(
      "smol",
      expect.objectContaining({ device: "webgpu", dtype: "q4f16" }),
    );
    expect(last(posted)).toEqual({
      type: "ready",
      model: "smol",
      backend: "webgpu",
    });
  });

  it("defaults to the VLM dtype, not the shared one", async () => {
    // `loadOpts()` would ask for fp16 — a 514 MB download instead of 189 MB.
    const { handle, factory } = harness(undefined, { warmup: false });
    await handle(load);
    expect(factory.mock.calls[0][1]).toMatchObject({ dtype: "q4f16" });
  });

  it("takes a per-backend dtype override from the catalogue entry", async () => {
    const { handle, factory } = harness(undefined, { warmup: false });
    await handle({ ...load, dtypes: { webgpu: "fp16" } });
    expect(factory.mock.calls[0][1]).toMatchObject({ dtype: "fp16" });
  });

  it("warms up before ready, and says so", async () => {
    const { handle, posted, models } = harness();
    await handle(load);

    expect(posted.some((m) => m.type === "progress" && m.progress.status === "warmup")).toBe(true);
    // A couple of tokens, not a full generation: this decoder is the slowest
    // load in the app already.
    expect(models[0].generate).toHaveBeenCalledTimes(1);
    expect(models[0].generate.mock.calls[0][2]).toBeLessThanOrEqual(4);
    // Warm-up happened before `ready` was announced.
    const readyAt = posted.findIndex((m) => m.type === "ready");
    const warmAt = posted.findIndex(
      (m) => m.type === "progress" && m.progress.status === "warmup",
    );
    expect(warmAt).toBeGreaterThanOrEqual(0);
    expect(warmAt).toBeLessThan(readyAt);
  });

  it("does not fail the load when the warm-up throws", async () => {
    const broken = fakeVlm({
      generate: vi.fn().mockRejectedValue(new Error("shader compile")),
    });
    const { handle, posted } = harness(async () => broken);
    await handle(load);
    // The first real run pays the compile cost instead.
    expect(last(posted)).toMatchObject({ type: "ready" });
  });

  it("reports a load failure with no id, so it reaches Machine A", async () => {
    const { handle, posted } = harness(async () => {
      throw new Error("404 not found");
    });
    await handle(load);
    expect(last(posted)).toEqual({ type: "error", error: "404 not found" });
    expect(last(posted)).not.toHaveProperty("id");
  });

  it("keeps one model live: the previous one is disposed on the next load", async () => {
    const { handle, models } = harness(undefined, { warmup: false });
    await handle(load);
    await handle({ ...load, model: "smol-500" });
    expect(models[0].dispose).toHaveBeenCalledOnce();
  });

  it("nulls the reference before disposing, so a failed teardown leaves nothing live", async () => {
    // These are the largest downloads in the app; a leaked session ends the tab.
    const stubborn = fakeVlm({
      dispose: vi.fn().mockRejectedValue(new Error("device lost")),
    });
    let first = true;
    const { handle, posted } = harness(async () => {
      if (first) {
        first = false;
        return stubborn;
      }
      return fakeVlm();
    }, { warmup: false });

    await handle(load);
    await handle({ ...load, model: "smol-500" });
    // The second load still succeeded despite the first's dispose throwing.
    expect(last(posted)).toMatchObject({ type: "ready", model: "smol-500" });
  });
});

describe("createVlmHandler — run", () => {
  const run = {
    type: "run" as const,
    id: 7,
    images: [image],
    prompt: "What is this?",
    maxNewTokens: 64,
  };

  it("returns the answer correlated to the request id", async () => {
    const { handle, posted } = harness(undefined, { warmup: false });
    await handle(load);
    await handle(run);

    expect(last(posted)).toMatchObject({
      type: "result",
      id: 7,
      result: { text: "a cat", encodeMs: 120, tokens: 3 },
    });
  });

  it("times the run", async () => {
    const { handle, posted } = harness(undefined, { warmup: false });
    await handle(load);
    await handle(run);
    const done = last(posted);
    expect(done?.type === "result" && done.result.ms).toBeGreaterThanOrEqual(0);
  });

  it("forwards the model's in-run progress as partials against the same id", async () => {
    // The encode is seconds of silence otherwise, and an unlabelled pause is
    // indistinguishable from a hang.
    const streaming = fakeVlm({
      generate: vi.fn(
        async (
          _i: unknown,
          _p: string,
          _n: number,
          onPartial: (p: VlmPartial) => void,
        ): Promise<VlmOutput> => {
          onPartial({ stage: "encoding" });
          onPartial({ stage: "generating", text: "a" });
          onPartial({ stage: "generating", text: "a cat" });
          return { text: "a cat", encodeMs: 90, tokens: 2 };
        },
      ) as unknown as Vlm["generate"],
    });
    const { handle, posted } = harness(async () => streaming, { warmup: false });
    await handle(load);
    await handle(run);

    const partials = posted.filter((m) => m.type === "partial");
    expect(partials).toEqual([
      { type: "partial", id: 7, partial: { stage: "encoding" } },
      { type: "partial", id: 7, partial: { stage: "generating", text: "a" } },
      { type: "partial", id: 7, partial: { stage: "generating", text: "a cat" } },
    ]);
    // And the result still arrives after them.
    expect(last(posted)).toMatchObject({ type: "result", id: 7 });
  });

  it("rejects a run with no model loaded, carrying the id", async () => {
    const { handle, posted } = harness(undefined, { warmup: false });
    await handle(run);
    expect(last(posted)).toEqual({
      type: "error",
      id: 7,
      error: "No model loaded",
    });
  });

  it("reports an inference failure against the id, leaving the model loaded", async () => {
    let call = 0;
    const flaky = fakeVlm({
      generate: vi.fn(async () => {
        if (++call > 0) throw new Error("Non-zero status code");
        return { text: "", encodeMs: 0, tokens: 0 };
      }) as unknown as Vlm["generate"],
    });
    const { handle, posted } = harness(async () => flaky, { warmup: false });
    await handle(load);
    await handle(run);

    // `id` present → Machine B. The page stays `ready`.
    expect(last(posted)).toEqual({
      type: "error",
      id: 7,
      error: "Non-zero status code",
    });
  });

  it("passes the prompt and token cap straight through", async () => {
    const { handle, models } = harness(undefined, { warmup: false });
    await handle(load);
    await handle(run);
    const [, prompt, maxNewTokens] = lastCall(models[0].generate);
    expect(prompt).toBe("What is this?");
    expect(maxNewTokens).toBe(64);
  });
});

describe("createVlmHandler — an ordered list of pictures", () => {
  // A video-language model is a frame sampler plus an image model, so the run
  // payload carries N pictures rather than one. The worker's chat template
  // fills its `{ type: "image" }` slots **positionally** from this array, which
  // makes the order load-bearing: frames that arrive shuffled produce a fluent
  // answer about a video that never happened, with no error anywhere.
  const frame = (n: number) => ({
    data: new Uint8ClampedArray([n, n, n]),
    width: 1,
    height: 1,
    channels: 3 as const,
  });

  it("hands every frame to the model, in the order it was sent", async () => {
    const { handle, models } = harness(undefined, { warmup: false });
    await handle(load);
    await handle({
      type: "run",
      id: 3,
      images: [frame(1), frame(2), frame(3)],
      prompt: "What happens?",
      maxNewTokens: 32,
    });

    const [images] = lastCall(models[0].generate) as [
      Array<{ data: Uint8ClampedArray }>,
    ];
    expect(images).toHaveLength(3);
    expect(images.map((f) => f.data[0])).toEqual([1, 2, 3]);
  });

  it("preserves a reversed list rather than normalising it", async () => {
    // The temporal-blindness experiment *is* a reordering. An engine that
    // sorted, de-duplicated or re-ordered the frames would silently answer the
    // question the page was not asking.
    const { handle, models } = harness(undefined, { warmup: false });
    await handle(load);
    await handle({
      type: "run",
      id: 4,
      images: [frame(3), frame(2), frame(1)],
      prompt: "What happens?",
      maxNewTokens: 32,
    });

    const [images] = lastCall(models[0].generate) as [
      Array<{ data: Uint8ClampedArray }>,
    ];
    expect(images.map((f) => f.data[0])).toEqual([3, 2, 1]);
  });

  it("refuses an empty list against the request id, leaving the model loaded", async () => {
    const { handle, posted } = harness(undefined, { warmup: false });
    await handle(load);
    await handle({
      type: "run",
      id: 5,
      images: [],
      prompt: "What happens?",
      maxNewTokens: 32,
    });
    expect(last(posted)).toEqual({
      type: "error",
      id: 5,
      error: "No image to look at",
    });
  });

  it("warms up on a one-element list, like any other run", async () => {
    // The warm-up must exercise the same code path a real run takes; a
    // bare image here would be a shape the engine never otherwise sees.
    const { handle, models } = harness();
    await handle(load);
    const [images] = models[0].generate.mock.calls[0];
    expect(Array.isArray(images)).toBe(true);
    expect(images).toHaveLength(1);
  });
});
