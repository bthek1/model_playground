import { beforeEach, describe, expect, it, vi } from "vitest";

import { TEXT_CACHE_LIMIT } from "../zeroShot";
import { createZeroShotHandler, promptsKey, type Embeddings } from "./engine";
import type { ZeroShotResponse } from "./types";

vi.mock("@/model/backend", () => ({
  pickBackend: vi.fn().mockResolvedValue("wasm"),
  loadOpts: () => ({ device: "wasm", dtype: "q8" }),
}));

const image = {
  data: new Uint8ClampedArray(12),
  width: 2,
  height: 2,
  channels: 3,
} as const;

/** One 2-D embedding per prompt, distinct per prompt so a mix-up is visible. */
function textEmbeds(prompts: string[]): Embeddings {
  const data = prompts.flatMap((p) => (p.includes("cat") ? [1, 0] : [0, 1]));
  return { data, rows: prompts.length, dim: 2 };
}

function harness({ warmup = false } = {}) {
  const posted: ZeroShotResponse[] = [];
  const encodeText = vi.fn(async (prompts: string[]) => textEmbeds(prompts));
  const encodeImage = vi.fn(async () => ({ data: [1, 0], rows: 1, dim: 2 }));
  const dispose = vi.fn(async () => {});
  const factory = vi.fn(async () => ({ encodeText, encodeImage, dispose }));

  const handle = createZeroShotHandler(
    (m) => posted.push(m),
    factory,
    { warmup },
  );
  return { handle, posted, encodeText, encodeImage, dispose, factory };
}

const LOAD = {
  type: "load" as const,
  model: "Xenova/clip-vit-base-patch32",
  family: "clip" as const,
  scoring: { kind: "softmax" as const, scale: 100 },
};

const run = (id: number, prompts: string[]) => ({
  type: "run" as const,
  id,
  image,
  prompts,
});

const results = (posted: ZeroShotResponse[]) =>
  posted.filter((m) => m.type === "result");

/** `Array.prototype.at` is past this project's lib target. */
const last = (posted: ZeroShotResponse[]) => posted[posted.length - 1];

describe("createZeroShotHandler", () => {
  beforeEach(() => vi.clearAllMocks());

  it("reports ready with the backend it resolved", async () => {
    const { handle, posted } = harness();
    await handle(LOAD);
    expect(last(posted)).toMatchObject({ type: "ready", backend: "wasm" });
  });

  it("warms up both towers, and a warm-up failure never fails the load", async () => {
    // Both towers compile shaders, so the first real run should pay for neither.
    const { handle, posted, encodeText, encodeImage } = harness({ warmup: true });
    encodeText.mockRejectedValueOnce(new Error("compile blew up"));
    await handle(LOAD);

    expect(posted.some((m) => m.type === "progress" && m.progress?.status === "warmup")).toBe(true);
    expect(last(posted)).toMatchObject({ type: "ready" });
    expect(encodeImage).not.toHaveBeenCalled(); // the text tower threw first
  });

  it("encodes the labels once and reuses them for the next image", async () => {
    // The entire reason this engine exists: on a live feed the label embeddings
    // are constant, and re-encoding them per frame is ~40% of the work thrown
    // away and redone.
    const { handle, posted, encodeText, encodeImage } = harness();
    await handle(LOAD);

    await handle(run(1, ["a cat", "a dog"]));
    await handle(run(2, ["a cat", "a dog"]));
    await handle(run(3, ["a cat", "a dog"]));

    expect(encodeText).toHaveBeenCalledTimes(1);
    expect(encodeImage).toHaveBeenCalledTimes(3);

    const [first, second] = results(posted);
    expect(first.result?.textCached).toBe(false);
    expect(second.result?.textCached).toBe(true);
    expect(second.result?.textMs).toBe(0);
  });

  it("re-encodes when the labels change", async () => {
    const { handle, encodeText } = harness();
    await handle(LOAD);
    await handle(run(1, ["a cat"]));
    await handle(run(2, ["a bicycle"]));
    expect(encodeText).toHaveBeenCalledTimes(2);
  });

  it("re-encodes when only the order changes, because the order is the answer", () => {
    // Scores come back positionally, so ["cat","dog"] and ["dog","cat"] are
    // different questions. A key that ignored order would label them backwards.
    expect(promptsKey(["a", "b"])).not.toBe(promptsKey(["b", "a"]));
  });

  it("keeps both templates live, so the comparison view does not thrash", async () => {
    // The page alternates two prompt sets on the same image. A single-entry
    // cache would evict on every alternation and buy nothing on exactly the
    // screen this feature exists for.
    const { handle, encodeText } = harness();
    await handle(LOAD);

    await handle(run(1, ["cat", "dog"]));
    await handle(run(2, ["a photo of a cat", "a photo of a dog"]));
    expect(encodeText).toHaveBeenCalledTimes(2);

    // A second image: both sets are still cached.
    await handle(run(3, ["cat", "dog"]));
    await handle(run(4, ["a photo of a cat", "a photo of a dog"]));
    expect(encodeText).toHaveBeenCalledTimes(2);
  });

  it("evicts the least recently used set once it is full", async () => {
    const { handle, encodeText } = harness();
    await handle(LOAD);

    for (let i = 0; i < TEXT_CACHE_LIMIT; i++) {
      await handle(run(i, [`label ${i}`]));
    }
    expect(encodeText).toHaveBeenCalledTimes(TEXT_CACHE_LIMIT);

    // Touch the oldest so it becomes the newest, then overflow by one.
    await handle(run(100, ["label 0"]));
    expect(encodeText).toHaveBeenCalledTimes(TEXT_CACHE_LIMIT); // still cached
    await handle(run(101, ["brand new"]));

    // "label 0" was refreshed, so "label 1" is the one that went.
    await handle(run(102, ["label 0"]));
    expect(encodeText).toHaveBeenCalledTimes(TEXT_CACHE_LIMIT + 1);
    await handle(run(103, ["label 1"]));
    expect(encodeText).toHaveBeenCalledTimes(TEXT_CACHE_LIMIT + 2);
  });

  it("scores the labels in the order they were asked for", async () => {
    const { handle, posted } = harness();
    await handle(LOAD);
    await handle(run(1, ["a dog", "a cat"]));

    // The image embedding is [1,0]; "a cat" maps to [1,0] and "a dog" to [0,1].
    const scores = results(posted)[0].result!.scores;
    expect(scores[1]).toBeGreaterThan(scores[0]);
  });

  it("drops the cache when a different checkpoint is loaded", async () => {
    // Embeddings from another model share no space with these ones; reusing
    // them would produce confident, meaningless scores.
    const { handle, encodeText } = harness();
    await handle(LOAD);
    await handle(run(1, ["a cat"]));
    expect(encodeText).toHaveBeenCalledTimes(1);

    await handle({ ...LOAD, model: "Xenova/siglip-base-patch16-224" });
    await handle(run(2, ["a cat"]));
    expect(encodeText).toHaveBeenCalledTimes(2);
  });

  it("frees the previous towers before loading the next", async () => {
    const { handle, dispose } = harness();
    await handle(LOAD);
    await handle({ ...LOAD, model: "Xenova/siglip-base-patch16-224" });
    expect(dispose).toHaveBeenCalledTimes(1);
  });

  it("survives a teardown that throws, without leaving a stale model live", async () => {
    const { handle, posted, dispose } = harness();
    await handle(LOAD);
    dispose.mockRejectedValueOnce(new Error("backend gone"));
    await handle(LOAD);
    expect(last(posted)).toMatchObject({ type: "ready" });
  });

  it("refuses to run before a model is loaded", async () => {
    const { handle, posted } = harness();
    await handle(run(1, ["a cat"]));
    expect(last(posted)).toMatchObject({ type: "error", id: 1 });
  });

  it("refuses an empty label set rather than scoring nothing", async () => {
    const { handle, posted } = harness();
    await handle(LOAD);
    await handle(run(1, []));
    expect(last(posted)).toMatchObject({ type: "error", id: 1 });
  });

  it("reports a load failure without an id, so it lands in the LOAD slot", async () => {
    const { handle, posted, factory } = harness();
    factory.mockRejectedValueOnce(new Error("404 no such repo"));
    await handle(LOAD);
    const failure = last(posted);
    expect(failure.type).toBe("error");
    // No `id`: a load failure belongs to the LOAD slot, not to a request.
    expect("id" in failure && failure.id).toBeFalsy();
  });

  it("keeps an inference failure correlated to its request", async () => {
    const { handle, posted, encodeImage } = harness();
    await handle(LOAD);
    encodeImage.mockRejectedValueOnce(new Error("session failed"));
    await handle(run(7, ["a cat"]));
    expect(last(posted)).toMatchObject({ type: "error", id: 7 });
  });
});
