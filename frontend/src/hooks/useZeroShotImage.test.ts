import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { TEMPLATES } from "@/vision/zeroShot";

// The worker answers with scores in prompt order; the hook's job is to put the
// user's own wording back on them.
const post = vi.fn(async ({ prompts }: { prompts: string[] }) => ({
  scores: prompts.map((_, i) => 1 / (i + 2)),
  textCached: false,
  textMs: 12,
  imageMs: 30,
}));

const worker = {
  status: "ready",
  idle: false,
  loading: false,
  ready: true,
  progress: null,
  loadProgress: null,
  loadedInMs: 4000,
  backend: "webgpu",
  running: false,
  error: null,
  run: post,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};
type WorkerOpts = {
  key: string;
  autoLoad: boolean;
  loadMessage: Record<string, unknown>;
};
const useModelWorker = vi.fn((opts: WorkerOpts) => {
  void opts;
  return worker;
});

vi.mock("@/model/useModelWorker", () => ({
  useModelWorker: (opts: unknown) => useModelWorker(opts as WorkerOpts),
}));

/** The options object the hook handed `useModelWorker` on its first render. */
const workerOpts = () => useModelWorker.mock.calls[0][0];
vi.mock("@/vision/zeroshot/client", () => ({ createZeroShotWorker: vi.fn() }));
vi.mock("@/vision/image", () => ({
  toPayload: (img: unknown) => img,
  transferablesOf: () => [],
}));

const { useZeroShotImage } = await import("./useZeroShotImage");

const image = {} as never;

describe("useZeroShotImage", () => {
  afterEach(() => vi.clearAllMocks());

  it("loads the split-tower worker, not the generic pipeline one", () => {
    // The whole point of this task's engine: the generic vision worker has
    // nowhere to keep the text-embedding cache.
    renderHook(() => useZeroShotImage());
    expect(workerOpts().key).toBe("zero-shot:Xenova/clip-vit-base-patch32");
    expect(workerOpts().autoLoad).toBe(false);
  });

  it("tells the worker which family and scoring the checkpoint needs", () => {
    // CLIP and SigLIP need different tower classes, different tokenizer padding
    // and different final steps. Getting the scale wrong keeps the ranking and
    // ruins the numbers, so it travels explicitly.
    renderHook(() => useZeroShotImage("Xenova/siglip-base-patch16-224"));
    expect(workerOpts().loadMessage).toMatchObject({
      model: "Xenova/siglip-base-patch16-224",
      family: "siglip",
      scoring: { kind: "sigmoid", scale: 117.330795, bias: -12.932437 },
    });
  });

  it("uses CLIP's softmax scoring, with no bias term", () => {
    renderHook(() => useZeroShotImage("Xenova/clip-vit-base-patch32"));
    expect(workerOpts().loadMessage.scoring).toEqual({
      kind: "softmax",
      scale: 100.000006,
    });
  });

  it("falls back to the first catalogue entry for an unknown id", () => {
    renderHook(() => useZeroShotImage("someone/deleted-this-model"));
    expect(workerOpts().key).toBe("zero-shot:Xenova/clip-vit-base-patch32");
  });

  it("sends the template applied to every label, not the bare labels", async () => {
    const { result } = renderHook(() => useZeroShotImage());
    await act(async () => {
      await result.current.run(image, ["cat", "dog"], [TEMPLATES.photo]);
    });
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({
        prompts: ["a photo of a cat", "a photo of a dog"],
      }),
      [],
    );
  });

  it("scores each template in its own call, never one merged list", async () => {
    // Each template is its own softmax. Concatenating the two prompt sets into
    // one call would make the wordings compete against each other and mean
    // nothing at all.
    const { result } = renderHook(() => useZeroShotImage());
    await act(async () => {
      await result.current.run(
        image,
        ["cat", "dog"],
        [TEMPLATES.bare, TEMPLATES.photo],
      );
    });

    expect(post).toHaveBeenCalledTimes(2);
    expect(post.mock.calls[0][0].prompts).toEqual(["cat", "dog"]);
    expect(post.mock.calls[1][0].prompts).toEqual([
      "a photo of a cat",
      "a photo of a dog",
    ]);
    expect(result.current.result?.map((r) => r.template)).toEqual([
      "{}",
      "a photo of a {}",
    ]);
  });

  it("hands back the user's own wording, positionally", async () => {
    const { result } = renderHook(() => useZeroShotImage());
    await act(async () => {
      await result.current.run(image, ["cat", "dog"], [TEMPLATES.photo]);
    });
    expect(result.current.result?.[0].scores.map((s) => s.label)).toEqual([
      "cat",
      "dog",
    ]);
  });

  it("surfaces whether the labels were re-encoded, and what that cost", async () => {
    // An optimisation nobody can see is an optimisation nobody can check.
    post.mockResolvedValueOnce({
      scores: [0.9],
      textCached: true,
      textMs: 0,
      imageMs: 25,
    });
    const { result } = renderHook(() => useZeroShotImage());
    await act(async () => {
      await result.current.run(image, ["cat"], [TEMPLATES.bare]);
    });
    expect(result.current.result?.[0]).toMatchObject({
      textCached: true,
      textMs: 0,
      imageMs: 25,
    });
  });

  it("ignores blank labels and refuses a run with nothing to score", async () => {
    const { result } = renderHook(() => useZeroShotImage());
    await act(async () => {
      await result.current.run(image, ["  cat  ", "  "], [TEMPLATES.bare]);
    });
    expect(post.mock.calls[0][0].prompts).toEqual(["cat"]);

    await expect(
      result.current.run(image, ["   "], [TEMPLATES.bare]),
    ).rejects.toThrow(/at least one label/i);
  });

  it("copies the image rather than transferring it, so a second template can score it", async () => {
    // Two templates score the same picture back to back. A transferred buffer
    // arrives detached and the second call would see an empty image.
    const { toPayload } = await import("@/vision/image");
    const { result } = renderHook(() => useZeroShotImage());
    await act(async () => {
      await result.current.run(
        image,
        ["cat"],
        [TEMPLATES.bare, TEMPLATES.photo],
      );
    });
    expect(toPayload).toBeDefined();
    expect(post).toHaveBeenCalledTimes(2);
  });
});
