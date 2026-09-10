import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { GalleryImage } from "@/vision/gallery";

const post = vi.fn();
const pipeState = {
  status: "ready" as const,
  idle: false,
  loading: false,
  ready: true,
  progress: null,
  loadProgress: null,
  loadedInMs: 120,
  backend: "webgpu",
  running: false,
  error: null,
  run: post,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
};
const useVisionPipeline = vi.fn(() => pipeState);
vi.mock("@/hooks/useVisionPipeline", () => ({
  useVisionPipeline: (...a: unknown[]) => useVisionPipeline(...(a as [])),
}));

const fakeImage = { width: 8, height: 8, channels: 3, data: [] } as never;
const fromUrl = vi.fn().mockResolvedValue(fakeImage);
const downscale = vi.fn(async (img: unknown) => img);
vi.mock("@/vision/image", () => ({
  fromUrl: (...a: unknown[]) => fromUrl(...a),
  downscale: (...a: unknown[]) => downscale(...(a as [never])),
}));

const { useImageFeatures } = await import("@/hooks/useImageFeatures");

/** `[1, 3, 2]`: CLS then two patch rows. */
const TENSOR = {
  data: Float32Array.from([1, 0, 0, 1, 0, 1]),
  dims: [1, 3, 2],
};

const gallery: GalleryImage[] = [
  { id: "a", label: "A", url: "https://example.test/a.jpg", group: "animal" },
  { id: "b", label: "B", url: "https://example.test/b.jpg", group: "city" },
];

beforeEach(() => {
  vi.clearAllMocks();
  post.mockResolvedValue(TENSOR);
  fromUrl.mockResolvedValue(fakeImage);
});

describe("useImageFeatures", () => {
  it("loads the default model, and never on mount", () => {
    renderHook(() => useImageFeatures());
    expect(useVisionPipeline).toHaveBeenCalledWith(
      "image-feature-extraction",
      "Xenova/dinov2-small",
      false,
      undefined,
    );
  });

  it("asks the pipeline for the token rows, not a pooled vector", async () => {
    // `pool: true` would throw the patch rows away and with them the control
    // this page exists for.
    const { result } = renderHook(() => useImageFeatures());
    await act(async () => {
      await result.current.run(fakeImage);
    });
    expect(post).toHaveBeenCalledWith(fakeImage, undefined, undefined);
  });

  it("derives both poolings from a single forward pass", async () => {
    const { result } = renderHook(() => useImageFeatures());
    let embedding!: Awaited<ReturnType<typeof result.current.run>>;
    await act(async () => {
      embedding = await result.current.run(fakeImage);
    });
    expect(Object.keys(embedding.vectors).sort()).toEqual(["cls", "mean"]);
    expect(post).toHaveBeenCalledTimes(1);
  });

  it("embeds the gallery one picture at a time, in order", async () => {
    // Never a Promise.all: two overlapping calls into one ONNX session is not a
    // guarantee worth relying on, and a fan-out makes the progress count lie.
    const order: string[] = [];
    fromUrl.mockImplementation(async (url: string) => {
      order.push(url);
      return fakeImage;
    });

    const { result } = renderHook(() => useImageFeatures());
    await act(async () => {
      await result.current.buildIndex(gallery);
    });

    expect(order).toEqual([gallery[0].url, gallery[1].url]);
    await waitFor(() => expect(result.current.index).toHaveLength(2));
    expect(result.current.index.map((e) => e.image.id)).toEqual(["a", "b"]);
    expect(result.current.indexing).toBeNull();
  });

  it("skips a picture that fails to decode instead of abandoning the gallery", async () => {
    // Eleven neighbours are still a working page, and the missing one is
    // visibly absent.
    fromUrl.mockRejectedValueOnce(new Error("404"));
    const { result } = renderHook(() => useImageFeatures());
    await act(async () => {
      await result.current.buildIndex(gallery);
    });
    await waitFor(() => expect(result.current.index).toHaveLength(1));
    expect(result.current.index[0].image.id).toBe("b");
  });

  it("drops the index on request, because vectors from another checkpoint are not comparable", async () => {
    const { result } = renderHook(() => useImageFeatures());
    await act(async () => {
      await result.current.buildIndex(gallery);
    });
    await waitFor(() => expect(result.current.index).toHaveLength(2));

    act(() => result.current.clearIndex());
    expect(result.current.index).toEqual([]);
  });

  it("does not let a superseded build write into the new index", async () => {
    // Switching models mid-build must not leave the old checkpoint's vectors
    // in an index labelled with the new one.
    let release: ((v: unknown) => void) | null = null;
    fromUrl.mockImplementationOnce(
      () => new Promise((resolve) => (release = resolve)),
    );

    const { result } = renderHook(() => useImageFeatures());
    let build!: Promise<void>;
    act(() => {
      build = result.current.buildIndex(gallery);
    });
    act(() => result.current.clearIndex());
    await act(async () => {
      release?.(fakeImage);
      await build;
    });

    expect(result.current.index).toEqual([]);
  });

  it("adds one decoded picture to the index, replacing a repeat id", async () => {
    const { result } = renderHook(() => useImageFeatures());
    await act(async () => {
      await result.current.addToIndex(gallery[0], fakeImage);
      await result.current.addToIndex(gallery[0], fakeImage);
    });
    await waitFor(() => expect(result.current.index).toHaveLength(1));
  });
});
