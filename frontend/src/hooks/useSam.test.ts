import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { SamPoint, SamResult } from "@/vision/sam/types";

const post = vi.fn();
const workerState = {
  status: "ready" as const,
  idle: false,
  loading: false,
  ready: true,
  progress: null,
  loadProgress: null,
  loadedInMs: 90,
  backend: "webgpu",
  running: false,
  result: null,
  error: null,
  load: vi.fn(),
  retry: vi.fn(),
  cancel: vi.fn(),
  run: post,
};
const useModelWorker = vi.fn(() => workerState);
vi.mock("@/model/useModelWorker", () => ({
  useModelWorker: (...a: unknown[]) => useModelWorker(...(a as [])),
}));
vi.mock("@/vision/sam/client", () => ({ createSamWorker: vi.fn() }));

const { useSam } = await import("@/hooks/useSam");

const image = {
  width: 8,
  height: 8,
  channels: 3,
  data: new Uint8ClampedArray(8 * 8 * 3),
} as never;

const encodeReply: SamResult = {
  kind: "encode",
  cached: false,
  ms: 420,
  width: 8,
  height: 8,
};
const decodeReply = (ms = 12): SamResult => ({
  kind: "decode",
  ms,
  masks: [
    { data: new Uint8Array(4), width: 2, height: 2, score: 0.9 },
    { data: new Uint8Array(4), width: 2, height: 2, score: 0.4 },
  ],
});

beforeEach(() => {
  vi.clearAllMocks();
  post.mockImplementation(async (payload: { kind: string }) =>
    payload.kind === "encode" ? encodeReply : decodeReply(),
  );
});

describe("useSam", () => {
  it("keys the worker on the model, and never loads on mount", () => {
    renderHook(() => useSam());
    expect(useModelWorker).toHaveBeenCalledWith(
      expect.objectContaining({
        key: "sam:Xenova/slimsam-77-uniform",
        autoLoad: false,
        loadMessage: { model: "Xenova/slimsam-77-uniform" },
      }),
    );
  });

  it("reports encoding as its own state, distinct from loading", async () => {
    // LOAD is the download; encoding is the per-image encoder pass, and it is
    // the slow half of the first click. Collapsing them tells the user nothing.
    let release: ((r: SamResult) => void) | null = null;
    post.mockImplementationOnce(
      () => new Promise((resolve) => (release = resolve)),
    );

    const { result } = renderHook(() => useSam());
    expect(result.current.encoding).toBe(false);
    expect(result.current.encoded).toBe(false);

    let done!: Promise<void>;
    act(() => {
      done = result.current.encode("a", image);
    });
    await waitFor(() => expect(result.current.encoding).toBe(true));
    expect(result.current.loading).toBe(false);

    await act(async () => {
      release?.(encodeReply);
      await done;
    });
    expect(result.current.encoding).toBe(false);
    expect(result.current.encoded).toBe(true);
    expect(result.current.encodedInMs).toBe(420);
  });

  it("reports no encode time when the embedding was reused", async () => {
    post.mockResolvedValueOnce({ ...encodeReply, cached: true, ms: 0 });
    const { result } = renderHook(() => useSam());
    await act(async () => {
      await result.current.encode("a", image);
    });
    expect(result.current.encoded).toBe(true);
    expect(result.current.encodedInMs).toBeNull();
  });

  it("transfers the image buffer rather than copying it", async () => {
    // The page keeps its own RawImage; the payload is built for this post.
    const { result } = renderHook(() => useSam());
    await act(async () => {
      await result.current.encode("a", image);
    });
    const [payload, transfer] = post.mock.calls[0];
    expect(payload).toMatchObject({ kind: "encode", token: "a" });
    expect(transfer).toHaveLength(1);
  });

  it("drops the previous masks the moment a new encode starts", async () => {
    // Otherwise the OUTPUT slot shows one image's mask over another's pixels.
    const { result } = renderHook(() => useSam());
    await act(async () => {
      await result.current.encode("a", image);
      await result.current.run([{ x: 1, y: 1, positive: true }]);
    });
    await waitFor(() => expect(result.current.result).not.toBeNull());

    await act(async () => {
      await result.current.encode("b", image);
    });
    expect(result.current.result).toBeNull();
  });

  it("decodes clicks and keeps the candidates, best first", async () => {
    const { result } = renderHook(() => useSam());
    await act(async () => {
      await result.current.encode("a", image);
      await result.current.run([{ x: 3, y: 3, positive: true }]);
    });
    await waitFor(() => expect(result.current.result?.masks).toHaveLength(2));
    expect(result.current.result?.ms).toBe(12);
    expect(post).toHaveBeenLastCalledWith({
      kind: "decode",
      points: [{ x: 3, y: 3, positive: true }],
    });
  });

  it("queues no deeper than one: a click mid-decode replaces the pending one", async () => {
    // The never-queue rule `useLiveFrames` follows, for the same reason: only
    // the latest click's answer is still wanted, and a backlog turns a fast
    // model into a laggy one.
    const decodes: SamPoint[][] = [];
    let release: ((r: SamResult) => void) | null = null;
    post.mockImplementation(async (payload: { kind: string; points?: SamPoint[] }) => {
      if (payload.kind === "encode") return encodeReply;
      decodes.push(payload.points!);
      if (decodes.length === 1) {
        return new Promise<SamResult>((resolve) => (release = resolve));
      }
      return decodeReply();
    });

    const { result } = renderHook(() => useSam());
    await act(async () => {
      await result.current.encode("a", image);
    });

    let first!: Promise<unknown>;
    act(() => {
      first = result.current.run([{ x: 1, y: 1, positive: true }]);
    });
    // Two more clicks while the first decode is in flight. The second is
    // superseded by the third rather than both being run.
    act(() => {
      void result.current.run([{ x: 2, y: 2, positive: true }]);
      void result.current.run([{ x: 3, y: 3, positive: true }]);
    });

    await act(async () => {
      release?.(decodeReply());
      await first;
    });

    await waitFor(() => expect(decodes).toHaveLength(2));
    expect(decodes[0]).toEqual([{ x: 1, y: 1, positive: true }]);
    expect(decodes[1]).toEqual([{ x: 3, y: 3, positive: true }]);
  });

  it("clears the masks and the encoded flag on reset", async () => {
    const { result } = renderHook(() => useSam());
    await act(async () => {
      await result.current.encode("a", image);
      await result.current.run([{ x: 1, y: 1, positive: true }]);
    });
    await waitFor(() => expect(result.current.result).not.toBeNull());

    act(() => result.current.reset());
    expect(result.current.result).toBeNull();
    expect(result.current.encoded).toBe(false);
    expect(result.current.encodedInMs).toBeNull();
  });

  it("stops encoding even when the encode fails", async () => {
    post.mockRejectedValueOnce(new Error("out of memory"));
    const { result } = renderHook(() => useSam());
    await act(async () => {
      await expect(result.current.encode("a", image)).rejects.toThrow(
        /out of memory/,
      );
    });
    expect(result.current.encoding).toBe(false);
    expect(result.current.encoded).toBe(false);
  });
});
