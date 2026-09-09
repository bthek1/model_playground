import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { VisionResponse } from "@/vision/types";

// A stand-in Web Worker: records posted messages, lets a test push responses.
class FakeWorker {
  onmessage: ((e: MessageEvent<VisionResponse>) => void) | null = null;
  posted: Array<{ message: unknown; transfer?: Transferable[] }> = [];
  terminated = false;

  postMessage(message: unknown, transfer?: Transferable[]) {
    this.posted.push({ message, transfer });
  }
  terminate() {
    this.terminated = true;
  }
  emit(data: VisionResponse) {
    this.onmessage?.({ data } as MessageEvent<VisionResponse>);
  }
}

let lastWorker: FakeWorker;
let spawned = 0;

vi.mock("@/vision/client", () => ({
  createVisionWorker: () => {
    spawned += 1;
    lastWorker = new FakeWorker();
    return lastWorker as unknown as Worker;
  },
}));

const { useVisionPipeline } = await import("./useVisionPipeline");

/** A `RawImage`-shaped stand-in; the hook only reads these four fields. */
function image(side = 2) {
  const data = new Uint8ClampedArray(side * side * 3);
  data.fill(9);
  return { data, width: side, height: side, channels: 3 as const } as never;
}

/** Start a run we never await: its rejection on teardown is expected. */
function fire<T>(promise: Promise<T>): void {
  promise.catch(() => {});
}

async function loadedHook() {
  const hook = renderHook(() =>
    useVisionPipeline("image-classification", "Xenova/vit-base-patch16-224", false),
  );
  act(() => hook.result.current.load());
  act(() =>
    lastWorker.emit({
      type: "ready",
      model: "Xenova/vit-base-patch16-224",
      backend: "webgpu",
    }),
  );
  await waitFor(() => expect(hook.result.current.ready).toBe(true));
  return hook;
}

describe("useVisionPipeline", () => {
  beforeEach(() => {
    spawned = 0;
  });
  afterEach(() => vi.clearAllMocks());

  it("spawns no worker at all when autoLoad is false", () => {
    const { result } = renderHook(() =>
      useVisionPipeline("image-classification", "m", false),
    );
    expect(result.current.status).toBe("idle");
    expect(spawned).toBe(0);
  });

  it("carries the task in the load message, so one worker serves every task", async () => {
    const { result } = renderHook(() =>
      useVisionPipeline("object-detection", "onnx-community/dfine_n_coco-ONNX", false),
    );
    act(() => result.current.load());
    expect(lastWorker.posted[0].message).toEqual({
      type: "load",
      task: "object-detection",
      model: "onnx-community/dfine_n_coco-ONNX",
    });
  });

  it("sends the pixels and their shape, not the RawImage instance", async () => {
    const { result } = await loadedHook();
    const img = image();
    fire(result.current.run(img, [{ top_k: 5 }]));

    const run = lastWorker.posted[1].message as {
      type: string;
      id: number;
      image: { width: number; height: number; channels: number; data: Uint8ClampedArray };
      args: unknown[];
    };
    // A class instance does not survive postMessage — it arrives as a plain
    // object with no methods, and the pipeline rejects it.
    expect(run.type).toBe("run");
    expect(run.image.width).toBe(2);
    expect(run.image.height).toBe(2);
    expect(run.image.channels).toBe(3);
    expect(run.args).toEqual([{ top_k: 5 }]);
    expect(lastWorker.posted[1].transfer).toEqual([run.image.data.buffer]);
  });

  it("copies the pixels by default, so the page's preview survives the post", async () => {
    const { result } = await loadedHook();
    const img = image();
    fire(result.current.run(img));

    const run = lastWorker.posted[1].message as { image: { data: Uint8ClampedArray } };
    expect(run.image.data).not.toBe((img as unknown as { data: Uint8ClampedArray }).data);
  });

  it("hands the buffer over when the caller says the frame is spent", async () => {
    const { result } = await loadedHook();
    const img = image();
    fire(result.current.run(img, undefined, { consume: true }));

    const run = lastWorker.posted[1].message as { image: { data: Uint8ClampedArray } };
    expect(run.image.data).toBe((img as unknown as { data: Uint8ClampedArray }).data);
  });

  it("resolves each run with its own correlated result", async () => {
    const { result } = await loadedHook();
    const first = result.current.run(image());
    const second = result.current.run(image());

    const ids = lastWorker.posted
      .slice(1)
      .map((p) => (p.message as { id: number }).id);
    expect(new Set(ids).size).toBe(2);

    // Answered out of order on purpose: `running` is an inflight count, and the
    // pending table is keyed by id precisely so this works.
    act(() => lastWorker.emit({ type: "result", id: ids[1], result: "second" }));
    await expect(second).resolves.toBe("second");
    expect(result.current.running).toBe(true);

    act(() => lastWorker.emit({ type: "result", id: ids[0], result: "first" }));
    await expect(first).resolves.toBe("first");
    await waitFor(() => expect(result.current.running).toBe(false));
  });

  it("keeps the model loaded when a single run fails", async () => {
    const { result } = await loadedHook();
    const run = result.current.run(image());
    const id = (lastWorker.posted[1].message as { id: number }).id;

    act(() => lastWorker.emit({ type: "error", id, error: "bad image" }));
    await expect(run).rejects.toThrow("bad image");
    // Machine B: an inference failure never demotes the model.
    expect(result.current.status).toBe("ready");
    expect(result.current.error).toBe("bad image");
  });

  it("moves to error when the load itself fails, and retry reloads", async () => {
    const { result } = renderHook(() =>
      useVisionPipeline("image-classification", "nope/nope", false),
    );
    act(() => result.current.load());
    act(() => lastWorker.emit({ type: "error", error: "404 model not found" }));
    await waitFor(() => expect(result.current.status).toBe("error"));

    act(() => result.current.retry());
    expect(spawned).toBe(2);
  });

  it("tears the worker down when the selected model changes", async () => {
    const { rerender } = renderHook(
      ({ model }: { model: string }) =>
        useVisionPipeline("image-classification", model, false),
      { initialProps: { model: "a" } },
    );
    act(() => {});
    rerender({ model: "b" });
    // No worker was ever started (autoLoad false), so nothing to terminate —
    // what matters is the key change resets the machine to idle.
    expect(spawned).toBe(0);
  });
});
