// `useGrounding` — `/robotics`'s pair, in **two** workers.
//
// These run the real `useZeroShotDetector` / `useDepth` / `useVisionPipeline` /
// `useModelWorker` stack over a fake Worker, rather than mocking the two task
// hooks away, because the claim under test is structural: that the pair is
// two independent workers. A mock of the hooks would assert the fan-out and
// nothing about the teardown — and the teardown is the claim the plan got
// wrong (it asked for `/pose`'s `Promise.allSettled`, which is for one worker
// owning two models).

import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { VisionResponse } from "@/vision/types";

class FakeWorker {
  onmessage: ((e: MessageEvent<VisionResponse>) => void) | null = null;
  posted: Array<{ message: Record<string, unknown>; transfer?: Transferable[] }> = [];
  terminated = false;
  throwOnTerminate = false;

  postMessage(message: Record<string, unknown>, transfer?: Transferable[]) {
    this.posted.push({ message, transfer });
    // The task arrives in the load message; index the worker by it so a test
    // can address "the detector" and "the depth model" by name.
    if (message.type === "load") byTask.set(message.task as string, this);
  }
  terminate() {
    this.terminated = true;
    if (this.throwOnTerminate) throw new Error("dispose failed");
  }
  emit(data: VisionResponse) {
    this.onmessage?.({ data } as MessageEvent<VisionResponse>);
  }
  lastRun() {
    return [...this.posted].reverse().find((p) => p.message.type === "run")!;
  }
}

const byTask = new Map<string, FakeWorker>();
let spawned: FakeWorker[] = [];

vi.mock("@/vision/client", () => ({
  createVisionWorker: () => {
    const w = new FakeWorker();
    spawned.push(w);
    return w as unknown as Worker;
  },
}));

vi.mock("@/model/backend", async (orig) => ({
  ...(await orig<Record<string, unknown>>()),
  pickBackend: vi.fn(async () => "webgpu"),
}));

const { useGrounding } = await import("./useGrounding");
const { GROUNDING_PAIR } = await import("@/vision/grounding");

const DETECT = "zero-shot-object-detection";
const DEPTH = "depth-estimation";
const det = () => byTask.get(DETECT)!;
const dep = () => byTask.get(DEPTH)!;

function image(side = 4) {
  const data = new Uint8ClampedArray(side * side * 3).fill(7);
  return { data, width: side, height: side, channels: 3 as const } as never;
}

async function loaded() {
  const hook = renderHook(() => useGrounding());
  await act(async () => {}); // let the backend probe settle
  act(() => hook.result.current.load());
  act(() => {
    det().emit({ type: "ready", model: GROUNDING_PAIR.detector.id, backend: "webgpu" });
    dep().emit({ type: "ready", model: GROUNDING_PAIR.depth.id, backend: "webgpu" });
  });
  await waitFor(() => expect(hook.result.current.ready).toBe(true));
  return hook;
}

beforeEach(() => {
  byTask.clear();
  spawned = [];
});
afterEach(() => vi.clearAllMocks());

describe("useGrounding", () => {
  it("downloads nothing on mount", () => {
    const hook = renderHook(() => useGrounding());
    expect(hook.result.current.status).toBe("idle");
    expect(spawned).toHaveLength(0);
  });

  it("fans one LOAD out to two workers, each with its own checkpoint", () => {
    const hook = renderHook(() => useGrounding());
    act(() => hook.result.current.load());

    expect(spawned).toHaveLength(2);
    expect(det().posted[0].message).toMatchObject({
      type: "load",
      task: DETECT,
      model: GROUNDING_PAIR.detector.id,
    });
    expect(dep().posted[0].message).toMatchObject({
      type: "load",
      task: DEPTH,
      model: GROUNDING_PAIR.depth.id,
    });
    expect(hook.result.current.status).toBe("loading");
  });

  it("is ready only when both halves are", async () => {
    const hook = renderHook(() => useGrounding());
    act(() => hook.result.current.load());

    act(() =>
      dep().emit({ type: "ready", model: GROUNDING_PAIR.depth.id, backend: "webgpu" }),
    );
    // The small half finishing first is the common case, and must not unlock
    // the GENERATE trigger.
    expect(hook.result.current.ready).toBe(false);
    expect(hook.result.current.status).toBe("loading");

    act(() =>
      det().emit({ type: "ready", model: GROUNDING_PAIR.detector.id, backend: "webgpu" }),
    );
    await waitFor(() => expect(hook.result.current.ready).toBe(true));
  });

  it("reports one bar against the pair's combined size, not either half's", async () => {
    const hook = renderHook(() => useGrounding());
    await act(async () => {});
    act(() => hook.result.current.load());

    // The depth model finishes its whole download; the detector has not begun.
    const depthBytes = GROUNDING_PAIR.depth.bytes!.webgpu!;
    act(() =>
      dep().emit({
        type: "progress",
        progress: {
          status: "progress",
          name: GROUNDING_PAIR.depth.id,
          file: "onnx/model_fp16.onnx",
          loaded: depthBytes,
          total: depthBytes,
        } as never,
      }),
    );

    const bar = hook.result.current.loadProgress!;
    expect(bar.total).toBe(GROUNDING_PAIR.bytes.webgpu);
    expect(bar.percent).toBeLessThan(20);
    expect(bar.percent).toBeGreaterThan(0);
  });

  it("surfaces either half's load failure, and retries only that half", async () => {
    const hook = renderHook(() => useGrounding());
    act(() => hook.result.current.load());
    act(() =>
      dep().emit({ type: "ready", model: GROUNDING_PAIR.depth.id, backend: "webgpu" }),
    );
    const depthWorker = dep();
    act(() => det().emit({ type: "error", error: "detector 404" }));

    expect(hook.result.current.status).toBe("error");
    expect(hook.result.current.error).toBe("detector 404");

    const before = spawned.length;
    act(() => hook.result.current.retry());
    // One new worker, for the detector. Re-downloading the depth model that is
    // already in memory would be the cost of a lazy retry.
    expect(spawned.length).toBe(before + 1);
    expect(depthWorker.terminated).toBe(false);
    expect(dep()).toBe(depthWorker);
  });

  it("cancels both downloads", () => {
    const hook = renderHook(() => useGrounding());
    act(() => hook.result.current.load());
    const [a, b] = spawned;
    act(() => hook.result.current.cancel());
    expect(a.terminated).toBe(true);
    expect(b.terminated).toBe(true);
    expect(hook.result.current.status).toBe("idle");
  });

  it("tears each worker down independently — one failing cannot keep the other alive", async () => {
    const hook = await loaded();
    const detector = det();
    const depth = dep();
    detector.throwOnTerminate = true;

    // React reports an error thrown from an effect cleanup; what matters is
    // what happened to the *other* worker, not how the first one's error
    // surfaced. No `allSettled` exists anywhere to make this true — the two
    // cleanups are separate effects in separate hooks.
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      hook.unmount();
    } catch {
      /* the detector's teardown threw, as arranged */
    }
    errors.mockRestore();

    expect(detector.terminated).toBe(true);
    expect(depth.terminated).toBe(true);
  });

  it("runs the detector and then the depth model on the same frame, copying for the first", async () => {
    const hook = await loaded();
    const img = image();

    let done: Promise<unknown> = Promise.resolve();
    act(() => {
      done = hook.result.current.run(img, [" a car ", ""], { consume: true });
    });

    // Sequential: the depth request is not posted until the detector answers.
    await waitFor(() => expect(det().lastRun()).toBeDefined());
    expect(dep().posted.some((p) => p.message.type === "run")).toBe(false);

    const detRun = det().lastRun();
    // The queries are cleaned, and the detector's pinned options travel.
    expect((detRun.message.args as unknown[])[0]).toEqual(["a car"]);
    expect((detRun.message.args as unknown[])[1]).toMatchObject({ percentage: false });
    // The detector gets a copy — the depth call still needs the pixels.
    expect((detRun.message.image as { data: unknown }).data).not.toBe(
      (img as { data: unknown }).data,
    );

    const boxes = [{ label: "a car", score: 0.4, box: { xmin: 0, ymin: 0, xmax: 2, ymax: 2 } }];
    act(() => det().emit({ type: "result", id: detRun.message.id as number, result: boxes }));

    await waitFor(() => expect(dep().lastRun()).toBeDefined());
    const depRun = dep().lastRun();
    // Only the last call may consume the caller's buffer.
    expect((depRun.message.image as { data: unknown }).data).toBe(
      (img as { data: unknown }).data,
    );
    const map = { predicted_depth: { data: [1, 2, 3, 4], dims: [1, 2, 2] } };
    act(() => dep().emit({ type: "result", id: depRun.message.id as number, result: map }));

    const out = (await done) as { detections: unknown; depth: unknown };
    expect(out.detections).toEqual(boxes);
    expect(out.depth).toEqual(map);
    await waitFor(() => expect(hook.result.current.result).toEqual(out));
  });
});
