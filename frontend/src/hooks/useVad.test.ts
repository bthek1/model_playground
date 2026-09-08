import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { VadResponse } from "@/audio/vad/types";

// A stand-in Web Worker: records posted messages, lets a test push responses.
class FakeWorker {
  onmessage: ((e: MessageEvent<VadResponse>) => void) | null = null;
  posted: Array<{ message: unknown; transfer?: Transferable[] }> = [];
  terminated = false;

  postMessage(message: unknown, transfer?: Transferable[]) {
    this.posted.push({ message, transfer });
  }
  terminate() {
    this.terminated = true;
  }
  emit(data: VadResponse) {
    this.onmessage?.({ data } as MessageEvent<VadResponse>);
  }
}

let lastWorker: FakeWorker;

vi.mock("@/audio/vad/vadClient", () => ({
  createVadWorker: () => {
    lastWorker = new FakeWorker();
    return lastWorker as unknown as Worker;
  },
}));

const { useVad } = await import("./useVad");

const SILERO = "onnx-community/silero-vad";
const ready = (model = SILERO) =>
  lastWorker.emit({ type: "ready", model, backend: "wasm" });

const result = (id: number, frames = 3): VadResponse => ({
  type: "result",
  id,
  result: {
    probabilities: new Float32Array(frames).fill(0.9),
    frameSamples: 512,
    sampleRate: 16000,
    samples: frames * 512,
  },
});

/** `Array.prototype.at` is outside this project's ES2020 lib target. */
function last<T>(items: readonly T[]): T | undefined {
  return items[items.length - 1];
}

describe("useVad", () => {
  afterEach(() => vi.clearAllMocks());

  it("downloads nothing until asked — even 2 MB is the user's bandwidth", () => {
    const { result: hook } = renderHook(() => useVad());
    expect(hook.current.idle).toBe(true);
    expect(hook.current.loading).toBe(false);
    expect(lastWorker).toBeUndefined();
  });

  it("posts a load message with the model when load() is called", () => {
    const { result: hook } = renderHook(() => useVad());
    act(() => hook.current.load());
    expect(hook.current.loading).toBe(true);
    expect(lastWorker.posted[0].message).toEqual({ type: "load", model: SILERO });
  });

  it("flips to ready and records the backend", async () => {
    const { result: hook } = renderHook(() => useVad());
    act(() => hook.current.load());
    act(() => ready());
    await waitFor(() => expect(hook.current.ready).toBe(true));
    expect(hook.current.backend).toBe("wasm");
  });

  it("transfers the audio buffer rather than copying it", async () => {
    const { result: hook } = renderHook(() => useVad());
    act(() => hook.current.load());
    act(() => ready());
    await waitFor(() => expect(hook.current.ready).toBe(true));

    const audio = new Float32Array(1024);
    const buffer = audio.buffer;
    // Never answered: the unmount at the end of the test rejects it, and an
    // unhandled rejection fails the suite as a whole.
    act(() => void hook.current.run(audio).catch(() => {}));

    const run = last(lastWorker.posted)!;
    expect(run.message).toMatchObject({ type: "run", id: expect.any(Number) });
    expect(run.transfer).toEqual([buffer]);
  });

  it("resolves each run with its own result and exposes the latest", async () => {
    const { result: hook } = renderHook(() => useVad());
    act(() => hook.current.load());
    act(() => ready());
    await waitFor(() => expect(hook.current.ready).toBe(true));

    let resolved: number | undefined;
    act(() => {
      void hook.current
        .run(new Float32Array(512))
        .then((r) => (resolved = r.probabilities.length));
    });
    const id = (last(lastWorker.posted)!.message as { id: number }).id;
    act(() => lastWorker.emit(result(id, 5)));

    await waitFor(() => expect(resolved).toBe(5));
    expect(hook.current.result?.probabilities).toHaveLength(5);
  });

  it("keeps `running` true until BOTH overlapping runs return", async () => {
    // The inflight count, not a boolean: the page's transport must stay disabled
    // while a second clip is still being scored.
    const { result: hook } = renderHook(() => useVad());
    act(() => hook.current.load());
    act(() => ready());
    await waitFor(() => expect(hook.current.ready).toBe(true));

    act(() => {
      void hook.current.run(new Float32Array(512));
      void hook.current.run(new Float32Array(512));
    });
    await waitFor(() => expect(hook.current.running).toBe(true));

    const ids = lastWorker.posted
      .filter((p) => (p.message as { type: string }).type === "run")
      .map((p) => (p.message as { id: number }).id);

    act(() => lastWorker.emit(result(ids[0])));
    expect(hook.current.running).toBe(true);
    act(() => lastWorker.emit(result(ids[1])));
    await waitFor(() => expect(hook.current.running).toBe(false));
  });

  it("treats an id-less error as a load failure and offers a retry", async () => {
    const { result: hook } = renderHook(() => useVad());
    act(() => hook.current.load());
    act(() =>
      lastWorker.emit({ type: "error", error: "onnx/model.onnx: 404 Not Found" }),
    );
    await waitFor(() => expect(hook.current.status).toBe("error"));
    expect(hook.current.error).toMatch(/404/);

    act(() => hook.current.retry());
    await waitFor(() => expect(hook.current.loading).toBe(true));
  });

  it("treats an error with an id as a run failure and stays ready", async () => {
    const { result: hook } = renderHook(() => useVad());
    act(() => hook.current.load());
    act(() => ready());
    await waitFor(() => expect(hook.current.ready).toBe(true));

    let rejected: string | undefined;
    act(() => {
      void hook.current
        .run(new Float32Array(512))
        .catch((e: Error) => (rejected = e.message));
    });
    const id = (last(lastWorker.posted)!.message as { id: number }).id;
    act(() => lastWorker.emit({ type: "error", id, error: "session closed" }));

    await waitFor(() => expect(rejected).toBe("session closed"));
    expect(hook.current.status).toBe("ready");
  });

  it("tears the worker down when the model changes", async () => {
    const { result: hook, rerender } = renderHook(
      ({ model }: { model: string }) => useVad(model),
      { initialProps: { model: SILERO } },
    );
    act(() => hook.current.load());
    act(() => ready());
    await waitFor(() => expect(hook.current.ready).toBe(true));

    const first = lastWorker;
    rerender({ model: "energy" });
    await waitFor(() => expect(first.terminated).toBe(true));
    expect(hook.current.ready).toBe(false);
  });
});
