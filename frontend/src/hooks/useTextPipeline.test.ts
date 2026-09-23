import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { TextResponse } from "@/text/types";

/** The most recent message. `Array.prototype.at` is past this project's lib. */
function last<T>(xs: T[]): T | undefined {
  return xs[xs.length - 1];
}

// A stand-in Web Worker: records posted messages, lets a test push responses.
class FakeWorker {
  onmessage: ((e: MessageEvent<TextResponse>) => void) | null = null;
  posted: unknown[] = [];
  terminated = false;

  postMessage(message: unknown) {
    this.posted.push(message);
  }
  terminate() {
    this.terminated = true;
  }
  emit(data: TextResponse) {
    this.onmessage?.({ data } as MessageEvent<TextResponse>);
  }
}

let lastWorker: FakeWorker;
let spawned = 0;

vi.mock("@/text/client", () => ({
  createTextWorker: () => {
    spawned += 1;
    lastWorker = new FakeWorker();
    return lastWorker as unknown as Worker;
  },
}));

const { useTextPipeline } = await import("./useTextPipeline");

const MODEL = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";

async function loadedHook() {
  const hook = renderHook(() =>
    useTextPipeline("text-classification", MODEL, false),
  );
  act(() => hook.result.current.load());
  act(() => lastWorker.emit({ type: "ready", model: MODEL, backend: "wasm" }));
  await waitFor(() => expect(hook.result.current.ready).toBe(true));
  return hook;
}

describe("useTextPipeline", () => {
  beforeEach(() => {
    spawned = 0;
  });

  it("starts idle and spawns no worker until load() is called", () => {
    const hook = renderHook(() =>
      useTextPipeline("text-classification", MODEL, false),
    );
    expect(hook.result.current.status).toBe("idle");
    expect(hook.result.current.idle).toBe(true);
    // Weights are the user's bandwidth: nothing is even constructed until the
    // LOAD button asks (model-page-pattern.md §1.2).
    expect(spawned).toBe(0);

    act(() => hook.result.current.load());
    expect(spawned).toBe(1);
    expect(lastWorker.posted[0]).toEqual({
      type: "load",
      task: "text-classification",
      model: MODEL,
    });
  });

  it("carries a catalogue dtype override on the load message", () => {
    const hook = renderHook(() =>
      useTextPipeline("text-classification", MODEL, false, { wasm: "fp32" }),
    );
    act(() => hook.result.current.load());
    expect(lastWorker.posted[0]).toEqual({
      type: "load",
      task: "text-classification",
      model: MODEL,
      dtypes: { wasm: "fp32" },
    });
  });

  it("posts the input and args, and resolves with the correlated result", async () => {
    const hook = await loadedHook();
    let out: unknown;
    act(() => {
      void hook.result.current
        .run("a triumph of tedium", [{ top_k: 6 }])
        .then((r) => {
          out = r;
        });
    });

    const sent = last(lastWorker.posted) as { id: number };
    // No transfer list and no payload conversion: a string crosses the wire as
    // itself, which is the whole reason this modality has no `io.ts`.
    expect(last(lastWorker.posted)).toEqual({
      type: "run",
      id: sent.id,
      input: "a triumph of tedium",
      args: [{ top_k: 6 }],
    });

    const result = [{ label: "NEGATIVE", score: 0.99 }];
    act(() => lastWorker.emit({ type: "result", id: sent.id, result }));
    await waitFor(() => expect(out).toEqual(result));
  });

  it("keeps `running` an inflight count across overlapping runs", async () => {
    const hook = await loadedHook();
    act(() => {
      void hook.result.current.run("one").catch(() => {});
      void hook.result.current.run("two").catch(() => {});
    });
    await waitFor(() => expect(hook.result.current.running).toBe(true));

    const ids = (lastWorker.posted.slice(-2) as { id: number }[]).map((m) => m.id);
    act(() => lastWorker.emit({ type: "result", id: ids[0], result: [] }));
    // A boolean would report idle here, with the second request still in flight.
    expect(hook.result.current.running).toBe(true);

    act(() => lastWorker.emit({ type: "result", id: ids[1], result: [] }));
    await waitFor(() => expect(hook.result.current.running).toBe(false));
  });

  it("separates a load failure from a run failure by the id", async () => {
    const hook = await loadedHook();
    act(() =>
      lastWorker.emit({ type: "error", id: 1, error: "input too long" }),
    );
    // Machine B: the model is still loaded.
    await waitFor(() => expect(hook.result.current.error).toBe("input too long"));
    expect(hook.result.current.status).toBe("ready");

    act(() => lastWorker.emit({ type: "error", error: "404 not found" }));
    await waitFor(() => expect(hook.result.current.status).toBe("error"));
  });

  it("tears the worker down when the model changes", async () => {
    const hook = renderHook(
      ({ model }) => useTextPipeline("text-classification", model, false),
      { initialProps: { model: MODEL } },
    );
    act(() => hook.result.current.load());
    const first = lastWorker;

    hook.rerender({ model: "Xenova/finbert" });
    expect(first.terminated).toBe(true);
    // And the new key starts idle: switching model is a choice, not a command.
    expect(hook.result.current.status).toBe("idle");
  });
});
