import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { QLearningRequest, RlProgress, RlTrainResult } from "@/rl/types";

const terminate = vi.fn();
const mockCreateWorker = vi.fn(() => ({ terminate }));
const mockTrain = vi.fn();

vi.mock("@/rl/client", () => ({
  createRlWorker: () => mockCreateWorker(),
  trainRlInWorker: (...args: unknown[]) => mockTrain(...args),
}));

const { runKey, useRlTraining } = await import("./useRlTraining");

const REQUEST: QLearningRequest = {
  algorithm: "q-learning",
  map: "4x4",
  slippery: false,
  alpha: 0.5,
  gamma: 0.95,
  epsilon: 1,
  decay: true,
  episodes: 10,
  seed: 1,
};

function progress(returns: number[], episode: number): RlProgress {
  return {
    returns,
    episode,
    totalEpisodes: 10,
    steps: episode * 7,
    epsilon: 0.5,
    render: { kind: "grid", map: "4x4", q: new Float32Array(64), agent: 3 },
  };
}

function result(over: Partial<RlTrainResult> = {}): RlTrainResult {
  return {
    episodes: 10,
    steps: 70,
    elapsedMs: 5,
    stopped: false,
    epsilonChanged: false,
    render: { kind: "grid", map: "4x4", q: new Float32Array(64), agent: 15 },
    ...over,
  };
}

let onProgress: ((p: RlProgress) => void) | null;
let resolveRun: (r: RlTrainResult) => void;
const cancel = vi.fn();
const control = vi.fn();
let frames: FrameRequestCallback[];

beforeEach(() => {
  vi.clearAllMocks();
  onProgress = null;
  frames = [];
  mockTrain.mockImplementation((_w: unknown, _req: unknown, cb: (p: RlProgress) => void) => {
    onProgress = cb;
    return {
      promise: new Promise<RlTrainResult>((res) => {
        resolveRun = res;
      }),
      cancel,
      control,
    };
  });
  // Hold frames so the test decides when one happens.
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    frames.push(cb);
    return frames.length;
  });
});

function frame() {
  act(() => {
    const due = frames;
    frames = [];
    for (const cb of due) cb(0);
  });
}

describe("useRlTraining", () => {
  it("creates no worker until the first Train", () => {
    renderHook(() => useRlTraining());
    expect(mockCreateWorker).not.toHaveBeenCalled();
    expect(mockTrain).not.toHaveBeenCalled();
  });

  it("flushes once per frame, not once per message", () => {
    let renders = 0;
    const { result: hook } = renderHook(() => {
      renders++;
      return useRlTraining();
    });
    act(() => hook.current.start(REQUEST, 3000));
    expect(mockTrain).toHaveBeenCalledWith(expect.anything(), REQUEST, expect.any(Function), 3000);

    const before = renders;
    act(() => {
      onProgress?.(progress([0, 1], 2));
      onProgress?.(progress([0], 3));
      onProgress?.(progress([1, 1], 5));
    });
    expect(renders).toBe(before); // three messages, no commit yet
    expect(frames).toHaveLength(1); // and exactly one frame asked for

    frame();
    expect(hook.current.returns).toEqual([0, 1, 0, 1, 1]);
    expect(hook.current.progress?.episode).toBe(5);
    expect(hook.current.render?.agent).toBe(3);
  });

  it("sends Stop and live controls to the run in progress", () => {
    const { result: hook } = renderHook(() => useRlTraining());
    act(() => hook.current.start(REQUEST));
    act(() => hook.current.control({ epsilon: 0 }));
    expect(control).toHaveBeenCalledWith({ epsilon: 0 });
    act(() => hook.current.stop());
    expect(cancel).toHaveBeenCalledTimes(1);
  });

  it("ignores a second Train while one is running", () => {
    const { result: hook } = renderHook(() => useRlTraining());
    act(() => hook.current.start(REQUEST));
    act(() => hook.current.start(REQUEST));
    expect(mockTrain).toHaveBeenCalledTimes(1);
  });

  it("builds a history, one entry per configuration, replacing a re-run", async () => {
    const { result: hook } = renderHook(() => useRlTraining());

    act(() => hook.current.start(REQUEST));
    act(() => onProgress?.(progress([0, 1, 1], 3)));
    await act(async () => resolveRun(result()));
    await waitFor(() => expect(hook.current.training).toBe(false));
    expect(hook.current.history).toHaveLength(1);
    expect(hook.current.history[0].returns).toEqual([0, 1, 1]);

    // A second seed adds to the comparison…
    const other = { ...REQUEST, seed: 2 };
    act(() => hook.current.start(other));
    await act(async () => resolveRun(result()));
    await waitFor(() => expect(hook.current.history).toHaveLength(2));

    // …and the same configuration again replaces its own entry.
    act(() => hook.current.start(REQUEST));
    act(() => onProgress?.(progress([1], 1)));
    await act(async () => resolveRun(result()));
    await waitFor(() => expect(hook.current.training).toBe(false));
    expect(hook.current.history.map((r) => r.key)).toEqual([runKey(other), runKey(REQUEST)]);
    expect(hook.current.history[1].returns).toEqual([1]);
    // The worker is reused across runs, never re-created.
    expect(mockCreateWorker).toHaveBeenCalledTimes(1);
  });

  it("adds nothing to the history when stopped before an episode finished", async () => {
    const { result: hook } = renderHook(() => useRlTraining());
    act(() => hook.current.start(REQUEST));
    await act(async () => resolveRun(result({ episodes: 0, steps: 3, stopped: true })));
    await waitFor(() => expect(hook.current.training).toBe(false));
    expect(hook.current.history).toEqual([]);
    expect(hook.current.result?.stopped).toBe(true);
  });

  it("terminates the worker on unmount", () => {
    const { result: hook, unmount } = renderHook(() => useRlTraining());
    act(() => hook.current.start(REQUEST));
    unmount();
    expect(cancel).toHaveBeenCalled();
    expect(terminate).toHaveBeenCalledTimes(1);
  });
});
