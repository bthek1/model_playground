import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { GnnMetrics } from "@/webgpu/gnn";
import type {
  GraphSummary,
  GraphTrainRequest,
  GraphTrainResult,
} from "@/webgpu/graphSession";

const terminate = vi.fn();
const mockCreateWorker = vi.fn(() => ({ terminate }));
const mockLoadGraph = vi.fn();
const mockTrainGraph = vi.fn();

vi.mock("@/webgpu/workerClient", () => ({
  createWebGPUWorker: () => mockCreateWorker(),
  loadGraphInWorker: (...args: unknown[]) => mockLoadGraph(...args),
  trainGraphInWorker: (...args: unknown[]) => mockTrainGraph(...args),
}));

const { useGraphTraining } = await import("./useGraphTraining");

function summary(): GraphSummary {
  return {
    nNodes: 3,
    nFeat: 4,
    nClasses: 2,
    nEdges: 4,
    rowPtr: Uint32Array.from([0, 1, 3, 4]),
    colIdx: Uint32Array.from([1, 0, 2, 1]),
    labels: Uint8Array.from([0, 1, 0]),
    trainMask: Uint8Array.from([1, 0, 0]),
    x: Float32Array.from([0, 0.5, 1]),
    y: Float32Array.from([0, 0.5, 1]),
    layoutMs: 900,
    backend: "webgpu",
  };
}

function metric(over: Partial<GnnMetrics> = {}): GnnMetrics {
  return {
    epoch: 0,
    totalEpochs: 2,
    loss: 1,
    trainAcc: 0.5,
    valAcc: 0.5,
    testAcc: 0.5,
    smoothness: 0.9,
    deadFraction: 0,
    ...over,
  };
}

const REQUEST: GraphTrainRequest = {
  arch: "gcn",
  layers: 2,
  hidden: 16,
  learningRate: 0.01,
  weightDecay: 5e-4,
  dropout: 0.5,
  epochs: 2,
};

let capturedOnEpoch: ((m: GnnMetrics, p: Uint8Array) => void) | null;
let resolveTraining: (r: GraphTrainResult) => void;
let rejectTraining: (e: unknown) => void;
const cancelSpy = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  capturedOnEpoch = null;
  mockLoadGraph.mockResolvedValue(summary());
  mockTrainGraph.mockImplementation(
    (
      _worker: unknown,
      _req: unknown,
      onEpoch: (m: GnnMetrics, p: Uint8Array) => void,
    ) => {
      capturedOnEpoch = onEpoch;
      return {
        promise: new Promise<GraphTrainResult>((res, rej) => {
          resolveTraining = res;
          rejectTraining = rej;
        }),
        cancel: cancelSpy,
      };
    },
  );
  // The hook batches streamed epochs to animation frames.
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    cb(0);
    return 0;
  });
});

/** Render the hook and take it through a successful load. */
async function loaded() {
  const hook = renderHook(() => useGraphTraining());
  await act(async () => {
    hook.result.current.load();
  });
  await waitFor(() => expect(hook.result.current.status).toBe("ready"));
  return hook;
}

describe("useGraphTraining — load", () => {
  it("starts idle and creates no worker, so arriving costs nothing", () => {
    const { result } = renderHook(() => useGraphTraining());
    expect(result.current.status).toBe("idle");
    expect(result.current.summary).toBeNull();
    expect(result.current.backend).toBeNull();
    expect(mockCreateWorker).not.toHaveBeenCalled();
    expect(mockLoadGraph).not.toHaveBeenCalled();
  });

  it("loads the graph and reports the backend a run would use", async () => {
    const { result } = await loaded();
    expect(mockLoadGraph).toHaveBeenCalledTimes(1);
    expect(result.current.summary?.nNodes).toBe(3);
    expect(result.current.backend).toBe("webgpu");
  });

  it("does not start a second load, however often it is clicked", async () => {
    // The layout behind this is the most expensive thing on the page, and two
    // clicks in one React batch would both read the same stale status.
    const { result } = renderHook(() => useGraphTraining());
    await act(async () => {
      result.current.load();
      result.current.load();
      result.current.load();
    });
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(mockLoadGraph).toHaveBeenCalledTimes(1);

    await act(async () => {
      result.current.load();
    });
    expect(mockLoadGraph).toHaveBeenCalledTimes(1);
  });

  it("surfaces a load failure and lets retry arm another attempt", async () => {
    mockLoadGraph.mockRejectedValueOnce(new Error("decode failed"));
    const { result } = renderHook(() => useGraphTraining());

    await act(async () => {
      result.current.load();
    });
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.loadError).toBe("decode failed");

    act(() => result.current.retry());
    expect(result.current.status).toBe("idle");
    expect(result.current.loadError).toBeNull();

    await act(async () => {
      result.current.load();
    });
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(mockLoadGraph).toHaveBeenCalledTimes(2);
  });
});

describe("useGraphTraining — train", () => {
  it("refuses to train before the graph is loaded", () => {
    const { result } = renderHook(() => useGraphTraining());
    act(() => result.current.start(REQUEST));
    expect(mockTrainGraph).not.toHaveBeenCalled();
    expect(result.current.trainError).toMatch(/load the graph first/i);
    expect(result.current.training).toBe(false);
  });

  it("streams epochs and the predictions the canvas draws", async () => {
    const { result } = await loaded();
    act(() => result.current.start(REQUEST));
    expect(result.current.training).toBe(true);

    act(() => {
      capturedOnEpoch?.(metric({ epoch: 0 }), Uint8Array.from([0, 0, 0]));
      capturedOnEpoch?.(metric({ epoch: 1 }), Uint8Array.from([0, 1, 0]));
    });

    expect(result.current.metrics.map((m) => m.epoch)).toEqual([0, 1]);
    expect([...(result.current.predictions ?? [])]).toEqual([0, 1, 0]);
  });

  it("records one depth point per run and reports how long it took", async () => {
    const { result } = await loaded();
    act(() => result.current.start(REQUEST));
    await act(async () => {
      resolveTraining({
        metrics: metric({ epoch: 1, testAcc: 0.78, smoothness: 0.95 }),
        predictions: Uint8Array.from([0, 1, 0]),
        backend: "webgpu",
        elapsedMs: 1400,
      });
    });

    await waitFor(() => expect(result.current.training).toBe(false));
    expect(result.current.elapsedMs).toBe(1400);
    expect(result.current.depthHistory).toEqual([
      { arch: "gcn", layers: 2, testAcc: 0.78, smoothness: 0.95, deadFraction: 0 },
    ]);
  });

  it("keeps one point per (architecture, depth), replacing a repeat", async () => {
    const { result } = await loaded();

    const run = async (req: GraphTrainRequest, testAcc: number) => {
      act(() => result.current.start(req));
      await act(async () => {
        resolveTraining({
          metrics: metric({ testAcc }),
          predictions: Uint8Array.from([0, 0, 0]),
          backend: "webgpu",
          elapsedMs: 1,
        });
      });
      await waitFor(() => expect(result.current.training).toBe(false));
    };

    await run(REQUEST, 0.78);
    await run({ ...REQUEST, layers: 8 }, 0.52);
    await run({ ...REQUEST, arch: "gat" }, 0.69);
    // Re-running an existing pair must move its point, not stack a second one.
    await run(REQUEST, 0.8);

    expect(
      result.current.depthHistory.map((p) => [p.arch, p.layers, p.testAcc]),
    ).toEqual([
      ["gcn", 8, 0.52],
      ["gat", 2, 0.69],
      ["gcn", 2, 0.8],
    ]);
  });

  it("treats a stop before the first epoch as a normal outcome", async () => {
    // metrics: null means the run was cancelled before it produced anything.
    // Recording it would put an all-zero prediction on the canvas and a bogus
    // point on the depth chart.
    const { result } = await loaded();
    act(() => result.current.start(REQUEST));
    await act(async () => {
      resolveTraining({
        metrics: null,
        predictions: new Uint8Array(3),
        backend: "webgpu",
        elapsedMs: 20,
      });
    });

    await waitFor(() => expect(result.current.training).toBe(false));
    expect(result.current.trainError).toBeNull();
    expect(result.current.depthHistory).toEqual([]);
    expect(result.current.predictions).toBeNull();
    expect(result.current.elapsedMs).toBeNull();
  });

  it("puts a training failure in trainError and clears the running flag", async () => {
    const { result } = await loaded();
    act(() => result.current.start(REQUEST));
    await act(async () => {
      rejectTraining(new Error("device lost"));
    });

    await waitFor(() => expect(result.current.training).toBe(false));
    expect(result.current.trainError).toBe("device lost");
    expect(result.current.depthHistory).toEqual([]);
  });

  it("clears the previous run's result when a new one starts", async () => {
    const { result } = await loaded();
    act(() => result.current.start(REQUEST));
    act(() => capturedOnEpoch?.(metric(), Uint8Array.from([1, 1, 1])));
    expect(result.current.metrics).toHaveLength(1);

    await act(async () => {
      resolveTraining({
        metrics: metric(),
        predictions: Uint8Array.from([1, 1, 1]),
        backend: "webgpu",
        elapsedMs: 5,
      });
    });
    await waitFor(() => expect(result.current.training).toBe(false));

    act(() => result.current.start({ ...REQUEST, layers: 4 }));
    expect(result.current.metrics).toEqual([]);
    expect(result.current.predictions).toBeNull();
    expect(result.current.elapsedMs).toBeNull();
    // The history is the one thing that survives — it is the comparison.
    expect(result.current.depthHistory).toHaveLength(1);
  });

  it("asks the worker to stop, and reuses the one worker across runs", async () => {
    const { result } = await loaded();
    act(() => result.current.start(REQUEST));
    act(() => result.current.stop());
    expect(cancelSpy).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveTraining({
        metrics: null,
        predictions: new Uint8Array(3),
        backend: "webgpu",
        elapsedMs: 1,
      });
    });
    await waitFor(() => expect(result.current.training).toBe(false));

    act(() => result.current.start(REQUEST));
    expect(mockCreateWorker).toHaveBeenCalledTimes(1);
  });

  it("stops being a no-op once nothing is running", async () => {
    const { result } = await loaded();
    act(() => result.current.stop());
    expect(cancelSpy).not.toHaveBeenCalled();
  });
});

describe("useGraphTraining — teardown", () => {
  it("cancels and terminates the worker on unmount", async () => {
    const { result, unmount } = await loaded();
    act(() => result.current.start(REQUEST));

    unmount();
    expect(cancelSpy).toHaveBeenCalledTimes(1);
    expect(terminate).toHaveBeenCalledTimes(1);
  });
});
