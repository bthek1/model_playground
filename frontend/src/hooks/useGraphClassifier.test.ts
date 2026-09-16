import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { GraphClassMetrics } from "@/webgpu/graphPool";
import type {
  GraphLayoutPayload,
  ProteinSummary,
  ProteinTrainRequest,
  ProteinTrainResult,
} from "@/webgpu/proteinSession";

const terminate = vi.fn();
const mockCreateWorker = vi.fn(() => ({ terminate }));
const mockLoad = vi.fn();
const mockLayout = vi.fn();
const mockTrain = vi.fn();

vi.mock("@/webgpu/workerClient", () => ({
  createWebGPUWorker: () => mockCreateWorker(),
  loadProteinsInWorker: (...args: unknown[]) => mockLoad(...args),
  layoutProteinInWorker: (...args: unknown[]) => mockLayout(...args),
  trainProteinsInWorker: (...args: unknown[]) => mockTrain(...args),
}));

const { useGraphClassifier } = await import("./useGraphClassifier");

function summary(): ProteinSummary {
  return {
    nGraphs: 4,
    nNodes: 10,
    nFeat: 3,
    nClasses: 2,
    nEdges: 12,
    nTrain: 2,
    nVal: 0,
    nTest: 2,
    labels: Uint8Array.from([0, 1, 0, 1]),
    graphPtr: Uint32Array.from([0, 3, 5, 8, 10]),
    testIdx: Uint32Array.from([2, 3]),
    baselineAcc: 0.5,
    fromCache: false,
    loadMs: 700,
    backend: "webgpu",
  };
}

function metric(over: Partial<GraphClassMetrics> = {}): GraphClassMetrics {
  return {
    epoch: 0,
    totalEpochs: 2,
    loss: 0.69,
    trainAcc: 0.6,
    valAcc: 0.6,
    testAcc: 0.62,
    baselineAcc: 0.5,
    ...over,
  };
}

function layout(index: number): GraphLayoutPayload {
  return {
    index,
    nNodes: 2,
    rowPtr: Uint32Array.from([0, 1, 2]),
    colIdx: Uint32Array.from([1, 0]),
    x: Float32Array.from([0, 1]),
    y: Float32Array.from([0.5, 0.5]),
  };
}

const REQUEST: ProteinTrainRequest = {
  arch: "gin",
  readout: "mean",
  layers: 3,
  hidden: 32,
  learningRate: 0.01,
  weightDecay: 0,
  dropout: 0.2,
  epochs: 2,
};

let capturedOnEpoch: ((m: GraphClassMetrics, p: Uint8Array) => void) | null;
let resolveTraining: (r: ProteinTrainResult) => void;
let rejectTraining: (e: unknown) => void;
const cancelSpy = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  capturedOnEpoch = null;
  mockLoad.mockResolvedValue(summary());
  mockLayout.mockImplementation(async (_worker: unknown, graph: number) =>
    layout(graph),
  );
  mockTrain.mockImplementation(
    (
      _worker: unknown,
      _req: unknown,
      onEpoch: (m: GraphClassMetrics, p: Uint8Array) => void,
    ) => {
      capturedOnEpoch = onEpoch;
      return {
        promise: new Promise<ProteinTrainResult>((res, rej) => {
          resolveTraining = res;
          rejectTraining = rej;
        }),
        cancel: cancelSpy,
      };
    },
  );
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    cb(0);
    return 0;
  });
});

async function loaded() {
  const hook = renderHook(() => useGraphClassifier());
  await act(async () => {
    hook.result.current.load();
  });
  await waitFor(() => expect(hook.result.current.status).toBe("ready"));
  return hook;
}

describe("useGraphClassifier — load", () => {
  it("starts idle and downloads nothing, so arriving costs nothing", () => {
    // The only page in the repo that fetches a dataset: this matters more here
    // than on the two routes that read a bundled file.
    const { result } = renderHook(() => useGraphClassifier());
    expect(result.current.status).toBe("idle");
    expect(mockCreateWorker).not.toHaveBeenCalled();
    expect(mockLoad).not.toHaveBeenCalled();
  });

  it("loads the dataset and carries the baseline in the summary", async () => {
    const { result } = await loaded();
    expect(mockLoad).toHaveBeenCalledTimes(1);
    expect(result.current.summary?.nGraphs).toBe(4);
    expect(result.current.summary?.baselineAcc).toBe(0.5);
    expect(result.current.backend).toBe("webgpu");
  });

  it("does not start a second download, however often it is clicked", async () => {
    const { result } = renderHook(() => useGraphClassifier());
    await act(async () => {
      result.current.load();
      result.current.load();
    });
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(mockLoad).toHaveBeenCalledTimes(1);
  });

  it("surfaces a failed download and lets retry arm another attempt", async () => {
    mockLoad.mockRejectedValueOnce(
      new Error("the dataset could not be fetched (503)"),
    );
    const { result } = renderHook(() => useGraphClassifier());
    await act(async () => {
      result.current.load();
    });
    await waitFor(() => expect(result.current.status).toBe("error"));
    expect(result.current.loadError).toMatch(/503/);

    act(() => {
      result.current.retry();
    });
    expect(result.current.status).toBe("idle");
    await act(async () => {
      result.current.load();
    });
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(mockLoad).toHaveBeenCalledTimes(2);
  });
});

describe("useGraphClassifier — layouts", () => {
  it("does nothing before there is a worker to ask", () => {
    const { result } = renderHook(() => useGraphClassifier());
    act(() => {
      result.current.requestLayout(0);
    });
    expect(mockLayout).not.toHaveBeenCalled();
  });

  it("lays a tile out on request and keeps the result", async () => {
    const hook = await loaded();
    await act(async () => {
      hook.result.current.requestLayout(2);
    });
    await waitFor(() => expect(hook.result.current.layouts.has(2)).toBe(true));
    expect(mockLayout).toHaveBeenCalledWith(expect.anything(), 2);
  });

  it("asks once per graph, even while the first ask is in flight", async () => {
    // A gallery re-renders while its layouts arrive, and every render asks for
    // whatever is still missing — so the in-flight set is what keeps this to one
    // request per tile rather than one per render.
    let release: (() => void) | undefined;
    mockLayout.mockImplementationOnce(
      (_worker: unknown, graph: number) =>
        new Promise((res) => {
          release = () => res(layout(graph));
        }),
    );
    const hook = await loaded();
    act(() => {
      hook.result.current.requestLayout(3);
      hook.result.current.requestLayout(3);
      hook.result.current.requestLayout(3);
    });
    expect(mockLayout).toHaveBeenCalledTimes(1);

    await act(async () => {
      release?.();
    });
    await waitFor(() => expect(hook.result.current.layouts.has(3)).toBe(true));
    act(() => {
      hook.result.current.requestLayout(3);
    });
    expect(mockLayout).toHaveBeenCalledTimes(1);
  });

  it("leaves a failed tile as a placeholder and lets it be asked for again", async () => {
    // A tile is a picture, not a result: one that cannot be laid out must not
    // fail the page, and must not be remembered as permanently in flight.
    mockLayout.mockRejectedValueOnce(new Error("no graph 9"));
    const hook = await loaded();
    await act(async () => {
      hook.result.current.requestLayout(1);
    });
    await waitFor(() => expect(mockLayout).toHaveBeenCalledTimes(1));
    expect(hook.result.current.layouts.has(1)).toBe(false);
    expect(hook.result.current.status).toBe("ready");

    await act(async () => {
      hook.result.current.requestLayout(1);
    });
    await waitFor(() => expect(hook.result.current.layouts.has(1)).toBe(true));
    expect(mockLayout).toHaveBeenCalledTimes(2);
  });
});

describe("useGraphClassifier — training", () => {
  it("refuses to train before the dataset is loaded", () => {
    const { result } = renderHook(() => useGraphClassifier());
    act(() => {
      result.current.start(REQUEST);
    });
    expect(result.current.trainError).toMatch(/load the dataset/i);
    expect(mockTrain).not.toHaveBeenCalled();
  });

  it("streams epochs and the per-graph predictions the gallery draws", async () => {
    const hook = await loaded();
    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    act(() => {
      capturedOnEpoch?.(metric({ epoch: 0 }), Uint8Array.from([0, 0, 0, 0]));
      capturedOnEpoch?.(metric({ epoch: 1 }), Uint8Array.from([0, 1, 0, 1]));
    });
    expect(hook.result.current.metrics).toHaveLength(2);
    expect(Array.from(hook.result.current.predicted!)).toEqual([0, 1, 0, 1]);
  });

  it("keeps one point per (architecture, readout), replacing a repeat", async () => {
    const hook = await loaded();
    const finish = async (req: ProteinTrainRequest, testAcc: number) => {
      await act(async () => {
        hook.result.current.start(req);
      });
      await act(async () => {
        resolveTraining({
          metrics: metric({ testAcc }),
          predicted: Uint8Array.from([0, 1, 0, 1]),
          backend: "webgpu",
          elapsedMs: 900,
        });
      });
      await waitFor(() => expect(hook.result.current.training).toBe(false));
    };

    await finish(REQUEST, 0.7);
    await finish(REQUEST, 0.72);
    expect(hook.result.current.history).toEqual([
      { arch: "gin", readout: "mean", testAcc: 0.72 },
    ]);

    // Sum and mean are different experiments, not repeats of one.
    await finish({ ...REQUEST, readout: "sum" }, 0.74);
    expect(hook.result.current.history).toHaveLength(2);
    expect(hook.result.current.elapsedMs).toBe(900);
  });

  it("treats a stop before the first epoch as a normal outcome", async () => {
    const hook = await loaded();
    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    await act(async () => {
      resolveTraining({
        metrics: null,
        predicted: new Uint8Array(4),
        backend: "webgpu",
        elapsedMs: 5,
      });
    });
    await waitFor(() => expect(hook.result.current.training).toBe(false));
    expect(hook.result.current.trainError).toBeNull();
    expect(hook.result.current.history).toEqual([]);
  });

  it("puts a training failure in trainError and clears the running flag", async () => {
    const hook = await loaded();
    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    await act(async () => {
      rejectTraining(new Error("device lost"));
    });
    await waitFor(() => expect(hook.result.current.training).toBe(false));
    expect(hook.result.current.trainError).toMatch(/device lost/);
  });

  it("clears the previous run's predictions when a new one starts", async () => {
    // Otherwise the gallery outlines the new run's tiles with the old run's
    // answers for as long as the first epoch takes.
    const hook = await loaded();
    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    act(() => {
      capturedOnEpoch?.(metric(), Uint8Array.from([1, 1, 1, 1]));
    });
    expect(hook.result.current.predicted).not.toBeNull();

    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    expect(hook.result.current.predicted).toBeNull();
    expect(hook.result.current.metrics).toEqual([]);
  });

  it("cancels and terminates the worker on unmount", async () => {
    const hook = await loaded();
    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    act(() => {
      hook.result.current.stop();
    });
    expect(cancelSpy).toHaveBeenCalledTimes(1);
    hook.unmount();
    expect(terminate).toHaveBeenCalled();
  });
});
