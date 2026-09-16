import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { LinkMetrics } from "@/webgpu/linkPredictor";
import type {
  LinkSummary,
  LinkTrainRequest,
  LinkTrainResult,
} from "@/webgpu/linkSession";

const terminate = vi.fn();
const mockCreateWorker = vi.fn(() => ({ terminate }));
const mockLoadLink = vi.fn();
const mockTrainLink = vi.fn();

vi.mock("@/webgpu/workerClient", () => ({
  createWebGPUWorker: () => mockCreateWorker(),
  loadLinkGraphInWorker: (...args: unknown[]) => mockLoadLink(...args),
  trainLinkInWorker: (...args: unknown[]) => mockTrainLink(...args),
}));

const { useLinkPrediction } = await import("./useLinkPrediction");

function summary(over: Partial<LinkSummary> = {}): LinkSummary {
  return {
    nNodes: 4,
    nEdges: 4,
    nTrainEdges: 2,
    nValEdges: 1,
    nTestEdges: 1,
    rescued: 0,
    rowPtr: Uint32Array.from([0, 1, 2, 3, 4]),
    colIdx: Uint32Array.from([1, 0, 3, 2]),
    labels: Uint8Array.from([0, 1, 2, 3]),
    x: Float32Array.from([0, 0.3, 0.6, 1]),
    y: Float32Array.from([0, 0.5, 0.5, 1]),
    layoutMs: 900,
    backend: "webgpu",
    ...over,
  };
}

function metric(over: Partial<LinkMetrics> = {}): LinkMetrics {
  return {
    epoch: 0,
    totalEpochs: 2,
    loss: 0.7,
    trainAuc: 0.8,
    valAuc: 0.78,
    testAuc: 0.77,
    testAp: 0.75,
    ...over,
  };
}

function result(over: Partial<LinkTrainResult> = {}): LinkTrainResult {
  return {
    metrics: metric({ epoch: 1 }),
    candidates: Uint32Array.from([0, 2]),
    candidateScores: Float32Array.from([2.5]),
    embedding: Float32Array.from([1, 0, 0, 1, 1, 1, 0, 0]),
    embeddingDim: 2,
    backend: "webgpu",
    elapsedMs: 1200,
    ...over,
  };
}

const REQUEST: LinkTrainRequest = {
  arch: "gcn",
  layers: 2,
  hidden: 32,
  embedding: 16,
  learningRate: 0.01,
  weightDecay: 0,
  dropout: 0.2,
  epochs: 2,
  topK: 10,
};

let capturedOnEpoch: ((m: LinkMetrics) => void) | null;
let resolveTraining: (r: LinkTrainResult) => void;
let rejectTraining: (e: unknown) => void;
const cancelSpy = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  capturedOnEpoch = null;
  mockLoadLink.mockResolvedValue(summary());
  mockTrainLink.mockImplementation(
    (_worker: unknown, _req: unknown, onEpoch: (m: LinkMetrics) => void) => {
      capturedOnEpoch = onEpoch;
      return {
        promise: new Promise<LinkTrainResult>((res, rej) => {
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

async function loaded(options?: Parameters<
  ReturnType<typeof useLinkPrediction>["load"]
>[0]) {
  const hook = renderHook(() => useLinkPrediction());
  await act(async () => {
    hook.result.current.load(options);
  });
  await waitFor(() => expect(hook.result.current.status).toBe("ready"));
  return hook;
}

describe("useLinkPrediction — load", () => {
  it("starts idle and creates no worker, so arriving costs nothing", () => {
    const { result: r } = renderHook(() => useLinkPrediction());
    expect(r.current.status).toBe("idle");
    expect(r.current.summary).toBeNull();
    expect(mockCreateWorker).not.toHaveBeenCalled();
    expect(mockLoadLink).not.toHaveBeenCalled();
  });

  it("passes the held-out fraction through to the worker", async () => {
    // The split *is* the question this page asks, so the fraction has to reach
    // the session rather than being remembered only by the page.
    await loaded({ testFrac: 0.3 });
    expect(mockLoadLink).toHaveBeenCalledWith(expect.anything(), {
      testFrac: 0.3,
    });
  });

  it("does not start a second load, however often it is clicked", async () => {
    const { result: r } = renderHook(() => useLinkPrediction());
    await act(async () => {
      r.current.load();
      r.current.load();
    });
    await waitFor(() => expect(r.current.status).toBe("ready"));
    expect(mockLoadLink).toHaveBeenCalledTimes(1);
  });

  it("discards the previous run when the split changes", async () => {
    // A new split is a new graph: the run on screen was scored against edges
    // this one does not hold out, so it is not an answer to the question now
    // being asked. Leaving it up would show a number for the wrong split.
    const hook = await loaded();
    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    await act(async () => {
      resolveTraining(result());
    });
    await waitFor(() => expect(hook.result.current.result).not.toBeNull());

    await act(async () => {
      hook.result.current.load({ testFrac: 0.2 });
    });
    await waitFor(() => expect(hook.result.current.status).toBe("ready"));
    expect(hook.result.current.result).toBeNull();
    expect(hook.result.current.metrics).toEqual([]);
    expect(hook.result.current.history).toEqual([]);
  });

  it("surfaces a load failure and lets retry arm another attempt", async () => {
    mockLoadLink.mockRejectedValueOnce(new Error("decode failed"));
    const { result: r } = renderHook(() => useLinkPrediction());
    await act(async () => {
      r.current.load();
    });
    await waitFor(() => expect(r.current.status).toBe("error"));
    expect(r.current.loadError).toMatch(/decode failed/);

    act(() => {
      r.current.retry();
    });
    expect(r.current.status).toBe("idle");

    mockLoadLink.mockResolvedValueOnce(summary());
    await act(async () => {
      r.current.load();
    });
    await waitFor(() => expect(r.current.status).toBe("ready"));
  });
});

describe("useLinkPrediction — training", () => {
  it("refuses to train before the graph is loaded", () => {
    const { result: r } = renderHook(() => useLinkPrediction());
    act(() => {
      r.current.start(REQUEST);
    });
    expect(r.current.trainError).toMatch(/load the graph/i);
    expect(mockTrainLink).not.toHaveBeenCalled();
  });

  it("streams metrics without a per-epoch picture", async () => {
    // Unlike /graph, nothing is drawn per epoch here: the candidates are scored
    // once after the last one, so only the metrics stream.
    const hook = await loaded();
    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    expect(hook.result.current.training).toBe(true);

    act(() => {
      capturedOnEpoch?.(metric({ epoch: 0 }));
      capturedOnEpoch?.(metric({ epoch: 1, loss: 0.5 }));
    });
    expect(hook.result.current.metrics).toHaveLength(2);
    expect(hook.result.current.result).toBeNull();
  });

  it("records one point per (architecture, depth), replacing a repeat", async () => {
    const hook = await loaded();
    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    await act(async () => {
      resolveTraining(result({ metrics: metric({ testAuc: 0.9 }) }));
    });
    await waitFor(() => expect(hook.result.current.history).toHaveLength(1));

    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    await act(async () => {
      resolveTraining(result({ metrics: metric({ testAuc: 0.93 }) }));
    });
    await waitFor(() =>
      expect(hook.result.current.history[0].testAuc).toBe(0.93),
    );
    expect(hook.result.current.history).toHaveLength(1);

    await act(async () => {
      hook.result.current.start({ ...REQUEST, arch: "gin" });
    });
    await act(async () => {
      resolveTraining(result());
    });
    await waitFor(() => expect(hook.result.current.history).toHaveLength(2));
  });

  it("treats a stop before the first epoch as a normal outcome", async () => {
    // Cancelling is not a failure: throwing here would put "training produced no
    // metrics" in the error slot every time someone pressed Stop quickly.
    const hook = await loaded();
    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    await act(async () => {
      resolveTraining(result({ metrics: null }));
    });
    await waitFor(() => expect(hook.result.current.training).toBe(false));
    expect(hook.result.current.trainError).toBeNull();
    expect(hook.result.current.result).toBeNull();
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

  it("keeps the embeddings so the page can score a pair itself", async () => {
    const hook = await loaded();
    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    await act(async () => {
      resolveTraining(result());
    });
    await waitFor(() => expect(hook.result.current.result).not.toBeNull());
    expect(hook.result.current.result?.embeddingDim).toBe(2);
    expect(hook.result.current.result?.embedding).toHaveLength(8);
    expect(hook.result.current.elapsedMs).toBe(1200);
  });

  it("asks the worker to stop, and reuses the one worker across runs", async () => {
    const hook = await loaded();
    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    act(() => {
      hook.result.current.stop();
    });
    expect(cancelSpy).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveTraining(result({ metrics: null }));
    });
    await waitFor(() => expect(hook.result.current.training).toBe(false));
    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    expect(mockCreateWorker).toHaveBeenCalledTimes(1);
  });

  it("cancels and terminates the worker on unmount", async () => {
    const hook = await loaded();
    await act(async () => {
      hook.result.current.start(REQUEST);
    });
    hook.unmount();
    expect(cancelSpy).toHaveBeenCalled();
    expect(terminate).toHaveBeenCalled();
  });
});
