import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { TrainMetrics, TrainResult } from "@/webgpu/linearModel";
import type { TrainingSettings } from "./useLinearTraining";

// --- Mock the dataset loader and the worker client -------------------------

const mockLoadMnistPool = vi.fn();
const mockSliceDataset = vi.fn();

vi.mock("@/lib/mnist", () => ({
  IMAGE_SIZE: 784,
  NUM_CLASSES: 10,
  loadMnistPool: (...args: unknown[]) => mockLoadMnistPool(...args),
  sliceDataset: (...args: unknown[]) => mockSliceDataset(...args),
}));

const mockLoadCachedPool = vi.fn();
const mockSaveCachedPool = vi.fn();

vi.mock("@/lib/mnistCache", () => ({
  loadCachedPool: (...args: unknown[]) => mockLoadCachedPool(...args),
  saveCachedPool: (...args: unknown[]) => mockSaveCachedPool(...args),
}));

const mockCreateWorker = vi.fn(() => ({ terminate: vi.fn() }));
const mockTrainLinearInWorker = vi.fn();

vi.mock("@/webgpu/workerClient", () => ({
  createWebGPUWorker: () => mockCreateWorker(),
  trainLinearInWorker: (...args: unknown[]) => mockTrainLinearInWorker(...args),
}));

const { useLinearTraining } = await import("./useLinearTraining");

// --- Fixtures --------------------------------------------------------------

const POOL = { images: new Float32Array(0), labels: new Uint8Array(0), count: 100 };
const DATASET = {
  xTrain: new Float32Array(0),
  yTrain: new Uint8Array(0),
  xTest: new Float32Array(0),
  yTest: new Uint8Array(0),
  trainSize: 80,
  testSize: 20,
};
const SETTINGS: TrainingSettings = {
  learningRate: 0.5,
  batchSize: 64,
  epochs: 2,
  trainSize: 80,
  testSize: 20,
};

let capturedOnProgress: ((m: TrainMetrics) => void) | null;
let resolveTraining: (r: TrainResult) => void;
let rejectTraining: (e: unknown) => void;
const cancelSpy = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  capturedOnProgress = null;
  mockLoadMnistPool.mockResolvedValue(POOL);
  mockSliceDataset.mockReturnValue(DATASET);
  // Default: nothing cached, writes succeed. Individual tests override.
  mockLoadCachedPool.mockResolvedValue(null);
  mockSaveCachedPool.mockResolvedValue(undefined);
  mockTrainLinearInWorker.mockImplementation(
    (_worker: unknown, _req: unknown, onProgress: (m: TrainMetrics) => void) => {
      capturedOnProgress = onProgress;
      return {
        promise: new Promise<TrainResult>((res, rej) => {
          resolveTraining = res;
          rejectTraining = rej;
        }),
        cancel: cancelSpy,
      };
    },
  );
});

async function loadPool(count = 100) {
  const hook = renderHook(() => useLinearTraining());
  await act(async () => {
    await hook.result.current.loadData(count);
  });
  return hook;
}

describe("useLinearTraining — dataset", () => {
  it("starts idle", () => {
    const { result } = renderHook(() => useLinearTraining());
    expect(result.current.datasetStatus).toBe("idle");
    expect(result.current.training).toBe(false);
    expect(result.current.metrics).toEqual([]);
    expect(result.current.result).toBeNull();
  });

  it("loads the MNIST pool and reports it ready", async () => {
    const { result } = await loadPool(100);
    expect(mockLoadMnistPool).toHaveBeenCalledTimes(1);
    expect(result.current.datasetStatus).toBe("ready");
    expect(result.current.poolCount).toBe(100);
  });

  it("surfaces a dataset load failure", async () => {
    mockLoadMnistPool.mockRejectedValueOnce(new Error("offline"));
    const { result } = renderHook(() => useLinearTraining());
    await act(async () => {
      await result.current.loadData(100);
    });
    expect(result.current.datasetStatus).toBe("error");
    expect(result.current.datasetError).toBe("offline");
  });

  it("persists a freshly downloaded pool to the cache", async () => {
    const { result } = await loadPool(100);
    expect(mockLoadMnistPool).toHaveBeenCalledTimes(1);
    expect(mockSaveCachedPool).toHaveBeenCalledWith(POOL);
    expect(result.current.datasetStatus).toBe("ready");
  });

  it("restores a cached pool on mount without downloading", async () => {
    mockLoadCachedPool.mockResolvedValue(POOL);
    const { result } = renderHook(() => useLinearTraining());
    // Let the mount-effect restore resolve.
    await act(async () => {
      await Promise.resolve();
    });
    expect(result.current.datasetStatus).toBe("ready");
    expect(result.current.poolCount).toBe(100);
    expect(mockLoadMnistPool).not.toHaveBeenCalled();
  });

  it("reuses a cached pool in loadData instead of re-downloading", async () => {
    mockLoadCachedPool.mockResolvedValue(POOL);
    const { result } = await loadPool(100);
    expect(result.current.datasetStatus).toBe("ready");
    expect(result.current.poolCount).toBe(100);
    expect(mockLoadMnistPool).not.toHaveBeenCalled();
    expect(mockSaveCachedPool).not.toHaveBeenCalled();
  });

  it("re-downloads when the cached pool is too small for the request", async () => {
    mockLoadCachedPool.mockResolvedValue({ ...POOL, count: 50 });
    const { result } = renderHook(() => useLinearTraining());
    await act(async () => {
      await result.current.loadData(100);
    });
    expect(mockLoadMnistPool).toHaveBeenCalledTimes(1);
    expect(result.current.poolCount).toBe(100);
  });
});

describe("useLinearTraining — training", () => {
  it("refuses to start before the dataset is loaded", () => {
    const { result } = renderHook(() => useLinearTraining());
    act(() => {
      result.current.start(SETTINGS);
    });
    expect(result.current.trainError).toMatch(/Load the MNIST dataset first/);
    expect(mockTrainLinearInWorker).not.toHaveBeenCalled();
    expect(result.current.training).toBe(false);
  });

  it("surfaces a bad train/test split without dispatching", async () => {
    mockSliceDataset.mockImplementationOnce(() => {
      throw new Error("only 100 are loaded");
    });
    const { result } = await loadPool();
    act(() => {
      result.current.start(SETTINGS);
    });
    expect(result.current.trainError).toMatch(/only 100 are loaded/);
    expect(mockTrainLinearInWorker).not.toHaveBeenCalled();
  });

  it("dispatches a training run with the right shape and hyperparameters", async () => {
    const { result } = await loadPool();
    act(() => {
      result.current.start(SETTINGS);
    });

    expect(mockCreateWorker).toHaveBeenCalledTimes(1);
    expect(mockTrainLinearInWorker).toHaveBeenCalledTimes(1);
    const req = mockTrainLinearInWorker.mock.calls[0][1];
    expect(req.shape).toEqual({ inputDim: 784, numClasses: 10 });
    expect(req.hp).toEqual({ learningRate: 0.5, batchSize: 64, epochs: 2 });
    expect(req.data).toBe(DATASET);
    expect(result.current.training).toBe(true);
  });

  it("streams metrics and stores the final result", async () => {
    const { result } = await loadPool();
    act(() => {
      result.current.start(SETTINGS);
    });

    const metric: TrainMetrics = {
      epoch: 0,
      step: 1,
      totalSteps: 4,
      loss: 2.1,
      trainAcc: null,
      testAcc: null,
    };
    act(() => {
      capturedOnProgress?.(metric);
    });

    const finalResult: TrainResult = {
      testAcc: 0.92,
      weights: new Float32Array(1),
      bias: new Float32Array(1),
    };
    await act(async () => {
      resolveTraining(finalResult);
      await Promise.resolve();
    });

    expect(result.current.result?.testAcc).toBe(0.92);
    expect(result.current.metrics).toContainEqual(metric);
    expect(result.current.training).toBe(false);
  });

  it("surfaces a training error", async () => {
    const { result } = await loadPool();
    act(() => {
      result.current.start(SETTINGS);
    });
    await act(async () => {
      rejectTraining(new Error("device lost"));
      await Promise.resolve();
    });
    expect(result.current.trainError).toBe("device lost");
    expect(result.current.training).toBe(false);
  });

  it("stop() cancels the active run", async () => {
    const { result } = await loadPool();
    act(() => {
      result.current.start(SETTINGS);
    });
    act(() => {
      result.current.stop();
    });
    expect(cancelSpy).toHaveBeenCalledTimes(1);
  });

  it("reuses one worker across successive runs", async () => {
    const { result } = await loadPool();
    act(() => {
      result.current.start(SETTINGS);
    });
    await act(async () => {
      resolveTraining({
        testAcc: 0.5,
        weights: new Float32Array(1),
        bias: new Float32Array(1),
      });
      await Promise.resolve();
    });
    act(() => {
      result.current.start(SETTINGS);
    });
    expect(mockCreateWorker).toHaveBeenCalledTimes(1);
    expect(mockTrainLinearInWorker).toHaveBeenCalledTimes(2);
  });
});
