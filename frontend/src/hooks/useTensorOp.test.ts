import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { TensorOpJob, TensorOpResult } from "@/webgpu/types";

const mockRunInWorker = vi.fn();
const mockTerminate = vi.fn();
const mockCreateWorker = vi.fn(() => ({ terminate: mockTerminate }));

vi.mock("@/webgpu/workerClient", () => ({
  createWebGPUWorker: () => mockCreateWorker(),
  runTensorOpInWorker: (worker: Worker, job: TensorOpJob) =>
    mockRunInWorker(worker, job),
}));

const { useTensorOp } = await import("@/hooks/useTensorOp");

const ADD_JOB: TensorOpJob = {
  op: "add",
  a: new Float32Array([1, 2, 3]),
  aRows: 1,
  aCols: 3,
  b: new Float32Array([7, 8, 9]),
  bRows: 1,
  bCols: 3,
};

describe("useTensorOp", () => {
  // Clear in beforeEach (not afterEach): Testing Library's auto-cleanup unmounts
  // the previous test's hook during afterEach — after our hooks run — so an
  // afterEach clear would miss that unmount's terminate() call.
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("starts idle", () => {
    const { result } = renderHook(() => useTensorOp());
    expect(result.current.running).toBe(false);
    expect(result.current.result).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("stores the worker result on success", async () => {
    const opResult: TensorOpResult = {
      data: new Float32Array([8, 10, 12]),
      rows: 1,
      cols: 3,
      gpuTimeMs: 1.5,
    };
    mockRunInWorker.mockResolvedValue(opResult);

    const { result } = renderHook(() => useTensorOp());
    await act(async () => {
      await result.current.run(ADD_JOB);
    });

    expect(result.current.result).toEqual(opResult);
    expect(result.current.error).toBeNull();
    expect(result.current.running).toBe(false);
    // The worker is created lazily and reused (one instance for the run).
    expect(mockCreateWorker).toHaveBeenCalledTimes(1);
    expect(mockRunInWorker).toHaveBeenCalledWith(expect.anything(), ADD_JOB);
  });

  it("surfaces a worker error as a string", async () => {
    mockRunInWorker.mockRejectedValue(new Error("Shapes must match"));

    const { result } = renderHook(() => useTensorOp());
    await act(async () => {
      await result.current.run(ADD_JOB);
    });

    expect(result.current.error).toBe("Shapes must match");
    expect(result.current.result).toBeNull();
    expect(result.current.running).toBe(false);
  });

  it("reset clears the last result", async () => {
    mockRunInWorker.mockResolvedValue({
      data: new Float32Array([1]),
      rows: 1,
      cols: 1,
      gpuTimeMs: 0,
    });

    const { result } = renderHook(() => useTensorOp());
    await act(async () => {
      await result.current.run(ADD_JOB);
    });
    expect(result.current.result).not.toBeNull();

    act(() => {
      result.current.reset();
    });
    expect(result.current.result).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("reuses a single worker across runs", async () => {
    mockRunInWorker.mockResolvedValue({
      data: new Float32Array([1]),
      rows: 1,
      cols: 1,
      gpuTimeMs: 0,
    });

    const { result } = renderHook(() => useTensorOp());
    await act(async () => {
      await result.current.run(ADD_JOB);
    });
    await act(async () => {
      await result.current.run(ADD_JOB);
    });

    expect(mockCreateWorker).toHaveBeenCalledTimes(1);
    expect(mockRunInWorker).toHaveBeenCalledTimes(2);
  });

  it("terminates the worker on unmount", async () => {
    mockRunInWorker.mockResolvedValue({
      data: new Float32Array([1]),
      rows: 1,
      cols: 1,
      gpuTimeMs: 0,
    });

    const { result, unmount } = renderHook(() => useTensorOp());
    await act(async () => {
      await result.current.run(ADD_JOB);
    });

    unmount();
    expect(mockTerminate).toHaveBeenCalledTimes(1);
  });
});
