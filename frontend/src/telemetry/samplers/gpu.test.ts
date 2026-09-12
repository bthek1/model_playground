import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { detectWebGPU } from "@/webgpu/capabilities";
import {
  resetAllocations,
  trackBuffer,
} from "@/webgpu/allocations";
import type { WebGPUCapabilities } from "@/webgpu/types";

import { gpuIdentity, resetGpuIdentity, sampleGpuMemory } from "./gpu";

vi.mock("@/webgpu/capabilities", () => ({ detectWebGPU: vi.fn() }));

const ready: WebGPUCapabilities = {
  status: "ready",
  adapter: {
    vendor: "nvidia",
    architecture: "ampere",
    device: "",
    description: "",
  },
  isFallbackAdapter: false,
  features: ["shader-f16", "timestamp-query"],
  limits: { maxBufferSize: 2 ** 31 },
};

beforeEach(() => {
  vi.clearAllMocks();
  resetGpuIdentity();
  resetAllocations();
  vi.mocked(detectWebGPU).mockResolvedValue(ready);
});

afterEach(() => resetAllocations());

describe("gpuIdentity", () => {
  it("reports the adapter when a device is acquirable", async () => {
    await expect(gpuIdentity()).resolves.toEqual({ status: "ok", value: ready });
  });

  it("probes once however often it is asked — requestAdapter is not free", async () => {
    await gpuIdentity();
    await gpuIdentity();
    await sampleGpuMemory();
    expect(detectWebGPU).toHaveBeenCalledOnce();
  });

  it.each([
    ["unsupported", /doesn't expose WebGPU/],
    ["no-adapter", /no compatible GPU/],
    ["no-device", /no GPU device/],
  ] as const)("explains a %s machine", async (status, reason) => {
    resetGpuIdentity();
    vi.mocked(detectWebGPU).mockResolvedValue({ ...ready, status });
    await expect(gpuIdentity()).resolves.toMatchObject({
      status: "unavailable",
      reason: expect.stringMatching(reason),
    });
  });
});

describe("sampleGpuMemory", () => {
  it("reports zero bytes on an idle GPU — a measurement, not a gap", async () => {
    await expect(sampleGpuMemory()).resolves.toEqual({
      status: "ok",
      value: { liveBytes: 0, peakBytes: 0, buffers: 0, lastPassMs: null },
    });
  });

  it("reports the ledger", async () => {
    trackBuffer({ destroy: vi.fn() } as unknown as GPUBuffer, 4096);
    await expect(sampleGpuMemory()).resolves.toMatchObject({
      status: "ok",
      value: { liveBytes: 4096, buffers: 1 },
    });
  });

  it("is unavailable, with the identity's reason, where there is no GPU", async () => {
    vi.mocked(detectWebGPU).mockResolvedValue({ ...ready, status: "unsupported" });
    trackBuffer({ destroy: vi.fn() } as unknown as GPUBuffer, 4096);
    await expect(sampleGpuMemory()).resolves.toMatchObject({
      status: "unavailable",
    });
  });
});
