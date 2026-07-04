import { afterEach, describe, expect, it, vi } from "vitest";

import { detectWebGPU, isWebGPUSupported } from "./capabilities";

// happy-dom does not implement WebGPU, so navigator.gpu is undefined. These
// tests pin the graceful-degradation path the UI relies on.
describe("webgpu capabilities — unsupported environment", () => {
  it("reports WebGPU as unsupported when navigator.gpu is missing", () => {
    expect(isWebGPUSupported()).toBe(false);
  });

  it("returns an unsupported report without throwing", async () => {
    const caps = await detectWebGPU();
    expect(caps.status).toBe("unsupported");
    expect(caps.adapter).toBeNull();
    expect(caps.features).toEqual([]);
    expect(caps.limits).toEqual({});
  });
});

// The adapter/device paths need a stubbed navigator.gpu. `ready` requires a
// device to actually be acquired — detectWebGPU calls requestDevice() and
// discards the probe device.
describe("webgpu capabilities — with a GPU adapter", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function makeAdapter(overrides: Record<string, unknown> = {}) {
    return {
      info: {
        vendor: "nvidia",
        architecture: "turing",
        device: "rtx",
        description: "desc",
      },
      isFallbackAdapter: false,
      features: new Set(["timestamp-query", "shader-f16"]),
      limits: {
        maxComputeWorkgroupsPerDimension: 65535,
        maxComputeInvocationsPerWorkgroup: 1024,
        maxComputeWorkgroupStorageSize: 49152,
        maxStorageBufferBindingSize: 2147483644,
        maxBufferSize: 4294967292,
        // Not in REPORTED_LIMITS — must be filtered out of the report.
        maxTextureDimension2D: 8192,
      },
      requestDevice: vi.fn(async () => ({ destroy: vi.fn() })),
      ...overrides,
    };
  }

  function stubGpu(requestAdapter: () => Promise<unknown>) {
    vi.stubGlobal("navigator", { gpu: { requestAdapter } });
  }

  it("returns ready with adapter info when a device is acquired", async () => {
    const device = { destroy: vi.fn() };
    const adapter = makeAdapter({ requestDevice: vi.fn(async () => device) });
    stubGpu(async () => adapter);

    const caps = await detectWebGPU();

    expect(caps.status).toBe("ready");
    expect(caps.adapter).toEqual({
      vendor: "nvidia",
      architecture: "turing",
      device: "rtx",
      description: "desc",
    });
    expect(caps.isFallbackAdapter).toBe(false);
    // Features are sorted and surfaced verbatim.
    expect(caps.features).toEqual(["shader-f16", "timestamp-query"]);
    // Only the allow-listed limits are reported.
    expect(caps.limits.maxBufferSize).toBe(4294967292);
    expect(caps.limits.maxTextureDimension2D).toBeUndefined();
    // The probe device is discarded, not leaked.
    expect(device.destroy).toHaveBeenCalledOnce();
  });

  it("propagates a software fallback adapter", async () => {
    stubGpu(async () => makeAdapter({ isFallbackAdapter: true }));
    const caps = await detectWebGPU();
    expect(caps.status).toBe("ready");
    expect(caps.isFallbackAdapter).toBe(true);
  });

  it("returns no-adapter when requestAdapter resolves null", async () => {
    stubGpu(async () => null);
    const caps = await detectWebGPU();
    expect(caps.status).toBe("no-adapter");
    expect(caps.adapter).toBeNull();
  });

  it("returns no-adapter when requestAdapter throws", async () => {
    stubGpu(async () => {
      throw new Error("adapter boom");
    });
    const caps = await detectWebGPU();
    expect(caps.status).toBe("no-adapter");
  });

  it("returns no-device (with adapter info) when requestDevice rejects", async () => {
    const adapter = makeAdapter({
      requestDevice: vi.fn(async () => {
        throw new Error("device lost");
      }),
    });
    stubGpu(async () => adapter);

    const caps = await detectWebGPU();

    expect(caps.status).toBe("no-device");
    // The adapter was found, so its details are still reported.
    expect(caps.adapter?.vendor).toBe("nvidia");
    expect(caps.features).toEqual(["shader-f16", "timestamp-query"]);
  });
});
