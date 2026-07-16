import { afterEach, describe, expect, it, vi } from "vitest";

import { loadOpts, pickBackend } from "./backend";

/** Install (or remove) a fake `navigator.gpu` for the duration of a test. */
function setGpu(gpu: unknown) {
  Object.defineProperty(navigator, "gpu", {
    value: gpu,
    configurable: true,
    writable: true,
  });
}

describe("pickBackend", () => {
  afterEach(() => {
    setGpu(undefined);
    vi.restoreAllMocks();
  });

  it("returns webgpu when an adapter is available", async () => {
    setGpu({ requestAdapter: vi.fn().mockResolvedValue({}) });
    expect(await pickBackend()).toBe("webgpu");
  });

  it("falls back to wasm when no adapter is returned", async () => {
    setGpu({ requestAdapter: vi.fn().mockResolvedValue(null) });
    expect(await pickBackend()).toBe("wasm");
  });

  it("falls back to wasm when requestAdapter throws", async () => {
    setGpu({ requestAdapter: vi.fn().mockRejectedValue(new Error("blocked")) });
    expect(await pickBackend()).toBe("wasm");
  });

  it("falls back to wasm when navigator.gpu is absent", async () => {
    setGpu(undefined);
    expect(await pickBackend()).toBe("wasm");
  });
});

describe("loadOpts", () => {
  it("uses fp16 on WebGPU and q8 on WASM", () => {
    expect(loadOpts("webgpu")).toEqual({ device: "webgpu", dtype: "fp16" });
    expect(loadOpts("wasm")).toEqual({ device: "wasm", dtype: "q8" });
  });
});
