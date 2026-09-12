import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// `device.ts` memoises its device for the lifetime of the realm, so every test
// here re-imports it after `vi.resetModules()` to get a fresh one. The class is
// taken from the same import for the same reason: a reset module has a new
// class identity, and `instanceof` against the stale one fails.

/** A device whose `lost` promise never settles, like a healthy one. */
function fakeDevice(features: string[] = []) {
  return {
    features: new Set(features),
    lost: new Promise(() => {}),
  } as unknown as GPUDevice;
}

function stubAdapter({
  features = ["timestamp-query", "shader-f16"],
  requestDevice = vi.fn(async () => fakeDevice()),
}: {
  features?: string[];
  requestDevice?: (descriptor?: GPUDeviceDescriptor) => Promise<GPUDevice>;
} = {}) {
  const adapter = { features: new Set(features), requestDevice };
  vi.stubGlobal("navigator", { gpu: { requestAdapter: async () => adapter } });
  return adapter;
}

beforeEach(() => vi.resetModules());
afterEach(() => vi.unstubAllGlobals());

describe("getGPUDevice", () => {
  it("asks for the optional features the adapter advertises", async () => {
    const requestDevice = vi.fn(async () => fakeDevice(["timestamp-query"]));
    stubAdapter({ requestDevice });

    const { getGPUDevice: get } = await import("./device");
    await get();

    expect(requestDevice).toHaveBeenCalledWith({
      requiredFeatures: ["timestamp-query"],
    });
  });

  it("asks for nothing extra when the adapter advertises nothing we want", async () => {
    const requestDevice = vi.fn(async () => fakeDevice());
    stubAdapter({ features: ["shader-f16"], requestDevice });

    const { getGPUDevice: get } = await import("./device");
    await get();

    expect(requestDevice).toHaveBeenCalledWith();
  });

  it("falls back to a bare device rather than losing the GPU over a diagnostic", async () => {
    const requestDevice = vi
      .fn<(descriptor?: GPUDeviceDescriptor) => Promise<GPUDevice>>()
      .mockRejectedValueOnce(new Error("feature not granted"))
      .mockResolvedValueOnce(fakeDevice());
    stubAdapter({ requestDevice });

    const { getGPUDevice: get } = await import("./device");
    await expect(get()).resolves.toBeDefined();
    expect(requestDevice).toHaveBeenCalledTimes(2);
    expect(requestDevice).toHaveBeenLastCalledWith();
  });

  it("memoises the device for the realm", async () => {
    const requestDevice = vi.fn(async () => fakeDevice());
    stubAdapter({ requestDevice });

    const { getGPUDevice: get } = await import("./device");
    const [a, b] = await Promise.all([get(), get()]);
    expect(a).toBe(b);
    expect(requestDevice).toHaveBeenCalledOnce();
  });

  it("throws where WebGPU is absent", async () => {
    vi.stubGlobal("navigator", {});
    const { getGPUDevice: get, WebGPUUnavailableError } = await import("./device");
    await expect(get()).rejects.toBeInstanceOf(WebGPUUnavailableError);
  });

  it("throws when no adapter is offered", async () => {
    vi.stubGlobal("navigator", {
      gpu: { requestAdapter: async () => null },
    });
    const { getGPUDevice: get } = await import("./device");
    await expect(get()).rejects.toThrow(/No WebGPU adapter/);
  });
});
