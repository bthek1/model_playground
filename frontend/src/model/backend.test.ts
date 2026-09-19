import { afterEach, describe, expect, it, vi } from "vitest";

import {
  asrLoadOpts,
  loadOpts,
  pickBackend,
  supportsShaderF16,
  vlmLoadOpts,
} from "./backend";

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

describe("asrLoadOpts", () => {
  it("keeps the decoder at fp32 on WASM but stays fp16 on WebGPU", () => {
    // The quantized Whisper/Moonshine decoders fail to open a session on the
    // WASM execution provider bundled with @huggingface/transformers 4.2.0.
    expect(asrLoadOpts("wasm")).toEqual({
      device: "wasm",
      dtype: { encoder_model: "q8", decoder_model_merged: "fp32" },
    });
    expect(asrLoadOpts("webgpu")).toEqual({ device: "webgpu", dtype: "fp16" });
  });
});

describe("vlmLoadOpts", () => {
  it("uses 4-bit weights on both backends, with fp16 activations only on the GPU", () => {
    // A VLM is a generative decoder: `loadOpts()`'s fp16 is a 514 MB download
    // for SmolVLM-256M against 189 MB at q4f16, and 3.4 GB against 1.4 GB for
    // a 2B. The `f16` half is dropped on WASM because fp16 activations are a
    // GPU format.
    expect(vlmLoadOpts("webgpu")).toEqual({ device: "webgpu", dtype: "q4f16" });
    expect(vlmLoadOpts("wasm")).toEqual({ device: "wasm", dtype: "q4" });
  });

  it("is not the shared default — a VLM page must ask for it", () => {
    // The whole reason it exists as a named export rather than a literal in the
    // worker: two copies of a precision decision drift silently.
    expect(vlmLoadOpts("webgpu")).not.toEqual(loadOpts("webgpu"));
  });
});

describe("supportsShaderF16", () => {
  it("is true only when the adapter advertises the feature", async () => {
    setGpu({
      requestAdapter: vi.fn().mockResolvedValue({
        features: new Set(["shader-f16"]),
      }),
    });
    expect(await supportsShaderF16()).toBe(true);
  });

  it("is false on an adapter without it — the SwiftShader case", async () => {
    // Measured: such an adapter loads q4f16 weights happily and then fails on
    // the first operator with "Program Gather requires f16 but the device does
    // not support it". Gating on `pickBackend()` alone charges the user the
    // whole download before the page turns out to be dead.
    setGpu({
      requestAdapter: vi.fn().mockResolvedValue({ features: new Set() }),
    });
    expect(await supportsShaderF16()).toBe(false);
  });

  it("is false, not throwing, when there is no adapter or no WebGPU at all", async () => {
    setGpu({ requestAdapter: vi.fn().mockResolvedValue(null) });
    expect(await supportsShaderF16()).toBe(false);

    setGpu({ requestAdapter: vi.fn().mockRejectedValue(new Error("blocked")) });
    expect(await supportsShaderF16()).toBe(false);

    setGpu(undefined);
    expect(await supportsShaderF16()).toBe(false);
  });
});
