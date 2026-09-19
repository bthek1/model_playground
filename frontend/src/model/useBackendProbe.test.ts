import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const pickBackend = vi.fn();
const supportsShaderF16 = vi.fn();
vi.mock("./backend", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    pickBackend: () => pickBackend(),
    supportsShaderF16: () => supportsShaderF16(),
  };
});

const { useBackendProbe } = await import("./useBackendProbe");

beforeEach(() => {
  vi.clearAllMocks();
  pickBackend.mockResolvedValue("webgpu");
  supportsShaderF16.mockResolvedValue(true);
});

describe("useBackendProbe", () => {
  it("starts undecided, which is not the same as 'no GPU'", () => {
    // `null` means "ask again in 10 ms". A picker that read it as WASM would
    // grey out every WebGPU model for a frame on each page load, which reads as
    // "this machine cannot run it".
    // Definite assignment, not `| null`: TypeScript narrows a nullable assigned
    // inside a Promise executor back down to `null`, and the optional call then
    // has no signature to call.
    let pending!: (b: string) => void;
    pickBackend.mockReturnValue(new Promise((resolve) => (pending = resolve)));

    const { result } = renderHook(() => useBackendProbe());
    expect(result.current).toBeNull();
    pending("webgpu");
  });

  it("reports the backend a load would resolve to", async () => {
    const { result } = renderHook(() => useBackendProbe());
    await waitFor(() => expect(result.current).toBe("webgpu"));
  });

  it("reports wasm on a machine with no usable adapter", async () => {
    pickBackend.mockResolvedValue("wasm");
    const { result } = renderHook(() => useBackendProbe());
    await waitFor(() => expect(result.current).toBe("wasm"));
  });

  it("probes once per mount, not once per render", async () => {
    // It is a `requestAdapter()` call, not a download — but it is still a
    // permission-shaped question, and asking it on every render is noise.
    const { result, rerender } = renderHook(() => useBackendProbe());
    await waitFor(() => expect(result.current).toBe("webgpu"));
    rerender();
    rerender();
    expect(pickBackend).toHaveBeenCalledTimes(1);
  });

  it("does not set state after unmount", async () => {
    // The probe outlives a fast navigation away from the page.
    let resolve!: (b: string) => void;
    pickBackend.mockReturnValue(new Promise((r) => (resolve = r)));

    const { unmount } = renderHook(() => useBackendProbe());
    unmount();

    const warn = vi.spyOn(console, "error").mockImplementation(() => {});
    resolve("webgpu");
    await Promise.resolve();
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });
});

describe("useBackendProbe — requireShaderF16", () => {
  it("reports wasm when the GPU cannot do f16 in shaders", async () => {
    // Not pedantry: such an adapter loads q4f16 weights and then fails on the
    // first operator, so for a q4f16 model the WebGPU path does not exist. The
    // honest answer to "what would this load on" is therefore wasm — and the
    // catalogue's `backends: ["webgpu"]` then disables the row *before* the
    // download rather than after it.
    supportsShaderF16.mockResolvedValue(false);
    const { result } = renderHook(() =>
      useBackendProbe({ requireShaderF16: true }),
    );
    await waitFor(() => expect(result.current).toBe("wasm"));
  });

  it("reports webgpu when the GPU can", async () => {
    supportsShaderF16.mockResolvedValue(true);
    const { result } = renderHook(() =>
      useBackendProbe({ requireShaderF16: true }),
    );
    await waitFor(() => expect(result.current).toBe("webgpu"));
  });

  it("does not ask about f16 unless a caller needs it", async () => {
    const { result } = renderHook(() => useBackendProbe());
    await waitFor(() => expect(result.current).toBe("webgpu"));
    expect(supportsShaderF16).not.toHaveBeenCalled();
  });

  it("does not ask about f16 on a machine with no GPU at all", async () => {
    // There is nothing to refine: wasm is already the answer.
    pickBackend.mockResolvedValue("wasm");
    const { result } = renderHook(() =>
      useBackendProbe({ requireShaderF16: true }),
    );
    await waitFor(() => expect(result.current).toBe("wasm"));
    expect(supportsShaderF16).not.toHaveBeenCalled();
  });
});
