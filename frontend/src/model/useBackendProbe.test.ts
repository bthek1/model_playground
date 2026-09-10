import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const pickBackend = vi.fn();
vi.mock("./backend", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, pickBackend: () => pickBackend() };
});

const { useBackendProbe } = await import("./useBackendProbe");

beforeEach(() => {
  vi.clearAllMocks();
  pickBackend.mockResolvedValue("webgpu");
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
