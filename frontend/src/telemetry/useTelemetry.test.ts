import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { watchAllocations } from "@/webgpu/allocations";

import { reportDownload, reportInflight, resetActivity } from "./activity";
import { resetGpuIdentity } from "./samplers/gpu";
import { sampleStorage } from "./samplers/storage";
import { DEFAULT_INTERVAL_MS, STORAGE_EVERY_TICKS, useTelemetry } from "./useTelemetry";

vi.mock("@/webgpu/capabilities", () => ({
  detectWebGPU: vi.fn(async () => ({
    status: "ready",
    adapter: { vendor: "nvidia", architecture: "ampere", device: "", description: "" },
    isFallbackAdapter: false,
    features: [],
    limits: {},
  })),
}));

vi.mock("@/webgpu/allocations", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/webgpu/allocations")>();
  return { ...actual, watchAllocations: vi.fn() };
});

vi.mock("./samplers/storage", () => ({
  sampleStorage: vi.fn(async () => ({
    status: "ok" as const,
    value: { usageBytes: 100, quotaBytes: 1000, cachedModels: 3 },
  })),
}));

/** Drain the microtask queue — each awaited sampler needs a turn. */
async function settle() {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

/**
 * Advance the fake clock by one or more intervals, settling in between.
 * The settle *before* advancing matters: `advanceTimersByTime` fires every due
 * callback synchronously without yielding, so without it the in-flight sample
 * would still be running and the hook would (correctly) skip the next tick.
 */
async function tickBy(ms: number) {
  await act(async () => {
    await settle();
    vi.advanceTimersByTime(ms);
    await settle();
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers();
  resetActivity();
  resetGpuIdentity();
});

afterEach(() => {
  vi.useRealTimers();
  resetActivity();
});

describe("useTelemetry", () => {
  it("samples nothing while the panel is closed", async () => {
    const { result } = renderHook(() => useTelemetry(false));
    await tickBy(DEFAULT_INTERVAL_MS * 5);

    expect(result.current.tick).toBe(0);
    expect(sampleStorage).not.toHaveBeenCalled();
    expect(watchAllocations).not.toHaveBeenCalledWith(true);
  });

  it("takes its first sample immediately on open, so the panel is never blank", async () => {
    const { result } = renderHook(() => useTelemetry(true));
    await act(settle);
    expect(result.current.tick).toBe(1);
    expect(watchAllocations).toHaveBeenCalledWith(true);
  });

  it("keeps sampling on the interval and grows the history", async () => {
    const { result } = renderHook(() => useTelemetry(true));
    await tickBy(DEFAULT_INTERVAL_MS);
    await tickBy(DEFAULT_INTERVAL_MS);
    await tickBy(DEFAULT_INTERVAL_MS);

    expect(result.current.tick).toBe(4);
    expect(result.current.lagHistory).toHaveLength(4);
    expect(result.current.busyHistory).toHaveLength(4);
  });

  it("skips a tick rather than queueing behind a sample still in flight", async () => {
    // Eight interval callbacks fire here without a single microtask in
    // between, so the first sample is still awaiting its samplers throughout.
    // Queueing them would leave the panel reporting a machine eight seconds
    // stale; skipping keeps every published sample current.
    const { result } = renderHook(() => useTelemetry(true));
    await act(async () => {
      vi.advanceTimersByTime(DEFAULT_INTERVAL_MS * 8);
      await settle();
    });
    expect(result.current.tick).toBe(1);
    expect(sampleStorage).toHaveBeenCalledOnce();
    expect(result.current.storage).not.toBeNull();
  });

  it("walks the model cache rarely, not every tick", async () => {
    renderHook(() => useTelemetry(true));
    for (let i = 0; i < STORAGE_EVERY_TICKS - 2; i++) {
      await tickBy(DEFAULT_INTERVAL_MS);
    }
    expect(sampleStorage).toHaveBeenCalledOnce();

    await tickBy(DEFAULT_INTERVAL_MS);
    expect(sampleStorage).toHaveBeenCalledTimes(2);
  });

  it("stops sampling, and stops the realms publishing, when the panel closes", async () => {
    const { result, rerender } = renderHook(
      ({ active }: { active: boolean }) => useTelemetry(active),
      { initialProps: { active: true } },
    );
    await tickBy(DEFAULT_INTERVAL_MS * 2);
    expect(result.current.tick).toBeGreaterThan(1);

    rerender({ active: false });
    expect(watchAllocations).toHaveBeenCalledWith(false);
    expect(result.current.tick).toBe(0);

    const callsAfterClose = vi.mocked(sampleStorage).mock.calls.length;
    await tickBy(DEFAULT_INTERVAL_MS * 5);
    expect(result.current.tick).toBe(0);
    expect(sampleStorage).toHaveBeenCalledTimes(callsAfterClose);
  });

  it("stops while the tab is hidden and resumes when it comes back", async () => {
    const hidden = vi.spyOn(document, "hidden", "get").mockReturnValue(false);
    const { result } = renderHook(() => useTelemetry(true));
    await tickBy(DEFAULT_INTERVAL_MS);
    expect(result.current.tick).toBe(2);

    hidden.mockReturnValue(true);
    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"));
    });
    expect(result.current.tick).toBe(0);

    await tickBy(DEFAULT_INTERVAL_MS * 3);
    expect(result.current.tick).toBe(0);

    hidden.mockReturnValue(false);
    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"));
      await settle();
    });
    expect(result.current.tick).toBe(1);
    hidden.mockRestore();
  });

  it("leaves no timer or frame behind on unmount", async () => {
    const clearInterval = vi.spyOn(globalThis, "clearInterval");
    const cancelFrame = vi.spyOn(globalThis, "cancelAnimationFrame");
    const { unmount } = renderHook(() => useTelemetry(true));
    await tickBy(DEFAULT_INTERVAL_MS);

    unmount();
    expect(clearInterval).toHaveBeenCalled();
    expect(cancelFrame).toHaveBeenCalled();
    expect(watchAllocations).toHaveBeenLastCalledWith(false);
  });

  it("reports an inference in flight as busy time", async () => {
    const { result } = renderHook(() => useTelemetry(true));
    await tickBy(DEFAULT_INTERVAL_MS);

    reportInflight("asr", 1, Date.now());
    await tickBy(DEFAULT_INTERVAL_MS);
    expect(result.current.cpu.busyFraction).toBeGreaterThan(0);

    reportInflight("asr", 0, Date.now());
    await tickBy(DEFAULT_INTERVAL_MS);
    expect(result.current.cpu.busyFraction).toBe(0);
  });

  it("derives download throughput from the byte aggregate", async () => {
    reportDownload("tts", { loadedBytes: 0, totalBytes: 1_000_000 });
    const { result } = renderHook(() => useTelemetry(true));
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    // First sample of a download has no previous point to difference against.
    expect(result.current.download).toMatchObject({ bytesPerSecond: null });

    reportDownload("tts", { loadedBytes: 500_000, totalBytes: 1_000_000 });
    await tickBy(DEFAULT_INTERVAL_MS);
    expect(result.current.download?.bytesPerSecond).toBeCloseTo(500_000, -2);

    reportDownload("tts", null);
    await tickBy(DEFAULT_INTERVAL_MS);
    expect(result.current.download).toBeNull();
  });
});
