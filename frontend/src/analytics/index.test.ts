import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// A getter, so each case can switch the build flag the facade reads.
const flags = vi.hoisted(() => ({ enabled: false }));
vi.mock("@/lib/features", () => ({
  get ANALYTICS_ENABLED() {
    return flags.enabled;
  },
  POSTHOG_KEY: "phc_test",
  POSTHOG_HOST: "/ingest",
}));

// Counts how often the SDK module is *imported*, not just used: a disabled or
// opted-out facade must never request the chunk.
const sdk = vi.hoisted(() => ({
  imported: 0,
  capture: vi.fn(),
  optOut: vi.fn(),
  optIn: vi.fn(),
  createClient: vi.fn(),
}));
vi.mock("./client", () => {
  sdk.imported++;
  sdk.createClient.mockImplementation(() => ({
    capture: sdk.capture,
    optOut: sdk.optOut,
    optIn: sdk.optIn,
  }));
  return { createClient: sdk.createClient };
});

import {
  __resetAnalyticsForTests,
  OPT_OUT_KEY,
  pageview,
  setOptedOut,
  track,
} from "./index";

beforeEach(() => {
  __resetAnalyticsForTests();
  localStorage.clear();
  sdk.capture.mockClear();
  sdk.createClient.mockClear();
  flags.enabled = false;
});
afterEach(() => vi.unstubAllGlobals());

const flushed = () => vi.waitFor(() => expect(sdk.createClient).toHaveBeenCalled());

describe("disabled (no key in the build)", () => {
  it("imports nothing and sends nothing", async () => {
    pageview("/asr");
    track("model_load_started", { modelId: "m" });
    await new Promise((r) => setTimeout(r, 20));
    expect(sdk.imported).toBe(0);
    expect(sdk.createClient).not.toHaveBeenCalled();
  });

  it("touches no storage and no network", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    track("feature_used", { feature: "tabular_classification" });
    await new Promise((r) => setTimeout(r, 20));
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(setItem).not.toHaveBeenCalled();
    fetchSpy.mockRestore();
    setItem.mockRestore();
  });
});

describe("enabled", () => {
  beforeEach(() => {
    flags.enabled = true;
    // The WebGPU probe runs after load; a stub keeps it deterministic.
    vi.stubGlobal("navigator", { ...navigator, gpu: undefined });
  });

  it("queues calls made before the SDK resolves and flushes them in order", async () => {
    pageview("/text-classification");
    track("model_load_started", { modelId: "a/b" });
    track("model_load_ready", { modelId: "a/b", backend: "wasm", loadedInMs: 120 });
    await flushed();
    await vi.waitFor(() => expect(sdk.capture).toHaveBeenCalledTimes(4));
    expect(sdk.capture.mock.calls.map((c) => c[0])).toEqual([
      "$pageview",
      "model_load_started",
      "model_load_ready",
      "webgpu_status",
    ]);
    expect(sdk.createClient).toHaveBeenCalledWith("phc_test", "/ingest");
  });

  it("stamps every event with the route pattern and filters the props", async () => {
    pageview("/asr");
    track("model_run_completed", { modelId: "m", runMs: 5, text: "secret" } as never);
    await flushed();
    await vi.waitFor(() => expect(sdk.capture).toHaveBeenCalledTimes(3));
    expect(sdk.capture.mock.calls[1]).toEqual([
      "model_run_completed",
      { modelId: "m", runMs: 5, route: "/asr" },
    ]);
  });

  it("reports webgpu_status once, from the adapter only", async () => {
    pageview("/a");
    pageview("/b");
    await flushed();
    await vi.waitFor(() =>
      expect(sdk.capture).toHaveBeenCalledWith("webgpu_status", {
        webgpuStatus: "unsupported",
        route: "/b",
      }),
    );
    expect(sdk.capture.mock.calls.filter((c) => c[0] === "webgpu_status")).toHaveLength(1);
  });
});

describe("opt-out", () => {
  beforeEach(() => {
    flags.enabled = true;
    vi.stubGlobal("navigator", { ...navigator, gpu: undefined });
  });

  it("opted out before anything loads, the SDK is never imported", async () => {
    const importsBefore = sdk.imported;
    localStorage.setItem(OPT_OUT_KEY, "1");
    pageview("/asr");
    track("model_load_started", { modelId: "m" });
    await new Promise((r) => setTimeout(r, 20));
    expect(sdk.createClient).not.toHaveBeenCalled();
    expect(sdk.imported).toBe(importsBefore);
  });

  it("is stored under its own key and survives a reload", () => {
    setOptedOut(true);
    expect(localStorage.getItem(OPT_OUT_KEY)).toBe("1");
    __resetAnalyticsForTests(); // a reload: module state gone, storage kept
    track("model_load_started", { modelId: "m" });
    expect(sdk.createClient).not.toHaveBeenCalled();
  });

  it("opting out mid-session stops the live client and drops the queue", async () => {
    pageview("/asr");
    await flushed();
    setOptedOut(true);
    expect(sdk.optOut).toHaveBeenCalled();
    const before = sdk.capture.mock.calls.length;
    track("model_load_started", { modelId: "m" });
    expect(sdk.capture.mock.calls.length).toBe(before);
  });
});
