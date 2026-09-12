import { afterEach, describe, expect, it, vi } from "vitest";

import { deviceMemoryGiB, sampleMemory } from "./memory";

type WithMemory = Performance & { memory?: unknown };

afterEach(() => {
  delete (performance as WithMemory).memory;
  vi.unstubAllGlobals();
});

describe("sampleMemory", () => {
  it("reads the Chrome heap when it is there", () => {
    (performance as WithMemory).memory = {
      usedJSHeapSize: 12_000_000,
      totalJSHeapSize: 20_000_000,
      jsHeapSizeLimit: 4_000_000_000,
    };
    const metric = sampleMemory();
    expect(metric).toEqual({
      status: "ok",
      value: {
        usedBytes: 12_000_000,
        totalBytes: 20_000_000,
        limitBytes: 4_000_000_000,
      },
    });
  });

  it("is unavailable with a reason where performance.memory is absent", () => {
    const metric = sampleMemory();
    expect(metric.status).toBe("unavailable");
    expect(metric).toMatchObject({ reason: expect.stringContaining("Chrome") });
  });

  it("does not trust a stub that is present but wrong shaped", () => {
    (performance as WithMemory).memory = {};
    expect(sampleMemory().status).toBe("unavailable");
  });

  it("never reports zero for unknown — the failure mode this type exists to stop", () => {
    const metric = sampleMemory();
    expect(metric).not.toMatchObject({ status: "ok" });
  });
});

describe("deviceMemoryGiB", () => {
  it("reports the coarse class Chrome gives", () => {
    vi.stubGlobal("navigator", { deviceMemory: 8 });
    expect(deviceMemoryGiB()).toEqual({ status: "ok", value: 8 });
  });

  it("is unavailable in a browser that withholds it", () => {
    vi.stubGlobal("navigator", {});
    expect(deviceMemoryGiB()).toMatchObject({
      status: "unavailable",
      reason: expect.stringContaining("device memory"),
    });
  });

  it("rejects a zero or non-numeric value", () => {
    vi.stubGlobal("navigator", { deviceMemory: 0 });
    expect(deviceMemoryGiB().status).toBe("unavailable");
    vi.stubGlobal("navigator", { deviceMemory: "8" });
    expect(deviceMemoryGiB().status).toBe("unavailable");
  });
});
