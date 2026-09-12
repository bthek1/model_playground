import { afterEach, describe, expect, it, vi } from "vitest";

import { sampleStorage } from "./storage";

vi.mock("@/model/cache", () => ({
  cachedModels: vi.fn(async () => new Set(["onnx-community/whisper", "Xenova/modnet"])),
}));

afterEach(() => vi.unstubAllGlobals());

describe("sampleStorage", () => {
  it("reports usage and quota", async () => {
    vi.stubGlobal("navigator", {
      storage: { estimate: async () => ({ usage: 1_000, quota: 10_000 }) },
    });
    await expect(sampleStorage()).resolves.toEqual({
      status: "ok",
      value: { usageBytes: 1_000, quotaBytes: 10_000, cachedModels: null },
    });
  });

  it("counts cached models only when asked — the walk is not free", async () => {
    vi.stubGlobal("navigator", {
      storage: { estimate: async () => ({ usage: 1, quota: 2 }) },
    });
    const metric = await sampleStorage({ countModels: true });
    expect(metric).toMatchObject({ status: "ok", value: { cachedModels: 2 } });
  });

  it("is unavailable where the API is missing", async () => {
    vi.stubGlobal("navigator", {});
    await expect(sampleStorage()).resolves.toMatchObject({
      status: "unavailable",
      reason: expect.stringContaining("storage estimate"),
    });
  });

  it("is unavailable, with the private-browsing reason, when the estimate rejects", async () => {
    vi.stubGlobal("navigator", {
      storage: {
        estimate: async () => {
          throw new Error("denied");
        },
      },
    });
    await expect(sampleStorage()).resolves.toMatchObject({
      status: "unavailable",
      reason: expect.stringContaining("private browsing"),
    });
  });

  it("does not report a partial estimate as zero", async () => {
    vi.stubGlobal("navigator", {
      storage: { estimate: async () => ({ usage: undefined, quota: 10 }) },
    });
    expect((await sampleStorage()).status).toBe("unavailable");
  });
});
