import { afterEach, describe, expect, it, vi } from "vitest";

import { cachedModels, evictModel, isModelCached } from "./cache";

/** A minimal Cache Storage holding the given request URLs. */
function stubCaches(urls: string[], over: { open?: () => never } = {}) {
  const keys = urls.map((url) => ({ url }));
  const cache = {
    keys: vi.fn(async () => keys),
    // Typed parameter so the assertions below can read back the deleted URL.
    delete: vi.fn(async (request: { url: string }) => Boolean(request)),
  };
  vi.stubGlobal("caches", {
    open: over.open ?? vi.fn(async () => cache),
  });
  return cache;
}

const hub = (id: string, file: string) =>
  `https://huggingface.co/${id}/resolve/main/onnx/${file}`;

afterEach(() => vi.unstubAllGlobals());

describe("cachedModels", () => {
  it("reads model ids out of the cached file URLs", async () => {
    stubCaches([
      hub("onnx-community/whisper-base", "encoder_model_fp16.onnx"),
      hub("onnx-community/whisper-base", "decoder_model_merged_fp16.onnx"),
      hub("Xenova/mms-tts-eng", "model.onnx"),
      "https://example.com/not-a-hub-file.bin",
    ]);
    expect(await cachedModels()).toEqual(
      new Set(["onnx-community/whisper-base", "Xenova/mms-tts-eng"]),
    );
  });

  it("is empty when Cache Storage is unavailable", async () => {
    vi.stubGlobal("caches", undefined);
    expect(await cachedModels()).toEqual(new Set());
  });

  it("is empty — never a rejection — when opening the cache throws", async () => {
    stubCaches([], {
      open: () => {
        throw new Error("blocked in this browsing mode");
      },
    });
    await expect(cachedModels()).resolves.toEqual(new Set());
  });
});

describe("isModelCached", () => {
  it("matches on the model's own path segments, not a substring", async () => {
    stubCaches([hub("onnx-community/whisper-base", "model.onnx")]);
    expect(await isModelCached("onnx-community/whisper-base")).toBe(true);
    // A false positive here would spend the user's bandwidth without asking.
    expect(await isModelCached("whisper-base")).toBe(false);
    expect(await isModelCached("onnx-community/whisper-base-quantized")).toBe(
      false,
    );
  });

  it("is false for an empty id", async () => {
    stubCaches([hub("a/b", "model.onnx")]);
    expect(await isModelCached("")).toBe(false);
  });
});

describe("evictModel", () => {
  it("deletes only that model's entries", async () => {
    const cache = stubCaches([
      hub("a/keep", "model.onnx"),
      hub("a/drop", "encoder.onnx"),
      hub("a/drop", "decoder.onnx"),
    ]);
    await evictModel("a/drop");
    expect(cache.delete).toHaveBeenCalledTimes(2);
    expect(cache.delete.mock.calls.map(([r]) => r.url)).toEqual([
      hub("a/drop", "encoder.onnx"),
      hub("a/drop", "decoder.onnx"),
    ]);
  });

  it("does nothing when there is no cache to evict from", async () => {
    vi.stubGlobal("caches", undefined);
    await expect(evictModel("a/b")).resolves.toBeUndefined();
  });
});
