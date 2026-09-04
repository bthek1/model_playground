import { afterEach, describe, expect, it, vi } from "vitest";

import { classifyLoadError } from "./errors";

afterEach(() => vi.unstubAllGlobals());

describe("classifyLoadError", () => {
  it("recognises a missing/private model repo", () => {
    const info = classifyLoadError(new Error("Unauthorized access to file"));
    expect(info.cause).toBe("not-found");
    expect(info.message).toMatch(/hugging face hub/i);
    expect(info.suggestsCpu).toBe(false);
  });

  it("recognises a failed network request", () => {
    expect(classifyLoadError(new Error("Failed to fetch")).cause).toBe("offline");
  });

  it("uses navigator.onLine to say *why* the fetch failed", () => {
    vi.stubGlobal("navigator", { onLine: false });
    const info = classifyLoadError(new Error("something opaque"));
    expect(info.cause).toBe("offline");
    expect(info.message).toMatch(/offline/i);
  });

  it("recognises memory exhaustion", () => {
    const info = classifyLoadError(new Error("Array buffer allocation failed"));
    expect(info.cause).toBe("out-of-memory");
    expect(info.hint).toMatch(/smaller model/i);
  });

  it("recognises a GPU failure and offers the CPU", () => {
    const info = classifyLoadError(new Error("GPUDevice device lost"));
    expect(info.cause).toBe("device-lost");
    expect(info.suggestsCpu).toBe(true);
  });

  it("keeps an unrecognised message verbatim", () => {
    const info = classifyLoadError(new Error("qdq_actions.cc:137 boom"));
    expect(info.cause).toBe("unknown");
    // The classified sentence is friendlier; the raw one is what gets pasted
    // into a bug report, so it is never discarded.
    expect(info.raw).toBe("qdq_actions.cc:137 boom");
  });

  it("survives a non-Error throw", () => {
    expect(classifyLoadError("plain string").raw).toBe("plain string");
    expect(classifyLoadError(undefined).raw).toBe("Unknown error");
  });
});
