import { describe, expect, it } from "vitest";

import { ALLOWED_PROPS, classifyError, sanitize } from "./schema";

describe("sanitize — the closed allowlist (#60)", () => {
  it("drops every key that is not on the list", () => {
    expect(
      sanitize({ modelId: "Xenova/distilbert", text: "secret", prompt: "x", columns: ["a"] }),
    ).toEqual({ modelId: "Xenova/distilbert" });
  });

  it("drops prose even under an allowed key — a sentence is not an identifier", () => {
    expect(sanitize({ task: "my private sentence", family: "forest" })).toEqual({
      family: "forest",
    });
  });

  it("keeps identifier-shaped values: catalogue ids, route patterns, enums", () => {
    expect(
      sanitize({
        modelId: "onnx-community/whisper-tiny.en",
        route: "/text-classification",
        backend: "webgpu",
        feature: "tabular_classification",
      }),
    ).toEqual({
      modelId: "onnx-community/whisper-tiny.en",
      route: "/text-classification",
      backend: "webgpu",
      feature: "tabular_classification",
    });
  });

  it("rounds numbers, drops non-finite ones, keeps booleans", () => {
    expect(sanitize({ runMs: 12.7, loadedInMs: Number.NaN, elapsedMs: Infinity, retry: true })).toEqual({
      runMs: 13,
      retry: true,
    });
  });

  it("drops objects and arrays, which could smuggle anything", () => {
    expect(sanitize({ modelId: { a: 1 } as unknown, family: ["x"] as unknown })).toEqual({});
  });

  it("has no key that could carry user content", () => {
    for (const banned of ["text", "input", "prompt", "label", "column", "file", "message", "error", "url"]) {
      expect(ALLOWED_PROPS).not.toContain(banned);
    }
  });
});

describe("classifyError — a category, never the message", () => {
  it.each([
    ["Failed to fetch", "network"],
    ["Could not locate file: onnx/model_fp16.onnx (404)", "not_found"],
    ["RangeError: Array buffer allocation failed", "out_of_memory"],
    ["Program Gather requires f16 but the device does not support it", "gpu_feature_missing"],
    ["WebGPU device was lost", "gpu_unavailable"],
    ["qdq_actions.cc:137 Missing required scale", "session_create"],
    ["Worker terminated", "worker_terminated"],
    ["Load cancelled", "cancelled"],
    ["something else entirely", "unknown"],
  ])("%s → %s", (message, kind) => {
    expect(classifyError(new Error(message))).toBe(kind);
  });

  it("an error quoting the user's own text leaves only its category", () => {
    const kind = classifyError(new Error('Failed to fetch "/home/me/payroll-2026.csv"'));
    expect(kind).toBe("network");
    expect(kind).not.toMatch(/payroll/);
  });
});
