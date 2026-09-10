import { describe, expect, it } from "vitest";

import {
  CAPTION_MODELS,
  DEFAULT_CAPTION_MODEL,
  isBoxMode,
  MAX_NEW_TOKENS,
  MODE_HINTS,
  MODE_LABELS,
  type CaptionMode,
} from "./types";

describe("CAPTION_MODELS", () => {
  it("makes the four-mode model the default", () => {
    const florence = CAPTION_MODELS.find((m) => m.id === DEFAULT_CAPTION_MODEL)!;
    expect(florence.modes).toHaveLength(4);
    expect(florence.family).toBe("florence2");
  });

  it("declares only the modes each checkpoint can actually answer", () => {
    // **A task token a model has never seen does not error** — it produces a
    // confident, fluent, unrelated sentence. The UI can only be as honest as
    // these flags, so a model that only captions must say only that.
    const vitgpt2 = CAPTION_MODELS.find((m) => m.family === "vision-encoder-decoder")!;
    expect(vitgpt2.modes).toEqual(["<CAPTION>"]);
  });

  it("gates the WebGPU-only model in the catalogue rather than at load", () => {
    // Four graphs and an autoregressive decoder on WASM is tens of seconds per
    // caption. `useBackendProbe` + `ModelPicker` read this and disable the row.
    const florence = CAPTION_MODELS.find((m) => m.id === DEFAULT_CAPTION_MODEL)!;
    expect(florence.backends).toEqual(["webgpu"]);
    // The CPU-capable one must not be gated, or a machine without a GPU has no
    // model at all on this route.
    const vitgpt2 = CAPTION_MODELS.find((m) => m.family === "vision-encoder-decoder")!;
    expect(vitgpt2.backends).toBeUndefined();
  });

  it("names every ONNX graph it downloads, since neither repo has model.onnx", () => {
    // A Hub-id check hard-coded to `onnx/model.onnx` would look at the wrong
    // file for both of these and pass.
    for (const model of CAPTION_MODELS) {
      expect(model.graphs.length, model.id).toBeGreaterThan(1);
      expect(model.graphs, model.id).not.toContain("model");
      expect(model.bytes.webgpu, model.id).toBeGreaterThan(0);
      expect(model.bytes.wasm, model.id).toBeGreaterThan(0);
    }
  });

  it("only ever declares modes the UI can label and explain", () => {
    for (const model of CAPTION_MODELS) {
      for (const mode of model.modes) {
        expect(MODE_LABELS[mode], mode).toBeTruthy();
        expect(MODE_HINTS[mode], mode).toBeTruthy();
        expect(MAX_NEW_TOKENS[mode], mode).toBeGreaterThan(0);
      }
    }
  });
});

describe("isBoxMode", () => {
  it("routes grounding to the canvas and everything else to prose", () => {
    // Routed by the *mode*, not by sniffing the result: a truncated generation
    // must not be drawn as prose because it happened to parse that way.
    expect(isBoxMode("<OD>")).toBe(true);
    for (const mode of ["<CAPTION>", "<DETAILED_CAPTION>", "<OCR>"] as CaptionMode[]) {
      expect(isBoxMode(mode), mode).toBe(false);
    }
  });
});

describe("MAX_NEW_TOKENS", () => {
  it("keeps a plain caption far cheaper than a detailed one", () => {
    // A caption is a sentence and a detailed one is a paragraph; the cap is
    // what keeps the cheap mode cheap on an autoregressive decoder.
    expect(MAX_NEW_TOKENS["<CAPTION>"]).toBeLessThan(
      MAX_NEW_TOKENS["<DETAILED_CAPTION>"],
    );
  });

  it("gives the two modes that emit many tokens the most room", () => {
    // OCR transcribes everything legible; grounding writes four `<loc_…>`
    // tokens per box. Both truncate visibly at a caption's budget.
    expect(MAX_NEW_TOKENS["<OCR>"]).toBeGreaterThanOrEqual(
      MAX_NEW_TOKENS["<DETAILED_CAPTION>"],
    );
    expect(MAX_NEW_TOKENS["<OD>"]).toBeGreaterThanOrEqual(
      MAX_NEW_TOKENS["<DETAILED_CAPTION>"],
    );
  });
});
