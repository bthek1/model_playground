import { describe, expect, it } from "vitest";

import { CLASSIFIER_MODELS } from "@/audio/classification";
import {
  LARGE_MODEL_BYTES,
  estimateBytes,
  formatBytes,
  sizeEstimate,
} from "./size";
import { TTS_MODELS } from "@/audio/tts";
import { ASR_MODELS } from "@/audio/types";

describe("estimateBytes", () => {
  it("uses 2 bytes per parameter for fp16 and 1 for q8", () => {
    expect(estimateBytes(74, "fp16")).toBe(148e6);
    expect(estimateBytes(74, "q8")).toBe(74e6);
  });
});

describe("formatBytes", () => {
  it("renders MB below a gigabyte and GB above it", () => {
    expect(formatBytes(148e6)).toBe("141 MB");
    expect(formatBytes(2.5 * 1024 ** 3)).toBe("2.5 GB");
  });
});

describe("sizeEstimate", () => {
  it("quotes both backends and keys the warning off the fp16 worst case", () => {
    // Whisper-base: the plan's reference figure (74M params ≈ 150 MB fp16).
    const whisper = sizeEstimate(74);
    expect(whisper.fp16).toBe(148e6);
    expect(whisper.q8).toBe(74e6);
    expect(whisper.label).toBe("≈141 MB on WebGPU · 71 MB on WASM");
    expect(whisper.large).toBe(false);
  });

  it("flags a model whose download passes the threshold", () => {
    const justOver = LARGE_MODEL_BYTES / 2e6 + 1; // params, in millions
    expect(sizeEstimate(justOver).large).toBe(true);
    expect(sizeEstimate(justOver - 2).large).toBe(false);
  });

  it("keys the warning off the larger backend, not just fp16", () => {
    // ASR on WASM keeps the decoder at fp32, so WASM is the heavier download —
    // a warning driven by fp16 alone would miss it.
    const asr = sizeEstimate(74, { webgpu: 146_276_352, wasm: 231_735_296 });
    expect(asr.fp16).toBeLessThan(LARGE_MODEL_BYTES);
    expect(asr.q8).toBeGreaterThan(LARGE_MODEL_BYTES);
    expect(asr.large).toBe(true);
  });

  it("prefers measured bytes over the params estimate", () => {
    const measured = sizeEstimate(74, { wasm: 231_735_296 });
    expect(measured.q8).toBe(231_735_296);
    expect(measured.fp16).toBe(estimateBytes(74, "fp16")); // no override → estimate
  });
});

describe("model catalogues", () => {
  const all = [...ASR_MODELS, ...CLASSIFIER_MODELS, ...TTS_MODELS];

  it("give every model a positive parameter count so the guardrail applies", () => {
    for (const m of all) {
      expect(m.params, m.id).toBeGreaterThan(0);
    }
  });

  it("keeps every shipped audio model inside a browser-sized budget", () => {
    // A tab is not a workstation — nothing here should be a multi-GB download.
    for (const m of all) {
      const s = sizeEstimate(m.params, "bytes" in m ? m.bytes : undefined);
      expect(Math.max(s.fp16, s.q8), m.id).toBeLessThan(1024 ** 3);
    }
  });

  it("fires the guardrail on the heaviest models we actually ship", () => {
    // If nothing trips it, the threshold is set wrong and the warning is dead code.
    const flagged = all.filter(
      (m) => sizeEstimate(m.params, "bytes" in m ? m.bytes : undefined).large,
    );
    expect(flagged.map((m) => m.id)).toEqual(
      expect.arrayContaining([
        "onnx-community/whisper-base",
        "Xenova/clap-htsat-unfused",
      ]),
    );
  });
});
