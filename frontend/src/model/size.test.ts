import { describe, expect, it } from "vitest";

import { CLASSIFIER_MODELS } from "@/audio/classification";
import { IMAGE_CLASSIFIER_MODELS } from "@/vision/classification";
import {
  LARGE_MODEL_BYTES,
  estimateBytes,
  formatBytes,
  sizeEstimate,
} from "./size";
import { TTS_MODELS } from "@/audio/tts";
import { ASR_MODELS } from "@/audio/types";
import { VLM_MODELS } from "@/multimodal/types";

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
  // Every modality, because the guardrail is shared: this module moved out of
  // `audio/` precisely so a second one would not grow its own copy.
  const all = [
    ...ASR_MODELS,
    ...CLASSIFIER_MODELS,
    ...TTS_MODELS,
    ...IMAGE_CLASSIFIER_MODELS,
  ];

  it("give every model a positive parameter count so the guardrail applies", () => {
    for (const m of all) {
      expect(m.params, m.id).toBeGreaterThan(0);
    }
  });

  it("keeps every shipped model inside a browser-sized budget", () => {
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

describe("q4 precision", () => {
  it("gives the 4-bit dtypes a bytes-per-param rather than yielding NaN", () => {
    // `BYTES_PER_PARAM` is keyed by `Dtype`, so widening the type without an
    // entry here would render "NaN MB" on a real page with nothing failing on
    // the way there.
    expect(estimateBytes(256, "q4f16")).toBeGreaterThan(0);
    expect(estimateBytes(256, "q4")).toBeGreaterThan(0);
    expect(Number.isNaN(estimateBytes(256, "q4f16"))).toBe(false);
  });

  it("estimates 4-bit above a naive half-byte, and below q8", () => {
    // A q4 export leaves embeddings, norms and biases at higher precision, so
    // the 4-bit blocks alone under-count.
    expect(estimateBytes(256, "q4f16")).toBeGreaterThan(estimateBytes(256, "q8") / 2);
    expect(estimateBytes(256, "q4f16")).toBeLessThan(estimateBytes(256, "q8"));
  });
});

describe("the VLM catalogue", () => {
  it("quotes measured bytes, because a q4f16 estimate is wrong by 30% on the small one", () => {
    // SmolVLM-256M's `embed_tokens_q4f16.onnx` is 56.8 MB — the same size as
    // its fp16 build, because the embedding table is not 4-bit quantized at
    // all. An estimate that assumes one precision across the model halves it.
    for (const m of VLM_MODELS) {
      expect(m.bytes.webgpu, m.id).toBeGreaterThan(0);
      expect(m.bytes.wasm, m.id).toBeGreaterThan(0);
    }
  });

  it("keeps both entries inside the browser budget the guide sets", () => {
    // ~1 GB, per docs/guides/adding-a-task-page.md §0. Qwen3-VL-2B is 1373 MB
    // at q4f16 and is deliberately absent for exactly this reason.
    for (const m of VLM_MODELS) {
      const s = sizeEstimate(m.params, m.bytes);
      expect(Math.max(s.fp16, s.q8), m.id).toBeLessThan(1024 ** 3);
    }
  });

  it("trips the large-model warning on both, so the gate is never dead code", () => {
    for (const m of VLM_MODELS) {
      expect(sizeEstimate(m.params, m.bytes).large, m.id).toBe(true);
    }
  });
});
