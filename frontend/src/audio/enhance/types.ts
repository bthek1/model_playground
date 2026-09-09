// Types for the audio-to-audio (speech enhancement) task. Unlike every other
// audio route this one has no Transformers.js pipeline behind it — the payloads
// carry raw samples, and the "model" is an ONNX graph plus a constants file.

import type { ModelProgress, ModelRequest, ModelResponse } from "@/model/types";

import type { Backend } from "@/model/backend";
import type { MeasuredBytes } from "@/model/size";
import { SAMPLE_RATE } from "./deepFilterNet";

export { SAMPLE_RATE };

export interface EnhanceModel {
  id: string;
  label: string;
  hint: string;
  /** Hugging Face repo the graph and its auxiliary constants come from. */
  repo: string;
  /** Parameter count in millions — the shared `ModelPicker` wants it. */
  params: number;
  /**
   * Measured download. Quoted for both backends because it is the same number:
   * the published graph is fp32 and there is no quantized variant, so unlike the
   * Transformers.js routes the WASM path downloads no less than WebGPU.
   */
  bytes: MeasuredBytes;
  /** Native rate. DeepFilterNet3 is 48 kHz — every other audio route is 16 kHz. */
  sampleRate: number;
}

/**
 * The catalogue. One entry today; the shape exists so a second enhancement
 * model does not force a route rewrite.
 */
export const ENHANCE_MODELS: EnhanceModel[] = [
  {
    id: "soniqo/DeepFilterNet3-ONNX",
    label: "DeepFilterNet3",
    hint: "48 kHz speech denoising — the reference real-time model.",
    repo: "soniqo/DeepFilterNet3-ONNX",
    params: 2.3,
    // deepfilter.onnx 8_608_859 + deepfilter-auxiliary.bin 126_976
    bytes: { webgpu: 8_735_835, wasm: 8_735_835 },
    sampleRate: 48000,
  },
];

export const DEFAULT_ENHANCE_MODEL = ENHANCE_MODELS[0].id;

export interface EnhanceResult {
  audio: Float32Array;
  sampleRate: number;
}

export type EnhanceProgress = ModelProgress;

/** Main thread → worker. */
export type EnhanceRequest = ModelRequest<
  { model: string; backend?: Backend },
  { audio: Float32Array }
>;

/** Worker → main thread. */
export type EnhanceResponse = ModelResponse<EnhanceResult>;
