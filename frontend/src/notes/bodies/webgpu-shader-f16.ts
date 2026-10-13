import type { NoteBlock } from "../types";

export const body: readonly NoteBlock[] = [
  {
    kind: "p",
    text: "If an fp16 or q4f16 model loads on WebGPU, reports that it is ready, and then fails on the first run with `Program Gather requires f16 but the device does not support it`, the GPU adapter lacks the `shader-f16` feature. Checking for a GPU, or for a working `GPUDevice`, is not enough. Check for the feature, and check before downloading anything.",
  },
  { kind: "h2", text: "Why this is the worst way to fail" },
  {
    kind: "p",
    text: "The weights download and the session opens, because loading half-precision tensors does not need f16 shader arithmetic. The failure comes at the first operator that computes in f16, after the user has waited through a download of hundreds of megabytes, and it repeats on every run. Software adapters such as SwiftShader, which a headless browser or a CI runner without a GPU device gets, do not offer `shader-f16`, and some real GPUs and drivers do not either.",
  },
  { kind: "h2", text: "The check" },
  {
    kind: "code",
    text: `async function supportsShaderF16(): Promise<boolean> {
  if (!navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter?.features.has("shader-f16") ?? false;
  } catch {
    return false;
  }
}

const device = (await supportsShaderF16()) ? "webgpu" : "wasm";`,
  },
  {
    kind: "p",
    text: "Use the answer in two places, and make sure both ask the same question. The picker should disable a model that needs f16 before anything is fetched, and name the missing feature, because \"no GPU\" is confusing on a machine that obviously has one. The code that loads the model should choose its backend with the same check. On this site those two drifted apart: the picker asked about f16 and the worker only asked whether a GPU existed. That was harmless while every f16 model was GPU-only. The first model offered with both an f16 GPU path and a CPU fallback broke it, because the picker correctly enabled it and the worker then sent the user to the adapter that could not run it.",
  },
  { kind: "h2", text: "q4f16 is smaller, but less smaller than you think" },
  {
    kind: "p",
    text: "4-bit weights are worth the trouble for vision-language models: SmolVLM-256M is 189 MB at q4f16 against 514 MB at fp16. Do not estimate the download from the parameter count, though. SmolVLM-256M's `embed_tokens_q4f16.onnx` is 56.8 MB, the same size as its fp16 build, because the embedding table is not 4-bit quantized at all, and that file is 30% of the download. Sum the real files on the Hub. For models that keep weights in external `.onnx_data` files, include those, or a 1.4 GB model measures as 1.2 MB.",
  },
  {
    kind: "p",
    text: "Pages on this site that are gated on `shader-f16`: [image question answering](/image-text-to-text), [video question answering](/video-text-to-text) and the GPU path of [text generation](/text-generation).",
  },
];
