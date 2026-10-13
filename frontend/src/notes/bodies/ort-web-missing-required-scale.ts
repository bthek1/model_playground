import type { NoteBlock } from "../types";

export const body: readonly NoteBlock[] = [
  {
    kind: "p",
    text: "If a quantized model fails to open on ONNX Runtime Web's WASM backend with an error from `qdq_actions.cc:137` ending in `Missing required scale`, the file is fine and so is your code. The session cannot be built from that particular quantized graph. The fix is to load the graph that fails at full precision while keeping the rest of the model quantized.",
  },
  { kind: "h2", text: "Where it shows up" },
  {
    kind: "p",
    text: "Seen with `@huggingface/transformers` 4.2.0 and the ONNX Runtime Web build it bundles, on the WASM execution provider, loading `q8` (`*_quantized.onnx`) builds. On this site it hit six families: the Whisper and Moonshine decoders, Donut, Marian (`Xenova/opus-mt-*`), DistilBART and GPT-2. Typically the error names a scale tensor with `_merged_0_scale` in it, for example `transformer.wte.weight_merged_0_scale` for GPT-2.",
  },
  {
    kind: "p",
    text: "It is a property of the export, not of the architecture. It looked like an encoder-decoder problem until GPT-2, which is decoder-only, failed the same way. And it is not about tied embeddings: `SmolLM2-360M-Instruct` ties them too, and its `model_quantized.onnx` opens and generates on WASM without trouble. The older `Xenova/*` q8 builds fuse the embedding matmul into a quantized node with a merged scale that the bundled runtime cannot find. Newer exports do not.",
  },
  { kind: "h2", text: "The fix: choose the precision per graph" },
  {
    kind: "p",
    text: "Transformers.js accepts a `dtype` per ONNX file as well as one for the whole model. The encoder quantizes fine, so only the decoder needs full precision:",
  },
  {
    kind: "code",
    text: `import { pipeline } from "@huggingface/transformers";

const translate = await pipeline("translation", "Xenova/opus-mt-en-de", {
  device: "wasm",
  dtype: { encoder_model: "q8", decoder_model_merged: "fp32" },
});`,
  },
  {
    kind: "p",
    text: "This works, at a cost. A Marian pair is 101 MB at a uniform q8 and 271 MB like this, 2.7 times larger. For DistilBART the fallback is 742.8 MB, which is too large to be worth offering, so on this site [summarization](/summarization) with DistilBART is GPU-only and a smaller T5 model provides the CPU path. The error is specific to the WASM backend: DistilBART's q8 build opens on WebGPU without trouble.",
  },
  {
    kind: "p",
    text: "A decoder-only model has nothing to split, because all its weights are in one graph. Either find a newer export of the same model or load it at full precision throughout. GPT-2's only unquantized build is 500 MB, which is why [text generation](/text-generation) here uses GPT-2 on the GPU only and runs SmolLM2 on both.",
  },
  { kind: "h2", text: "Why your tests did not catch it" },
  {
    kind: "ul",
    items: [
      "Unit tests mock the runtime, so no session is ever built.",
      "A check that the files exist on the Hub passes, because they do.",
      "The identical call loads cleanly under `onnxruntime-node`, so a Node script reproduces nothing.",
    ],
  },
  {
    kind: "p",
    text: "Only a real load in a real browser on the WASM backend fails. If you ship a model with a CPU fallback, test that fallback by loading it in a browser, not only the GPU path.",
  },
  {
    kind: "p",
    text: "Pages on this site that use the workaround: [translation](/translation), [speech recognition](/asr) and [summarization](/summarization).",
  },
];
