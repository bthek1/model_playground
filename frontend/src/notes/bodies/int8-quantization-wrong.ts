import type { NoteBlock } from "../types";

export const body: readonly NoteBlock[] = [
  {
    kind: "p",
    text: "The usual story about 8-bit quantization is that the model gets four times smaller and slightly less accurate. Sometimes it is not slightly less accurate: it gives a different, confident answer. These are cases measured while building this site, with the ONNX exports Transformers.js loads in the browser.",
  },
  { kind: "h2", text: "A tiger becomes a rattlesnake" },
  {
    kind: "p",
    text: "`onnx-community/mobilenetv4_conv_small` at q8 labels a photo of a tiger \"sidewinder, horned rattlesnake\" at 44%. The fp32 build of the same model says \"tiger\" at 62%. MobileNets are built from depthwise-separable convolutions, which are a well-known casualty of int8 quantization because each depthwise filter has few weights and a wide range. A test that checks \"five labels appeared\" passes on both.",
  },
  { kind: "h2", text: "An upscaler worse than not upscaling" },
  {
    kind: "p",
    text: "Swin2SR, a 2× super-resolution model, scored 27.39 dB PSNR at q8 against a ground truth. Plain bicubic interpolation scored 27.55 dB, so the quantized model made the image slightly worse than doing nothing clever. At fp32 it scored 27.80 dB. The q8 output still looks sharp, so you only see this by measuring against a reference.",
  },
  { kind: "h2", text: "Two quantized halves that disagree" },
  {
    kind: "p",
    text: "CLIP and SigLIP publish a combined `model.onnx` and separate `text_model.onnx` and `vision_model.onnx` files, which are quantized separately. At q8, embeddings from the split files differ from the combined model's enough that CLIP's softmax at a scale of 100 turns the difference into a probability gap of about 0.07. At fp32 they agree to six decimal places. Neither q8 path is wrong as such, but you cannot compare numbers across them.",
  },
  { kind: "h2", text: "Results that depend on the batch" },
  {
    kind: "p",
    text: "With `all-MiniLM-L6-v2` at q8, a sentence's embedding depends slightly on what else was in the batch. With the same weights and the same text, embedding the corpus as one batch and the query alone ranked a decoy (0.4556) above the right answer (0.4459). Embedding them all together reversed the order (0.4394 against 0.4538). It is a small effect, but it can reorder close results, so choose test cases using the same call pattern your app uses.",
  },
  { kind: "h2", text: "4-bit is not a size lever for encoders" },
  {
    kind: "p",
    text: "On every encoder measured, the published `model_q4.onnx` was larger than `model_quantized.onnx` (the q8 build). For a DistilBERT sentiment model, q4 was 118.9 MiB and q8 was 64.5 MiB. 4-bit formats pay off for large decoders, not for small encoders.",
  },
  { kind: "h2", text: "How to catch it" },
  {
    kind: "ul",
    items: [
      "Test a known answer on a known input with the real weights: \"tiger\" for the tiger, a PSNR above bicubic for the upscaler. \"A result appeared\" passes on a broken model.",
      "Pin precision per backend when a measurement says to, for example `dtypes: { wasm: \"fp32\" }`, and quote the larger download that results.",
      "Write down whether each pin is a measurement or a precaution. Only a measurement is evidence, and a precaution invites someone to try removing it.",
    ],
  },
  {
    kind: "p",
    text: "Pages on this site that carry these pins: [image classification](/image-classification), [super-resolution](/super-resolution), [zero-shot image classification](/zero-shot-image-classification) and [text ranking](/text-ranking).",
  },
];
