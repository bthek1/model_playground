import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Ask a question about a passage and watch the model point at the answer — entirely in your browser, on your GPU (WebGPU) or CPU (WASM).",
  about: [
    "Extractive question answering means the model does not write an answer; it picks the stretch of your passage most likely to hold one. Paste a passage, ask a question about a text you already have, and the page marks the answer where it sits in the passage, with its score beside it. Seeing the span in place makes a wrong answer easy to spot, which a quoted string on its own would not.",
    "The reader is DistilBERT fine-tuned on SQuAD 1.1, a set of questions about Wikipedia paragraphs: a 131 MB download on WebGPU, or 66 MB quantized for the CPU, run in a Web Worker on your own machine. It is the only extractive reader with an ONNX export the browser can load. The highlight is worked out by mapping the tokenizer's pieces back onto your own characters, so a hyphenated answer is marked exactly as you wrote it, and when that mapping fails the page quotes the answer rather than guessing at a range.",
    "This model cannot abstain. It always returns a span, even when the passage contains no answer, and the page says so before the first result appears. One sample shows it: asked who won the 1998 World Cup about a passage on the Eiffel Tower, it answers Gustave Eiffel with a score of 0.94, so a high score is not evidence that the question was answerable. The SQuAD 2.0 models that can decline have no ONNX export. Your passage and question are not uploaded.",
  ],
  aliases: [
    "extractive question answering",
    "question answering",
    "ask a question about a text",
    "SQuAD",
    "DistilBERT",
  ],
};
