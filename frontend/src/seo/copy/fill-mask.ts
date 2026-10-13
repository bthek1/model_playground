import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline: "Take a word out of a sentence and ask the model to put it back — entirely in your browser, on your GPU (WebGPU) or CPU (WASM).",
  about: [
    "Fill-mask is masked language modelling: hide one word behind a mask token, like a fill in the blank exercise, and the model ranks the words it thinks belong there, with a probability for each. It is the objective BERT-style encoders were pretrained on, so the predictions read back what was common in their training text. This page shows the full ranked list rather than a single answer, and runs paired prompts that differ by one word side by side, as a way to probe bias in what a model learned.",
    "Four models from three tokenizer families are offered. BERT base (uncased) is 219 MB on WebGPU or 111 MB on the CPU; DistilBERT base is 134 MB and 68 MB; RoBERTa base is 250 MB and 126 MB; ModernBERT base is 300 MB and 151 MB. BERT, DistilBERT and ModernBERT use [MASK] as their mask token and RoBERTa uses <mask>, so the page inserts the right token with a button and rewrites it in place when you switch model.",
    "The page refuses a sentence with more than one mask, because the underlying pipeline would fill the first and silently drop the rest. Its answers are evidence about a training corpus, not facts about the world: DistilBERT does not know the capital of France, while BERT and ModernBERT answer Paris. Everything runs in a Web Worker and the sentence is never sent anywhere.",
  ],
  aliases: [
    "fill-mask",
    "masked language model",
    "BERT",
    "RoBERTa",
    "mask token",
    "fill in the blank",
  ],
};
