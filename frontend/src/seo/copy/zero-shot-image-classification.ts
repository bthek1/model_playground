import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Type your own labels and let CLIP score the picture against them — no training, no fixed class list.",
  about: [
    "Zero-shot image classification sorts a picture into categories you name yourself, with no training step. You type a list of labels, and a model trained on pairs of images and captions compares the picture with each one and ranks them. Upload a photo, pick a sample or use your webcam, edit the labels, and press the button to score it.",
    "Two models are offered. CLIP ViT-B/32 is the default, a 304 MB download on WebGPU or 154 MB quantized for the CPU, and its scores are a softmax that always sums to one across your labels. SigLIP base/16 (407 MB, or 211 MB) was trained with a sigmoid loss and scores each label on its own. The text and image halves of the model run separately in a Web Worker, so the label embeddings are encoded once and reused, and the page shows how long each half took.",
    "The wording of a label matters, and the page measures by how much: on a still picture it scores your labels as bare words and inside the prompt template 'a photo of a {}', side by side. With CLIP a score is a comparison among the labels you supplied — a high score for 'cat' against 'cat' and 'dog' means more cat than dog, not that a cat is in the picture. The image never leaves your device.",
  ],
  aliases: [
    "zero-shot image classification",
    "CLIP",
    "SigLIP",
    "prompt template",
    "your own labels",
  ],
};
