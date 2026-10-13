import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline: "Ask a picture a question with a short answer.",
  about: [
    "Visual question answering (VQA) means answering a natural-language question about an image: what animal is this, how many people are there, what colour is the main object. Pick a sample or upload a photo, type a question or tap a preset, and choose whether to ask for a one-word answer. With the toggle on, the page adds \"Answer in one word.\" to your question and caps the reply at 16 tokens; with it off, the question goes as typed, with room for 128. The exact prompt is shown before you press Generate and printed above the answer afterwards, and flipping the toggle runs nothing until you do.",
    "Classical VQA models such as ViLT and BLIP-VQA treat the task as choosing from a fixed list of answers, and neither publishes an ONNX export, so neither can run in a browser. This page prompts a general vision-language model instead: SmolVLM 256M (189 MB) or SmolVLM 500M (358 MB), the same checkpoints as the Image Text to Text page, so if you have loaded one there it is already cached. They run on WebGPU only, on a GPU whose adapter supports shader-f16.",
    "A short answer is not a sure one: these models answer whether or not they can see what was asked about. A question that already asks for brevity is not instructed a second time, because a small model given the same order twice tends to answer the instruction instead. Your photo is processed on your own device and never uploaded.",
  ],
  aliases: [
    "visual question answering",
    "VQA",
    "ask a picture a question",
    "one-word answer",
    "SmolVLM",
  ],
};
