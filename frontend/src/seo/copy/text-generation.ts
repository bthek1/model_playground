import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline: "Run a small language model in your browser and compare greedy decoding with temperature and top-p sampling.",
  about: [
    "A language model generates text one token at a time, and how it picks each token is the decoding strategy. Greedy decoding always takes the most likely token, so the same prompt gives the same text every time, and it falls into repetition loops. Sampling draws from the distribution instead, shaped by temperature, top-p (nucleus sampling) and top-k, and reads better but cannot be repeated. This page makes those settings the controls, with presets for greedy, a long greedy run that loops, ordinary sampling and a hot setting, and can run two settings on the same prompt side by side.",
    "Three models are offered. SmolLM2 360M Instruct is the default and the only one with a CPU path: 273 MB on WebGPU, or 365 MB at 8-bit on WASM, where it writes about one token every 180 ms. GPT-2 (124M) is a 251 MB download that runs on the GPU only, kept because greedy decoding degenerates on it within a sentence. Qwen2.5 0.5B Instruct is 483 MB, GPU only, and the one that writes coherent paragraphs. The GPU models need the shader-f16 feature, and the picker disables them on an adapter without it.",
    "This is not a chatbot and keeps no conversation. Output streams as it is produced, and changing a setting runs nothing until you press generate again. The prompt and the text stay on your device.",
  ],
  aliases: [
    "small LLM",
    "text generation",
    "greedy decoding",
    "temperature",
    "top-p",
    "nucleus sampling",
    "decoding strategy",
    "SmolLM2",
  ],
};
