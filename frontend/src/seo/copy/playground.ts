import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "See how every model page here runs machine learning models in the browser: select, load, run, read the output.",
  about: [
    "In-browser inference means the model runs on your own computer, inside the web page, instead of on a server: the weights are downloaded to the browser and the input never leaves the device. This page is the map of how that works across the app. Every task in the sidebar, from speech recognition to text classification, follows the same four steps, and the card here sets them out.",
    "Select picks a model from the task's catalogue, with its download size quoted before a byte is fetched. Load streams the weights in and caches them, and it happens only when you press the button. Run gives the model an input — text, a recording, a file or a matrix — and Output shows the result in whatever form suits it, such as audio to play back, a heatmap or a transcript.",
    "Where it runs depends on the page. Pretrained models go through Transformers.js and ONNX Runtime Web, on WebGPU where the browser offers it and on the CPU through WebAssembly where it does not. The theory pages use hand-written WGSL compute shaders instead. Either way the work happens in a Web Worker, so the page stays responsive. WebGPU inference needs a secure context, so the site is served over HTTPS; the Home dashboard reports what your GPU exposes.",
    "This page loads no model and runs nothing itself: it explains the pattern, and the task pages are where the models are.",
  ],
  aliases: [
    "in-browser inference",
    "WebGPU inference",
    "machine learning models in the browser",
    "Transformers.js",
    "ONNX Runtime Web",
    "WebGPU",
  ],
};
