import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Translate English to German, French, Spanish or Chinese, and German or French back into English.",
  about: [
    "Machine translation turns text in one language into another. This page does it one language pair at a time, with Marian models from the opus-mt family, each trained for a single direction: English to German, German to English, English to French, French to English, English to Spanish and English to Chinese. To translate text, type or paste it into the box, or start from a sample written in the source language.",
    "A pair is a model. A Marian checkpoint carries its own direction and takes no language setting, so choosing a direction chooses a download, and going from English–German to German–English means loading a second model. Each pair is 209–223 MB on WebGPU. On the CPU it is 271–289 MB, larger rather than smaller, because these decoders cannot open a quantized session in the browser's WebAssembly runtime and load at full precision there instead.",
    "The alternative would be one multilingual model. The smallest with a browser export, NLLB-200-distilled-600M, is 1.76 GB at the precision this page would load — about eight times a single pair, and past the app's size limit — so it is quoted beside the picker rather than offered. If you need one direction, one specialist is the cheaper choice. The model runs in a Web Worker on your GPU or CPU, and the text is translated on your device, not uploaded.",
  ],
  aliases: [
    "machine translation",
    "translate English to German",
    "English to French",
    "Marian",
    "opus-mt",
    "translate text",
  ],
};
