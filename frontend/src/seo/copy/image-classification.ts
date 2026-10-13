import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Label a photo with the 1,000 ImageNet classes, entirely in your browser.",
  about: [
    "Image classification is how a model names what a picture shows. To classify a photo here, upload one, drop it onto the page or pick a sample, press the button, and read back the five most likely of the thousand ImageNet labels, each with its score.",
    "Three models are offered. ViT-Base/16 is the transformer baseline, a 174 MB download on WebGPU or 88 MB quantized for the CPU. ResNet-50 is the CNN baseline at 51 MB, or 26 MB. MobileNetV4 Small loads almost at once: 7.6 MB on WebGPU, and 15 MB on the CPU, where it runs at full fp32 precision because its 8-bit export labelled a tiger as a rattlesnake. Each runs in a Web Worker, on your GPU through WebGPU or on your CPU through WebAssembly when there is no GPU to use.",
    "The page shows the top five rather than a single answer, and prints the margin between the first and second label, because a near-tie is the result worth seeing. An ImageNet classifier only knows those thousand categories: shown anything else, it names the closest one it has, and the score can still look sure of itself. The photo is never uploaded, and the weights are fetched once and then cached by the browser.",
  ],
  aliases: [
    "image classification",
    "ImageNet classifier",
    "classify a photo",
    "ResNet-50",
    "MobileNetV4",
  ],
};
