import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline: "Label every pixel, not just the picture.",
  about: [
    "Image segmentation assigns a class to every pixel of a picture, so the result is a set of masks rather than a single label. Most models here do semantic segmentation: each pixel gets one class, and two people standing side by side fall into one shared mask, because instances are not told apart. Upload a photo, pick a sample or turn on the webcam, then press the button to colour each region and list the classes found. Toggling a class off or moving the opacity slider redraws the masks already in hand without running the model again.",
    "The default is SegFormer-B0 trained on ADE20K, 150 everyday scene classes in a 7.9 MB download on WebGPU, or 4.4 MB on the CPU. Face parsing splits a portrait into 19 parts such as skin, eyes, lips and hair (172 MB, or 89 MB on the CPU). A SegFormer-B2 clothes model labels garments on a person at 110 MB on either backend, since its repository publishes full precision only. DETR panoptic covers things and stuff together at 87 MB, the only panoptic segmentation model here with a browser export.",
    "Mask2Former, OneFormer and EoMT publish no ONNX export, so they are not offered. To cut out a single object by clicking on it, Segment Anything on the Mask Generation page is the better fit. Everything runs in a Web Worker through WebGPU or WebAssembly, and the picture stays on your computer.",
  ],
  aliases: [
    "image segmentation",
    "semantic segmentation",
    "panoptic segmentation",
    "SegFormer",
    "ADE20K",
    "face parsing",
  ],
};
