import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline: "Click anything in the picture and SAM cuts it out.",
  about: [
    "Mask generation cuts an object out of a picture from a prompt as small as one point. This page runs Segment Anything (SAM) in your browser: upload a photo or pick a sample, click on the thing you want, and press the button to get its mask. An Alt-click adds a point that marks what is not part of the object, which is how you tell SAM you meant the wheel rather than the whole car.",
    "SAM is built in two parts. A heavy image encoder runs once per picture, and a small decoder turns each set of points into masks, so after the first encode every new mask is a decode measured in milliseconds, with the time shown beside it. A single point is ambiguous by nature, so all three candidate masks SAM returns are offered, each with the model's own estimate of its quality.",
    "The default is SlimSAM-77, the original SAM pruned to a fifth of its size: 21 MB on WebGPU, 14 MB on the CPU. SAM 2.1 tiny has a hierarchical encoder that does better on small objects, at 78 MB, or 62 MB on the CPU. Both run on your own hardware through WebGPU or WebAssembly, and the picture is not uploaded. SAM-HQ and SAM 3 have no browser export and are not offered.",
  ],
  aliases: [
    "Segment Anything",
    "SAM",
    "mask generation",
    "image encoder",
    "SlimSAM",
    "SAM 2.1",
  ],
};
