import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline: "Cut the subject out of a photo on your own machine.",
  about: [
    "Background removal here happens in your browser rather than on an online background remover. The page runs an image matting model that predicts an alpha matte — how much of each pixel belongs to the subject — and composites the cut-out in the browser. Choose a file, a sample or a webcam frame, then show the result on a transparent background, white, black or a studio green or blue, view the raw matte, and save the cut-out as a transparent PNG.",
    "The matte is soft on purpose. A strand of hair is partly subject and partly background, so its value is somewhere between the two, and the page blends rather than thresholds. That edge is the part worth judging, which is why the matte can be viewed on its own.",
    "Two models are offered, and the licence decides the default. MODNet, Apache-2.0, is the default: about 6.5 million parameters, a 13 MB download on WebGPU or 7 MB on the CPU. It is a portrait matting model, so it works on people and returns a near-empty matte on most other subjects — the portrait samples come first for that reason. RMBG-1.4 is general-purpose and gives a visibly better edge on hair and fur, at 88 MB or 44 MB, but its licence permits non-commercial use only; picking it puts that restriction on screen beside the choice.",
    "Inference runs in a Web Worker on WebGPU or on the CPU through WASM. The weights are fetched from the model host and cached; the picture is never uploaded.",
  ],
  aliases: [
    "remove image backgrounds",
    "background removal",
    "background remover",
    "image matting",
    "transparent PNG",
    "alpha matte",
    "portrait matting",
  ],
};
