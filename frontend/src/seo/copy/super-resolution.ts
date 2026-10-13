import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Double a picture's resolution on your own machine, one overlapping tile at a time.",
  about: [
    "Super-resolution predicts the detail a larger version of an image would have, rather than stretching the pixels it already has. This page runs Swin2SR classical x2, an AI image upscaler: use it to upscale a photo to twice its width and height. Upload a picture or pick a sample; a larger source is scaled down to 512 pixels on its long side first, because the run time grows with the area. The result can be saved as a PNG.",
    "The model has 12 million parameters: a 32 MB download on WebGPU, or 54 MB on the CPU, where it runs at full fp32 precision. That is a measurement rather than caution — at 8 bits the upscale scored lower than plain bicubic resizing by PSNR, which is worse than not running the model at all. The photo is cut into overlapping tiles, each tile is one forward pass, and the tiles are blended back together at the seams. On the CPU a tile can take half a minute, so the page gives the tile count and an estimated time before you start, and Stop ends a run part way through.",
    "The result sits under a draggable comparison against a bicubic upscale of the same input at the same size, because a 2x image on its own gives the eye nothing to judge it by. This page covers only the super-resolution half of image-to-image: editing a picture from a prompt, or img2img, needs a diffusion model, and none is run here. Your image is processed in the browser and never uploaded.",
  ],
  aliases: [
    "AI image upscaler",
    "super-resolution",
    "upscale a photo",
    "Swin2SR",
    "image-to-image",
  ],
};
