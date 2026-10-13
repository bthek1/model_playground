import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline: "Turn one photo into a depth map, on your own GPU.",
  about: [
    "Depth estimation predicts how far away every pixel of a photo is, from a single image — monocular depth estimation, with no stereo pair and no depth sensor. This page runs Depth Anything V2 Small in your browser: upload a picture, drop one in, pick a sample or point your webcam at a scene, and get a per-pixel depth map painted as a colour ramp beside the original.",
    "The model is about 25 million parameters — a 50 MB download on WebGPU at fp16, or 27 MB quantized for the CPU fallback — and the estimate is one forward pass in a Web Worker. The weights come to your browser once and are cached; the photo is never uploaded.",
    "What it produces is relative depth, not metres. Depth Anything outputs inverse depth on an arbitrary scale that is re-fitted to every image, so a brighter pixel is nearer than a darker one in the same picture, but two pictures cannot be compared and no value is a distance. The page says so beside the colour bar. For a 3D view of the same estimate, Image to 3D turns the depth map into a point cloud.",
  ],
  aliases: [
    "depth map",
    "monocular depth estimation",
    "Depth Anything",
    "relative depth",
    "single image",
  ],
};
