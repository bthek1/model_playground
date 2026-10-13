import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline: "Turn one photo into a point cloud you can orbit.",
  about: [
    "This page converts a 2D photo to 3D in the simplest way that runs in a tab: it estimates a depth map from a single image, then unprojects every sampled pixel into space to build a coloured point cloud you can drag to orbit and scroll to zoom. Pick a file, a sample or a webcam frame and press Build point cloud. Scenes with a clear foreground and background give the most convincing result.",
    "It covers only the depth-to-point-cloud half of image to 3D. Full 3D reconstruction — a mesh with the hidden sides filled in — comes from diffusion models derived from Stable Diffusion, which are far too large for a browser and stay on a server. Here there is no new model at all: depth comes from Depth Anything V2 Small, a 50 MB download on WebGPU or 27 MB for the CPU, with the older Depth Anything v1 mirror as an alternative of the same size.",
    "Inference runs in a Web Worker through Transformers.js, and the cloud is drawn by a hand-written WGSL render pipeline on your GPU. Without WebGPU the depth estimate still runs on the CPU and the page shows the depth map instead of the 3D view.",
    "The geometry is plausible, not measured. Depth Anything predicts relative depth with no camera intrinsics, so the focal length is an assumption you set with a slider, and two clouds from two photos cannot be compared. The focal-length and point-density sliders rebuild the cloud from the depth already computed without re-running the model. Your photo is never uploaded.",
  ],
  aliases: [
    "photo to 3D",
    "image to 3D",
    "point cloud",
    "2D photo to 3D",
    "depth map",
    "3D reconstruction",
  ],
};
