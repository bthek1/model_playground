import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Classify what a video clip shows, frame by frame with CLIP, in your browser.",
  about: [
    "Video classification puts a label on a clip: what is happening in it. A true video model such as VideoMAE, TimeSformer or X-CLIP attends across time as well as space, and none of them has an ONNX export a browser can run. So this page is a frame-level baseline. It samples frames from the video, scores each one against your labels with a zero-shot image model, and pools the scores over a sliding window. The model only ever sees single still frames, never motion, and the page says so beside the result.",
    "Upload a video file or choose a sample clip, edit the labels you want it scored against, and press the button. Frames are sampled at 2 per second by default, up to 120 frames, by seeking rather than playing, so a long clip is not decoded in full. The chart draws one line per label over time, and the pooling window slider re-smooths the scores already in hand without scoring the clip again.",
    "The frames are scored by CLIP ViT-B/32 (a 304 MB download on WebGPU, 154 MB on the CPU) or SigLIP base/16 (407 MB, or 211 MB), in a Web Worker on your GPU or CPU. The label embeddings are computed once and reused for every frame. Read the result as a control experiment: if single frames already name the action, motion was not needed to tell it apart. The video is decoded in your browser and is not uploaded.",
  ],
  aliases: [
    "video classification",
    "frame-level baseline",
    "video clip",
    "CLIP",
    "VideoMAE",
  ],
};
