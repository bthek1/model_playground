import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Find every person and draw their skeleton — two models running back to back on your own GPU, one to find people and one to pose them.",
  about: [
    "Human pose estimation locates the joints of each person in an image — shoulders, elbows, wrists, hips, knees and ankles — and joins them into a stick-figure skeleton. This page does keypoint detection for the 17 COCO keypoints on a photo, a sample or a live webcam feed, entirely in your browser.",
    "It is top-down pose estimation, so two models run one after the other: a detector finds the people, then ViTPose base estimates the keypoints on each person's crop. The default pair is D-FINE nano with ViTPose base, a 180 MB combined download at fp16 on WebGPU or 92 MB quantized for the CPU, and it is fast enough for the webcam. The alternative pairs RT-DETR r50 with the same pose model, 260 MB, a stronger detector that runs on WebGPU only and is offered for still images. The picker quotes the combined size, because neither half works alone.",
    "A person-confidence slider and a cap on how many people to pose travel with each run: every extra person is another pass through the pose model, so the cap keeps a crowded scene from stalling the tab. Each joint is drawn with an opacity that follows the model's confidence, because a heatmap always peaks somewhere and an occluded ankle comes back as a guess. Detection and pose timings are reported separately.",
    "Both models run in a Web Worker on your device, and neither the photo nor the camera frames are uploaded.",
  ],
  aliases: [
    "human pose estimation",
    "keypoint detection",
    "skeleton",
    "ViTPose",
    "webcam",
    "top-down pose estimation",
  ],
};
