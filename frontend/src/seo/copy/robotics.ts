import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Find the object you name in a photo or camera frame and rank it by distance, or train a behaviour cloning policy in your browser.",
  about: [
    "Robot perception starts with grounding: turning words such as \"a red block\" into a place in the camera image. Type one or more short object names, pick a sample picture, upload one or use your webcam, and press Locate. OWLv2, an open-vocabulary object detection model, draws a box around each match, and Depth Anything V2 Small estimates depth so the page can rank the boxes from nearest to furthest. Moving the confidence threshold re-filters the boxes without running either model again.",
    "The pair is a 358 MB download, run in two Web Workers on your GPU, and both are the same checkpoints the zero-shot object detection and depth pages use. The depth is relative, so the page reports an ordering, never a distance in metres. OWLv2 works best with labels rather than sentences, and a phrase that reads like an instruction is flagged. Its quantized CPU export does not currently open, so in practice this half needs WebGPU. Your camera frames and pictures stay on your device.",
    "This is perception, not control: nothing here moves a robot. The control half is the second entry in the picker, behaviour cloning, the simplest form of imitation learning, on a toy two-link arm trained in the tab with no download. A small policy network copies a scripted demonstrator that steers round an obstacle. When the demonstrations go both ways round, the policy averages them and collides in five seeds out of six; trained on demonstrations that all go one way, with the same network, budget and seed, it reaches the target in six of six. The two are shown side by side, because the failure alone would look like a bug.",
  ],
  aliases: [
    "robot perception",
    "behaviour cloning",
    "imitation learning",
    "open-vocabulary object detection",
    "OWLv2",
    "grounding",
  ],
};
