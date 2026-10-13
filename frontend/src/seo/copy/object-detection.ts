import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Find and box every object in a picture — or in your webcam, frame by frame — on your own GPU.",
  about: [
    "Object detection locates the things in an image and draws a bounding box around each one, with a label and a confidence score. Here you can detect objects in a photo you upload, in a sample, or in a live webcam feed, where a new frame is sent only once the previous result is back, so the boxes never fall seconds behind the camera. Every model was trained on the COCO dataset's everyday categories: people, vehicles, animals, furniture and the like.",
    "The default is D-FINE nano, small enough for real-time detection on a webcam: 7.9 MB on WebGPU or 4.5 MB on the CPU. D-FINE small (21 MB) draws tighter boxes. YOLOS small (62 MB) is a plain vision transformer used as a detector, RT-DETR r50 (88 MB) is a real-time DETR that runs on WebGPU only, and DETR r50 (84 MB) is the original set-prediction detector, slow and worth running once. Each runs in a Web Worker, on your GPU through WebGPU or on the CPU through WebAssembly.",
    "The confidence slider filters the boxes already returned rather than running the model again: the detector is asked once at a low floor, so dragging it costs nothing. A closed-set detector can only find the kinds of object it was trained on; to look for something else by typing its name, use Zero-Shot Object Detection instead. Pictures and camera frames stay on your device.",
  ],
  aliases: [
    "object detection",
    "bounding box",
    "real-time detection",
    "D-FINE",
    "DETR",
    "COCO",
  ],
};
