import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Ask a question about a video clip and get a written answer from SmolVLM2, running on your own GPU.",
  about: [
    "Video-text-to-text, also called video question answering, means asking a question about a video and getting an answer in text: what happens in the clip, what the people are doing, whether anything changes between the first and last frame. Pick a sample clip or upload a video file, type a question or tap a preset, choose how many frames to sample, and press Generate.",
    "The model is SmolVLM2 256M Video, a 189 MB download that runs on WebGPU only and needs a GPU adapter with shader-f16 support. It does not watch the video. The page takes between one and eight still frames, one from the centre of each equal slice of the clip, scales each to 512 pixels on its longest side, and hands them to the model together in one prompt. The frames it was actually given appear as a filmstrip beside the answer. Each frame adds about 64 image tokens, so the frame count is the cost: eight frames is roughly twice the work of four.",
    "Because the model sees a handful of stills, it has no sense of motion between them. A reverse toggle feeds the same frames backwards; if the answer does not change, the model was describing pictures rather than following a sequence, which is the usual result at this size. That comparison is a real second inference, and the page says so before you press. The clip is decoded in your browser and never uploaded.",
  ],
  aliases: [
    "video question answering",
    "video-text-to-text",
    "SmolVLM2",
    "ask questions about a video",
    "video clip",
  ],
};
