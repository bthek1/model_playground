import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Ask a question about a picture and get an answer in prose — a vision encoder and a language decoder, both running on your own GPU.",
  about: [
    "Image-text-to-text is what a vision-language model does: it reads a picture together with a written question and writes an answer. You can ask it to describe an image, count the people in it, read the text on a sign or say what is unusual about a scene. Upload a photo, drop one in or pick a sample, type a question or tap a preset, and press Generate. The answer streams in word by word after a labelled pause while the image encoder runs.",
    "Two SmolVLM checkpoints are on offer: SmolVLM 256M, a 189 MB download, and SmolVLM 500M at 358 MB, which gives noticeably better answers for twice the size. Both use 4-bit weights with half-precision activations, so they run on WebGPU only and need a GPU adapter that supports shader-f16; where it is missing, the picker disables them before anything is fetched. The picture is scaled to 512 pixels on its longest side, SmolVLM's own tile size, so one question costs one tile rather than seventeen. The weights are cached by your browser and the picture is never uploaded.",
    "These are small models, and a generated answer reads fluently whether or not it is right: a model that cannot make out what you asked about still writes a confident sentence, so check it against the picture. It is the nearest thing this site has to OCR, but it answers a question rather than producing a transcription. Larger models such as Qwen3-VL-2B, at 1373 MB, are past the size this site allows. To ask the same models for one-word answers, see Visual Question Answering.",
  ],
  aliases: [
    "vision-language model",
    "image-text-to-text",
    "SmolVLM",
    "ask questions about an image",
    "describe an image",
  ],
};
