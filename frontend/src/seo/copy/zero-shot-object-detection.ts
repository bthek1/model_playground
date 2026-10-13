import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Describe what you are looking for and the model boxes it — no class list, no fine-tuning.",
  about: [
    "Zero-shot object detection finds things in a picture from a typed description instead of a fixed list of classes. An ordinary detector only knows the 80 categories it was trained on; an open-vocabulary object detector reads your phrases and draws a box wherever one matches, so you can detect objects by name — \"a traffic light\", \"a red umbrella\" — without retraining anything. Upload a photo, pick a sample or use your webcam, write one phrase per line, and press Detect.",
    "Three models are on offer. OWLv2 base/16 is the default, a 308 MB download at fp16; OWL-ViT base/32, the original, is 307 MB and quicker but misses small objects; Grounding DINO tiny, 360 MB, treats detection as phrase grounding and expects a sentence rather than a class name. All three run on WebGPU in a Web Worker on your own graphics card, and the picker states the size before anything downloads. Smaller quantized builds exist for the CPU, but OWLv2's currently fails to open in the WASM runtime, so treat this as a GPU page.",
    "Your phrases are sent to the model exactly as typed, with no hidden template, which is why the default queries keep their articles. The confidence slider starts at 0.1 because an open-vocabulary model spreads its scores thinly, and moving it only re-filters the boxes already returned — editing a phrase is what re-runs the model. Phrases that found nothing stay listed so you can see which ones missed. The photo is processed in the tab and never uploaded.",
  ],
  aliases: [
    "zero-shot object detection",
    "open-vocabulary object detector",
    "detect objects by name",
    "OWL-ViT",
    "OWLv2",
    "Grounding DINO",
  ],
};
