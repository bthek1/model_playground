import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline: "Turn a sentence into a vector, in your browser, on your GPU (WebGPU) or CPU (WASM).",
  about: [
    "A text embedding is a fixed-length list of numbers that stands for a piece of text, so that texts with similar meanings land near each other. This page is feature extraction for sentences: type or paste text, press the button, and see the sentence embedding drawn as one row of a diverging heatmap, with its width and its length printed beneath it. It is the vector that semantic search and clustering compare.",
    "Five sentence-transformers checkpoints are offered. The default, all-MiniLM-L6-v2, produces 384 dimensions and is a 45 MB download on WebGPU or 23 MB quantized for the CPU. BGE base v1.5 and all-mpnet-base-v2 are 768-dimensional at 218 MB on WebGPU and 110 MB on the CPU; Nomic Embed v1.5 is 274 MB and 137 MB; GTE ModernBERT is the largest at 298 MB and 150 MB. Each one says how it pools its token rows, mean or CLS, before anything downloads, because that choice is part of how the model was trained rather than a setting.",
    "A slider cuts the vector to its first few dimensions and renormalises it. That is Matryoshka embedding truncation, and only Nomic Embed was trained for it, so on the other four the page reports how much of the vector the cut kept instead of implying it was free. Moving the slider recomputes on the main thread and runs no model. A vector means nothing on its own; to compare two, use Sentence Similarity. The text stays in your browser.",
  ],
  aliases: [
    "text embedding",
    "sentence embedding",
    "feature extraction",
    "Matryoshka embedding",
    "all-MiniLM-L6-v2",
    "embedding truncation",
  ],
};
