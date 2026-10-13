import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Turn pictures into vectors and search them by similarity — the index is built in this tab and never leaves it.",
  about: [
    "Image feature extraction turns a picture into an embedding: a fixed-length list of numbers where similar images land close together. Those image embeddings are what powers similar-image search, deduplication and clustering. This page builds a small image similarity search in your browser — embed the twelve-picture gallery, add your own photos to it, then pick a query from a file, a sample or the webcam and see its nearest neighbours ranked by cosine similarity.",
    "Four models are offered. DINOv2 small is the default, a 384-dimensional self-supervised model at 44 MB on WebGPU or 24 MB for the CPU; DINOv3 small/16 is 22 MB and runs quantized on both backends because no fp16 export exists; DINOv2 base gives 768-dimensional vectors for 173 MB; and CLIP ViT-B/32 downloads only its vision tower, 176 MB, for a 512-dimensional language-aligned vector. Inference runs in a Web Worker on WebGPU, or on the CPU through WASM where there is no GPU.",
    "For the DINO models, the page lets you choose which vector to take: the CLS token or the mean of the patch tokens. Both come from the same forward pass, so switching re-ranks the neighbours without re-embedding, as does changing how many are shown. CLIP returns one projected vector, so it has no such choice. Switching models clears the index, because vectors from different models are not comparable. The pre-normalisation length of each vector is shown beside the unit-length result.",
    "The index lives in this tab's memory only. Nothing is uploaded, and closing the tab discards it.",
  ],
  aliases: [
    "image embeddings",
    "image feature extraction",
    "image similarity search",
    "similar-image search",
    "CLIP",
    "DINOv2",
  ],
};
