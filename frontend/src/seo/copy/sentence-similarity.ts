import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline: "How close two sentences are, as a cosine between their embeddings — computed in your browser.",
  about: [
    "Sentence similarity, or semantic textual similarity, is a way to compare two sentences by meaning: do they say the same thing even when they share few words? This page embeds both sentences with a sentence-transformers model and scores the pair by cosine similarity, a number between −1 and 1. A score on its own has no scale, so the page can score the sample pairs together and rank them, which shows where your own pair sits.",
    "Four sample pairs span the range: a paraphrase with almost no words in common, an unrelated pair that sets the floor, two uses of the word bank that share letters but not meaning, and a negation. \"The flight to Berlin was cancelled\" and its opposite score very high on every model here. That is a known limitation of sentence embeddings, not a fault in the page, and the expectation is written beside the sample.",
    "The models are the same five as Text Features: all-MiniLM-L6-v2 by default (45 MB on WebGPU, 23 MB on the CPU), up to GTE ModernBERT at 298 MB. Vectors are cached by the exact sentence, so editing one side and comparing again embeds only that side. Truncating both vectors to fewer dimensions re-scores the pairs without running the model. Switching model clears the cache, because vectors from different checkpoints cannot be compared. The sentences are not uploaded.",
  ],
  aliases: [
    "sentence similarity",
    "semantic textual similarity",
    "cosine similarity",
    "compare two sentences",
    "paraphrase",
  ],
};
