import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Score a sentence with a sentiment classifier running entirely in your browser — on your GPU (WebGPU) or CPU (WASM), in a Web Worker.",
  about: [
    "Text classification assigns one label to a whole piece of text, and sentiment analysis is its most familiar form: is this review, post or headline positive or negative? Type or paste a sentence, or start from a sample — a bad review, a market report, a sarcastic post, a flat statement — and the page lists every label the model knows with a score for each, not only the winner. A label at 0.51 and the same label at 0.99 are different answers, and a near-tie is said in words.",
    "Three sentiment classifiers are offered, each trained on different writing. DistilBERT SST-2 learned from film reviews and is a 134 MB download on WebGPU or 68 MB quantized for the CPU. Twitter RoBERTa learned from tweets and is the only one with a neutral class, at 250 MB or 126 MB. FinBERT learned from financial news, at 219 MB or 111 MB. Whichever you pick runs in a Web Worker on your own hardware, and the sentence you type is never sent anywhere.",
    "The comparison is the reason the page exists. Load a second model and run the same sentence through both: a classifier reading text from outside its training domain tends to be confidently wrong rather than unsure, and FinBERT reads “the stock plunged” as news, not as a mood. The second model is a second download and a second model in memory, so its size is quoted before you load it. Typing never runs anything; only the classify button does.",
  ],
  aliases: [
    "sentiment analysis",
    "text classification",
    "sentiment classifier",
    "positive or negative",
    "FinBERT",
  ],
};
