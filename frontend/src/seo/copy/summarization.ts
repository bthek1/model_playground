import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Summarize an article in your browser, and compare the summary against the first three sentences of the same article.",
  about: [
    "Summarization condenses a longer text into a few sentences. This page runs abstractive summarizers — models that write new sentences rather than copying them — on an article you paste or one of the samples, and sets the result beside a lead-3 baseline: the article's own first three sentences. The baseline needs no model, so it is on screen before anything downloads. News puts its main facts first, which makes those three sentences hard to beat, and watching a model fall short of them is part of what the page is for.",
    "Three models are offered. T5-small is the default and the only one with a CPU path, at 154 MB on WebGPU or 202 MB on the CPU; it is a weak summarizer, which keeps the comparison honest. DistilBART CNN 6-6 (284 MB) and BART-large CNN (463 MB) were fine-tuned on CNN/DailyMail news and run on WebGPU only: DistilBART's quantized decoder will not open a session on the CPU runtime and its full-precision build is 743 MB, and BART-large shares the architecture and the restriction. As a text summarizer, either reads like a news desk because that is what it learned from.",
    "A summary can read well and still be wrong, for instance by attaching a figure to the wrong person. An optional faithfulness check loads a small entailment model, with its own button and download, and scores each summary sentence against the article separately, so one unsupported clause is not averaged away. The length settings change the generation itself, so editing them runs nothing until you press the button again. The article is processed in a Web Worker on your device and is not uploaded.",
  ],
  aliases: [
    "text summarizer",
    "summarize an article",
    "T5",
    "DistilBART",
    "lead-3 baseline",
    "abstractive summarizer",
  ],
};
