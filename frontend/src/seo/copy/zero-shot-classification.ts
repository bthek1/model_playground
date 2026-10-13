import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Classify text against labels you invent here — no training, no fine-tuning, no head that has ever seen them.",
  about: [
    "Zero-shot classification sorts text into categories you name on the spot, with no training data and no fine-tuning. Write your own custom labels, paste a message, and the page ranks the labels by how well each one fits. The samples cover a support ticket, a message that fits two labels at once, a news sentence and an urgent outage report.",
    "It works through natural language inference (NLI). Each label is put into a hypothesis template — “This example is {}.” by default — and an entailment model judges whether your text implies that sentence. That is one forward pass per label, so ten labels cost ten passes, and the count is shown before you press the button. The template is an editable field, and one without the {} placeholder is refused, because every label would then become the same hypothesis and the ranking would mean nothing. Multi-label mode scores each label independently instead of sharing one softmax across them; switching it runs nothing until you press the button again.",
    "Four NLI models are offered: DeBERTa v3 xsmall, the default, at 143 MB on WebGPU or 87 MB on the CPU; MobileBERT MNLI, the smallest, at 50 MB or 27 MB; DistilBERT MNLI at 134 MB or 68 MB; and BART-large MNLI, the usual benchmark, at 816 MB or 411 MB, which asks for confirmation before it downloads. Each runs in a Web Worker on your GPU or CPU, and neither the text nor the labels leave the browser.",
  ],
  aliases: [
    "zero-shot classification",
    "zero-shot text classification",
    "custom labels",
    "natural language inference",
    "NLI",
    "BART-large MNLI",
  ],
};
