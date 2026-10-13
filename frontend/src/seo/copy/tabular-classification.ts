import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Drop in a CSV, pick the column to predict, and compare four classifiers trained on your own data.",
  about: [
    "Tabular classification predicts a category from the rows of a spreadsheet: which customer defaults, which species a penguin is. On this page you load a CSV or one of three samples, choose the target column and the features, and fit a ladder of four model families on the same train/test split: logistic regression as the floor, a random forest, gradient boosting, and a small neural network (MLP). Each fit reports accuracy beside the majority-class baseline, a confusion matrix, and permutation importance showing which columns the model relied on.",
    "Everything is trained in the tab. The logistic regression and the MLP run as WGSL on your GPU through WebGPU; the trees are TypeScript in a Web Worker on the CPU, because recursive splitting has no matrix multiply to speed up, and the page says which compute each fit used. Files up to 50 000 rows are fitted whole; larger ones are sampled evenly rather than cut to their first rows. A threshold slider recomputes precision, recall and the confusion matrix from the held-out probabilities without fitting again.",
    "Your CSV never leaves your device: nothing is uploaded, nothing is stored, and a reload forgets the file. The samples are chosen to disagree. On Palmer penguins and the wine data the linear model already wins; on the synthetic credit-risk file, built around an interaction no straight line can express, the random forest measured 0.731 against a 0.556 baseline, gradient boosting 0.700, logistic regression 0.611 and the MLP 0.631. Neural networks rarely win on tabular data, and the page lets you watch that happen rather than take it on trust.",
  ],
  aliases: [
    "tabular classification",
    "train a classifier",
    "CSV",
    "random forest",
    "gradient boosting",
    "logistic regression",
    "confusion matrix",
    "permutation importance",
  ],
};
