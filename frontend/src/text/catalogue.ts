// The text model catalogue — every NLP route's entries live here, grouped by
// task, so one module answers "what does this category download".
//
// **Every entry carries measured `bytes` for both backends**, which is a
// stricter rule than vision's "measure where an estimate would mislead", and it
// is a finding rather than a preference. The NLP roadmap's size tables are all
// q8 figures, while `loadOpts()` asks for fp16 on WebGPU — the backend any
// machine with a GPU adapter gets. The real default download is roughly double,
// all the way down the category:
//
//   distilbert-sst-2            q8  64.5 MiB   fp16 127.9 MiB
//   twitter-roberta-sentiment   q8 120.1 MiB   fp16 238.1 MiB
//   finbert                     q8 105.6 MiB   fp16 209.2 MiB
//
// Read off the Hub's blob listing (`just fe-e2e-models` re-checks them), never
// estimated. Two of these cross `LARGE_MODEL_BYTES` on WebGPU while reading as
// comfortably under it at q8 — `sizeEstimate` keys its warning off the *bigger*
// of the two downloads, so that is handled, but only because the numbers here
// are real.
//
// A note on `q4`, because it looks like an obvious lever and is not: on every
// encoder measured for this category, `model_q4.onnx` is *larger* than
// `model_quantized.onnx` (distilbert-sst-2: 118.9 MiB q4 against 64.5 MiB q8),
// and `q4f16` is usually larger too. 4-bit is a decoder format. No encoder page
// should reach past the `loadOpts()` default.

import type { TextModel } from "./types";

export interface TextClassifierModel extends TextModel {
  task: "text-classification";
  /** The labels this checkpoint can emit, for the picker. */
  labels: readonly string[];
  /** The domain it was fine-tuned on — the page's whole subject. */
  domain: string;
}

/**
 * §3.1's head-to-head: three sentiment classifiers trained on three different
 * kinds of writing. Running one sentence through two of them is the lesson —
 * "domain-matched" is a claim about a *specific* domain, and a model outside it
 * is confidently wrong rather than uncertain.
 *
 * **`onnx-community/ModernBERT-base-ONNX` is deliberately absent**, against the
 * roadmap's §3.1 table, which listed it third with the note "the base model, so
 * fine-tune before it classifies anything". That note is the disqualification:
 * a base encoder has no trained classification head, so on this page it emits
 * `LABEL_0` / `LABEL_1` from randomly initialised weights — a confident, fluent,
 * meaningless answer, with nothing failing on the way there. It is the exact
 * class of silent wrongness this repo pins tests against, and a page is not the
 * place to demonstrate it. `Xenova/finbert` takes the slot instead: a real
 * fine-tune, a third domain, and it makes the head-to-head demonstrable on a
 * sentence both of the others misread. ModernBERT keeps its place on `/fill-mask`,
 * where a base model is exactly what is wanted.
 */
export const TEXT_CLASSIFIER_MODELS: TextClassifierModel[] = [
  {
    id: "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
    label: "DistilBERT SST-2",
    hint: "The general default — binary sentiment, trained on film reviews.",
    domain: "film reviews (SST-2)",
    labels: ["NEGATIVE", "POSITIVE"],
    params: 67,
    task: "text-classification",
    bytes: { webgpu: 134_088_280, wasm: 67_581_197 },
  },
  {
    id: "Xenova/twitter-roberta-base-sentiment-latest",
    label: "Twitter RoBERTa",
    hint: "Domain-matched to social posts, and the only one of the three with a neutral class.",
    domain: "tweets (~124M of them)",
    labels: ["negative", "neutral", "positive"],
    params: 125,
    task: "text-classification",
    bytes: { webgpu: 249_669_194, wasm: 125_905_426 },
  },
  {
    id: "Xenova/finbert",
    label: "FinBERT",
    hint: "Financial news. Reads “the stock plunged” as information, not as a mood.",
    domain: "financial news",
    labels: ["positive", "negative", "neutral"],
    params: 110,
    task: "text-classification",
    bytes: { webgpu: 219_322_230, wasm: 110_717_965 },
  },
];

export const DEFAULT_TEXT_CLASSIFIER = TEXT_CLASSIFIER_MODELS[0].id;

/**
 * How many labels to show.
 *
 * All of them, for a classifier this small — these heads have two to six
 * classes, and the interesting case is a 0.51 / 0.49 split that a single
 * confident-looking label hides. `ScoreList` refuses to render one row for the
 * same reason.
 */
export const TOP_K = 6;

/**
 * Sentences chosen so the three models *disagree*, because a sample set every
 * model gets right demonstrates nothing about any of them. Each one names the
 * model it is a trap for.
 */
export interface TextSample {
  id: string;
  label: string;
  text: string;
  hint: string;
}

export const CLASSIFIER_SAMPLES: TextSample[] = [
  {
    id: "tedium",
    label: "A bad review",
    text: "The film was a triumph of tedium — two hours I will never get back.",
    hint: "The easy case. Every model should call this negative.",
  },
  {
    id: "plunged",
    label: "A market report",
    text: "Shares plunged 12% after the company slashed its full-year guidance.",
    hint: "A fact, not a mood. The film-review model reads it as an outburst; FinBERT does not.",
  },
  {
    id: "sarcasm",
    label: "A sarcastic post",
    text: "oh brilliant, another four hour outage. exactly what i wanted today 🙃",
    hint: "Emoji, lowercase, sarcasm. The tweet-trained model has seen this register; SST-2 has not.",
  },
  {
    id: "neutral",
    label: "A flat statement",
    text: "The meeting has been moved to Thursday at eleven.",
    hint: "Genuinely neutral — and SST-2 has no neutral class, so it must pick a side.",
  },
];
