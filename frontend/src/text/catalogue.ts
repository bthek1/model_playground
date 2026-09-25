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

import { SEQ2SEQ_WASM_DTYPES } from "@/model/backend";

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

// --- Token classification (NER) ----------------------------------------------

export interface NerModel extends TextModel {
  task: "token-classification";
  /** The entity types this head can emit, for the picker and the legend. */
  entities: readonly string[];
  /** What it was trained on — languages, corpus. */
  domain: string;
}

/**
 * §3.2's two entries.
 *
 * **The roadmap is wrong that multilingual NER is lost.** It sends the task to a
 * server on the strength of two repos with no ONNX export
 * (`Babelscape/wikineural-multilingual-ner`,
 * `Jean-Baptiste/roberta-large-ner-english`) — but the runtime's *own default*
 * for `token-classification` is the multilingual entry below, measured at
 * 178.5 MB q8 / 354.9 MB fp16, under the bar. A page was planned without it.
 *
 * The two heads emit different fourth types — `MISC` against `DATE` — which is
 * why `entitySlot()` lets them share a colour slot: only one model is ever
 * live, so they cannot appear together.
 */
export const NER_MODELS: NerModel[] = [
  {
    id: "Xenova/bert-base-NER",
    label: "BERT base NER",
    hint: "English CoNLL-2003 — people, organisations, locations and a catch-all MISC.",
    domain: "English newswire (CoNLL-2003)",
    entities: ["PER", "ORG", "LOC", "MISC"],
    params: 108,
    task: "token-classification",
    bytes: { webgpu: 215_805_679, wasm: 108_952_255 },
  },
  {
    id: "Xenova/bert-base-multilingual-cased-ner-hrl",
    label: "mBERT NER (10 languages)",
    hint: "Ten languages, and it tags DATE instead of MISC. A third again the download.",
    domain: "10 high-resource languages",
    entities: ["PER", "ORG", "LOC", "DATE"],
    params: 177,
    task: "token-classification",
    bytes: { webgpu: 354_892_016, wasm: 178_495_423 },
  },
];

export const DEFAULT_NER_MODEL = NER_MODELS[0].id;

/**
 * The types a redact button removes by default: the two that actually identify
 * a person. Organisations and dates are usually the part worth keeping, which
 * is why the choice is per-type rather than all-or-nothing.
 */
export const DEFAULT_REDACTED: readonly string[] = ["PER", "LOC"];

/**
 * Paragraphs with entities of every type in them, so the overlay has something
 * to show on each model, and one that crosses a sentence boundary — a span that
 * ends at a full stop is where an off-by-one offset first becomes visible.
 */
export const NER_SAMPLES: TextSample[] = [
  {
    id: "memo",
    label: "An internal memo",
    text: "Priya Raman flew from Wellington to Berlin on Tuesday to meet the team at Siemens, and filed her expenses with Deloitte the following week.",
    hint: "People, cities and companies in one sentence — the case the redact button is for.",
  },
  {
    id: "news",
    label: "A news lede",
    text: "The European Space Agency confirmed on Monday that Ariane 6 lifted off from Kourou, French Guiana, carrying a payload built by Airbus in Toulouse.",
    hint: "Organisations nested inside locations, and a product name the English model files under MISC.",
  },
  {
    id: "multilingual",
    label: "A sentence in French",
    text: "Marie Curie est née à Varsovie et a travaillé à Paris avec Pierre Curie à la Sorbonne.",
    hint: "The English model finds some of this; the multilingual one is what it is for.",
  },
];

// --- Zero-shot classification (NLI) ------------------------------------------

export interface ZeroShotTextModel extends TextModel {
  task: "zero-shot-classification";
  /** The NLI corpus it was fine-tuned on — what "entailment" means to it. */
  domain: string;
}

/**
 * §3.4's four entries.
 *
 * **The roadmap's own correction is confirmed, and there is a fourth entry it
 * did not list.** `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` publishes
 * exactly one ONNX file — `onnx/model.onnx` at fp32, 738.6 MB — so the
 * "~180 MB" that #5 recommended as this page's *default* was an estimate of a
 * file that does not exist. It is cut. `Xenova/distilbert-base-uncased-mnli`
 * takes a slot instead: it is what `pipeline("zero-shot-classification")`
 * loads when you name no model at all, which makes its absence from the
 * roadmap's table the more surprising of the two findings.
 *
 * `Xenova/bart-large-mnli` is not "411 MB, under the bar, but gate it" either.
 * That is its q8 size; on WebGPU `loadOpts()` asks for fp16 and the download is
 * **815,853,432 bytes** — 816 MB decimal, which `formatBytes` renders as
 * "778 MB" because it divides by 1024. It still ships: §0's bar is about a
 * page's *floor*, and this page's floor is a 26 MB model. But it carries
 * measured bytes for both backends so the picker quotes the number the user
 * will actually pay, and it trips `isHeavyDownload` into a second, explicit
 * opt-in.
 *
 * **Every one of these four declares `entailment` in its `label2id`, and that
 * is load-bearing rather than incidental.** The pipeline looks the index up by
 * name and falls back to `2` with only a `console.warn` if it is missing — and
 * the right index is 1 for the DeBERTa head, 0 for MobileBERT and DistilBERT,
 * and 2 for BART. A checkpoint without the mapping would score the *neutral*
 * or *contradiction* logit on three of these four, producing a confident,
 * plausible, wrongly-ordered list. `model-ids.spec.ts` checks the mapping on
 * the Hub, because nothing in a run would tell you.
 */
export const ZERO_SHOT_TEXT_MODELS: ZeroShotTextModel[] = [
  {
    id: "Xenova/nli-deberta-v3-xsmall",
    label: "DeBERTa v3 xsmall (NLI)",
    hint: "The default — the best accuracy per megabyte of the four, and a modern tokenizer.",
    domain: "SNLI + MultiNLI",
    params: 71,
    task: "zero-shot-classification",
    bytes: { webgpu: 142_841_421, wasm: 87_246_587 },
  },
  {
    id: "Xenova/mobilebert-uncased-mnli",
    label: "MobileBERT MNLI",
    hint: "The smallest entailment model that works at all — 26 MB on CPU. Coarse, and fast enough to feel free.",
    domain: "MultiNLI",
    params: 25,
    task: "zero-shot-classification",
    bytes: { webgpu: 50_077_049, wasm: 26_967_165 },
  },
  {
    id: "Xenova/distilbert-base-uncased-mnli",
    label: "DistilBERT MNLI",
    hint: "What Transformers.js loads when you name no model — the runtime's own default for this task.",
    domain: "MultiNLI",
    params: 67,
    task: "zero-shot-classification",
    bytes: { webgpu: 134_089_818, wasm: 67_581_975 },
  },
  {
    id: "Xenova/bart-large-mnli",
    label: "BART-large MNLI",
    hint: "The classic, and the one everyone benchmarks against — and the only one here where a pass per label is a wait per label.",
    domain: "MultiNLI",
    params: 407,
    task: "zero-shot-classification",
    bytes: { webgpu: 815_853_432, wasm: 411_300_557 },
  },
];

export const DEFAULT_ZERO_SHOT_TEXT = ZERO_SHOT_TEXT_MODELS[0].id;

/**
 * A sample is a premise **and the labels to score it against**, because half a
 * zero-shot example is not an example: the interesting thing about this task is
 * that the label set is the user's, so a sample that fills only the text box
 * demonstrates nothing the previous page did not.
 *
 * Each set is a realistic routing problem — the actual reason someone reaches
 * for zero-shot classification instead of fine-tuning a head.
 */
export interface ZeroShotSample extends TextSample {
  labels: readonly string[];
}

export const ZERO_SHOT_SAMPLES: ZeroShotSample[] = [
  {
    id: "ticket",
    label: "A support ticket",
    text: "I was charged twice for the same subscription this month and the second payment has not been refunded.",
    labels: ["billing", "outage", "feature request"],
    hint: "The routing case. One obvious answer, and a label set with no overlap — this is what the task is for.",
  },
  {
    id: "overlap",
    label: "Two labels at once",
    text: "The app crashed during checkout and I was still charged for the order.",
    labels: ["billing", "bug report", "praise"],
    hint: "Genuinely two things. Under a single-label softmax the model must split its score; turn multi-label on and both can be high.",
  },
  {
    id: "topic",
    label: "A news sentence",
    text: "The central bank held rates steady, citing softer wage growth in the services sector.",
    labels: ["economics", "sport", "technology", "health"],
    hint: "Topic labelling with four candidates — four forward passes, which is where the pass count starts to be visible.",
  },
  {
    id: "urgency",
    label: "An urgency call",
    text: "Production is down for every customer in the EU region and nobody can log in.",
    labels: ["urgent", "routine"],
    hint: "Two labels, so two passes. The template matters most here — “This example is urgent.” is a sentence; “urgent” alone is not.",
  },
];

// --- Extractive question answering -------------------------------------------

export interface QaModel extends TextModel {
  task: "question-answering";
  /** What it was fine-tuned on — the page's whole subject. */
  domain: string;
  /**
   * Whether this checkpoint can answer "there is no answer in this passage".
   *
   * A field rather than a line of copy on the route, and the distinction is the
   * page's subject. SQuAD **1.1** models always answer: every training example
   * had an answer in its passage, so the head has no way to express its
   * absence. SQuAD **2.0** models can — `deepset/roberta-base-squad2` and
   * `deepset/deberta-v3-large-squad2` are the usual ones — and **neither has an
   * ONNX export**, so the behaviour is not merely unshipped here, it is
   * unavailable. Keying the page's disclaimer off this flag rather than
   * hard-coding it is the repo's "rewrite a mechanism rather than deleting it
   * with its subject" rule: when a squad2 export appears, the entry sets `true`
   * and the note goes away on its own.
   */
  canAbstain: boolean;
}

/**
 * §3.3's single entry, and it really is the only one.
 *
 * `deepset/roberta-base-squad2` and `deepset/deberta-v3-large-squad2` have no
 * ONNX export, and #5's suggested `onnx-community/Qwen3-0.6B-ONNX` reader is
 * 569.8 MB — over the feasibility bar, to produce the same `{span, score}` a
 * 63 MB encoder produces. So §0's second question is satisfied here by the
 * **floor being 63 MB**, not by there being an alternative to fall back to.
 *
 * Sizes measured off the Hub's blob listing, like every entry in this file:
 * `onnx/model_quantized.onnx` 65_811_627 (62.8 MiB) and `onnx/model_fp16.onnx`
 * 130_563_083 (124.5 MiB). `just fe-e2e-models` re-checks both.
 *
 * The tokenizer config matters here in a way it does not elsewhere, because
 * `text/offsets.ts` aligns this checkpoint's WordPiece tokens back onto the
 * user's own characters: `do_lower_case: false`, `strip_accents: null`. A
 * lowercasing entry would fail that alignment — safely, by reporting no
 * highlight, but it would lose the page its whole output.
 */
export const QA_MODELS: QaModel[] = [
  {
    id: "Xenova/distilbert-base-cased-distilled-squad",
    label: "DistilBERT SQuAD",
    hint: "The only exported extractive reader — cased, English, SQuAD 1.1.",
    domain: "Wikipedia paragraphs (SQuAD 1.1)",
    canAbstain: false,
    params: 65,
    task: "question-answering",
    bytes: { webgpu: 130_563_083, wasm: 65_811_627 },
  },
];

export const DEFAULT_QA_MODEL = QA_MODELS[0].id;

/**
 * The label every answer span carries. Extractive QA marks one range and the
 * page names it above the passage, so `SpanOverlay` is asked to hide the inline
 * tag — but the span still needs a type for its colour slot.
 */
export const ANSWER_LABEL = "ANSWER";

/** A passage and a question, for the sample buttons. */
export interface QaSample {
  id: string;
  label: string;
  question: string;
  context: string;
  hint: string;
  /**
   * The passage does not contain an answer to the question. The model will
   * answer anyway — that is the demonstration, not a bug to be filtered out.
   */
  unanswerable?: boolean;
}

/**
 * Four pairs, each chosen for something the page has to be able to show.
 *
 * The scores quoted below are measured against the q8 build via
 * `onnxruntime-node`, and they are what the `@slow` spec pins — not "a result
 * appeared", which is exactly what a broken alignment also produces.
 */
export const QA_SAMPLES: QaSample[] = [
  {
    id: "eiffel",
    label: "A fact in the passage",
    question: "Who built the Eiffel Tower?",
    context:
      "The Eiffel Tower was built by Gustave Eiffel for the 1889 World's Fair in Paris. It stood as the world's tallest man-made structure for 41 years, until the Chrysler Building in New York was finished in 1930.",
    hint: "The easy case — “Gustave Eiffel”, characters 30–44, at 0.997.",
  },
  {
    // The sample that justifies `text/offsets.ts` existing at all. The model's
    // own decode of this answer is `general - purpose compute shaders`, which
    // does not occur in the passage — so the obvious shortcut, searching the
    // passage for the answer string, finds nothing and highlights nothing.
    // Slicing [213, 244) returns the passage's characters, hyphen intact.
    id: "webgpu",
    label: "An answer the tokenizer re-spaces",
    question: "What does it support that WebGL does not?",
    context:
      "WebGPU is a web standard that exposes modern GPU capabilities to the browser. It was first shipped in Chrome 113 in May 2023, and is developed by the W3C GPU for the Web Community Group. Unlike WebGL, it supports general-purpose compute shaders.",
    hint: "The span is hyphenated; the model's own decode of it is not. This is why the highlight is sliced from your text rather than searched for.",
  },
  {
    id: "w3c",
    label: "A question with a weaker answer",
    question: "Who develops WebGPU?",
    context:
      "WebGPU is a web standard that exposes modern GPU capabilities to the browser. It was first shipped in Chrome 113 in May 2023, and is developed by the W3C GPU for the Web Community Group. Unlike WebGL, it supports general-purpose compute shaders.",
    hint: "The passage says “the W3C GPU for the Web Community Group”; the model takes three words of it, at 0.31. A low score is the model being unsure, and it is worth seeing beside the confident ones.",
  },
  {
    // The abstention demonstration, and the reason it is this pair rather than
    // an obviously off-topic one with a low score: the model does not merely
    // answer, it answers **confidently**. 0.94 on a question the passage never
    // addresses is the finding — the score is not a usable "do I know this"
    // signal either.
    id: "unanswerable",
    label: "A question the passage cannot answer",
    question: "Who won the 1998 World Cup?",
    context:
      "The Eiffel Tower was built by Gustave Eiffel for the 1889 World's Fair in Paris. It stood as the world's tallest man-made structure for 41 years, until the Chrysler Building in New York was finished in 1930.",
    hint: "Nothing here answers this. The model returns “Gustave Eiffel” at 0.94 — it has no way to say “I don't know”, and the score does not warn you.",
    unanswerable: true,
  },
];

// --- Fill-mask (masked language modelling) -----------------------------------

export interface FillMaskModel extends TextModel {
  task: "fill-mask";
  /**
   * The literal this checkpoint's tokenizer uses for its mask.
   *
   * **Data, not a constant, and that is the whole point of the page.** BERT,
   * DistilBERT and ModernBERT use `[MASK]`; RoBERTa uses `<mask>`. It is
   * declared here so the UI can show the token — and insert it — before a model
   * has been downloaded, and it is re-checked against each repo's real
   * `tokenizer_config.json` by `just fe-e2e-models`. At run time the engine
   * defers to the **loaded tokenizer**, so an entry that has drifted produces a
   * note on the page rather than a failed run (`text/mask.ts`).
   */
  maskToken: string;
  /** What it was pretrained on — which is what the page actually measures. */
  domain: string;
}

/**
 * §3.9's four entries: **three tokenizer families**, deliberately.
 *
 * A page with one family cannot demonstrate its own hazard. The mask-token
 * trap is a click away here, and it is worth being precise about what the trap
 * does, because the plan for this page assumed something stronger than what was
 * measured: a wrong literal **throws** rather than lying.
 * `FillMaskPipeline` looks `mask_token_id` up in the token ids and raises
 * "Mask token (<mask>) not found in text." So the hard-coding bug is loud —
 * and it is still a failure the user did nothing to cause, which is why the
 * token is inserted by a button, rewritten on a model change, and reconciled
 * against the tokenizer in the engine.
 *
 * The genuinely silent failure on this page is a *second* mask: the pipeline
 * takes `findIndex` over the ids, fills the first and drops the rest, so
 * "The [MASK] of France is [MASK]." comes back as "the border of france is." —
 * one filling, no error, a sentence quietly missing a word. The route refuses
 * to run on anything but exactly one mask for that reason.
 *
 * All four sizes are measured off the Hub's blob listing. Three of the four
 * cross `LARGE_MODEL_BYTES` on WebGPU, which `sizeEstimate` warns about because
 * these numbers are real rather than estimated from `params`.
 */
export const FILL_MASK_MODELS: FillMaskModel[] = [
  {
    id: "Xenova/bert-base-uncased",
    label: "BERT base (uncased)",
    hint: "The original masked language model, 2018. Lowercases everything it reads.",
    domain: "Wikipedia + BookCorpus, 2018",
    maskToken: "[MASK]",
    params: 110,
    task: "fill-mask",
    bytes: { webgpu: 219_386_932, wasm: 110_848_579 },
  },
  {
    id: "Xenova/distilbert-base-uncased",
    label: "DistilBERT base",
    hint: "Half of BERT, distilled — and it does not know the capital of France.",
    domain: "distilled from BERT, same corpus",
    maskToken: "[MASK]",
    params: 66,
    task: "fill-mask",
    bytes: { webgpu: 134_152_313, wasm: 67_710_626 },
  },
  {
    // The reason the page ships more than one family. Nothing else about this
    // entry is unusual; the token is.
    id: "Xenova/roberta-base",
    label: "RoBERTa base",
    hint: "Same architecture, a different tokenizer — its mask is <mask>, not [MASK].",
    domain: "160 GB of web text, 2019",
    maskToken: "<mask>",
    params: 125,
    task: "fill-mask",
    bytes: { webgpu: 249_771_089, wasm: 126_111_809 },
  },
  {
    // §3.1 cut this model from `/text-classification` precisely because it is a
    // *base* model with no trained head. Here that is the qualification rather
    // than the disqualification: masked language modelling is the objective it
    // was actually pretrained on, so the head is the real one.
    id: "onnx-community/ModernBERT-base-ONNX",
    label: "ModernBERT base",
    hint: "A 2024 encoder on a 2024 corpus. The sharpest of the four, and the largest.",
    domain: "2 trillion tokens of web, code and papers, 2024",
    maskToken: "[MASK]",
    params: 149,
    task: "fill-mask",
    bytes: { webgpu: 299_694_721, wasm: 151_074_492 },
  },
];

export const DEFAULT_FILL_MASK = FILL_MASK_MODELS[0].id;

/**
 * Every mask literal the catalogue knows about, deduplicated.
 *
 * Derived rather than listed, so a fifth tokenizer family arrives as a
 * catalogue entry and `retargetMasks` picks it up for free. This is the list
 * the route rewrites *from* when the user switches model.
 */
export const MASK_TOKENS: readonly string[] = [
  ...new Set(FILL_MASK_MODELS.map((m) => m.maskToken)),
];

/**
 * How many candidate fillings to ask for.
 *
 * More than a classifier's label set, because here the ranking *is* the
 * evidence: the second through eighth guesses are what show the corpus's shape,
 * and a top-1 answer would hide exactly the distribution the page is about.
 */
export const FILL_TOP_K = 8;

/**
 * A sample prompt. `template` carries `{}` where the mask belongs, expanded
 * with whichever token the selected model uses — a stored literal would be the
 * page's own bug baked into its samples.
 */
export interface MaskSample {
  id: string;
  label: string;
  template: string;
  hint: string;
}

export const MASK_SAMPLES: MaskSample[] = [
  {
    id: "capital",
    label: "A fact",
    template: "The capital of France is {}.",
    hint: "BERT says paris at 0.33 and ModernBERT at 0.88 — but DistilBERT says marseille. Distillation costs factual recall, and this is one click.",
  },
  {
    id: "era",
    label: "A date stamp",
    template: "I looked up the address on {}.",
    hint: "There is no fact here to get right. The answer tells you roughly when the corpus was collected.",
  },
  {
    id: "syntax",
    label: "Grammar, not knowledge",
    template: "The keys to the cabinet {} on the table.",
    hint: "Agreement across an intervening noun. A model can have the syntax and none of the facts.",
  },
  {
    id: "domain",
    label: "A specialist sentence",
    template: "The patient was prescribed a course of {}.",
    hint: "General-corpus models thin out fast outside the register they were trained on.",
  },
];

/**
 * A pair of prompts identical but for one word.
 *
 * **This is a measurement of the training corpus, not of the world**, and the
 * page says so beside the result rather than leaving it implied. §3.9 is
 * explicit about the framing: a masked language model's output is the text it
 * was fitted to, read back out, and the pair is what makes that legible — a
 * single prompt shows you a plausible sentence, two identical prompts show you
 * what changed when one word did.
 */
export interface MaskProbe {
  id: string;
  label: string;
  /** Two templates differing in one word. `{}` marks the mask in each. */
  templates: readonly [string, string];
  /** What the two columns differ *by*. */
  varies: readonly [string, string];
  hint: string;
}

export const MASK_PROBES: MaskProbe[] = [
  {
    id: "occupation",
    label: "Occupation",
    templates: ["The man worked as a {}.", "The woman worked as a {}."],
    varies: ["man", "woman"],
    hint: "The classic probe. On BERT: lawyer, farmer, carpenter against nurse, waitress, teacher.",
  },
  {
    id: "pronoun",
    label: "Pronoun",
    templates: [
      "The doctor finished {} shift and went home.",
      "The nurse finished {} shift and went home.",
    ],
    varies: ["doctor", "nurse"],
    hint: "Nothing in either sentence carries gender. Whatever the model fills in came from the corpus.",
  },
  {
    id: "age",
    label: "Age",
    templates: ["The young man is very {}.", "The old man is very {}."],
    varies: ["young", "old"],
    hint: "Adjectives rather than nouns — the same effect in a different part of speech.",
  },
];

// --- Embeddings: feature extraction and sentence similarity ------------------

/**
 * A sentence-embedding checkpoint.
 *
 * Two fields here are **facts about the checkpoint that the catalogue has to
 * carry because nothing downstream can discover them**, which is the same
 * reason `FillMaskModel.maskToken` exists:
 *
 *  - `pooling`. A sentence embedding is not "the model's output" — it is a
 *    pooling of the token rows, and *which* pooling is part of how the model
 *    was trained. all-MiniLM and all-mpnet are mean-pooled; BGE and
 *    gte-modernbert are CLS-pooled. Getting it wrong produces a vector that is
 *    the right width, ranks plausibly, and is wrong, and there is no error
 *    anywhere. The upstream sentence-transformers repo states it in
 *    `1_Pooling/config.json` — and the `Xenova/*` ONNX mirrors **do not carry
 *    that file**, so it cannot be read at load time. `upstream` names the repo
 *    the answer lives in, and `just fe-e2e-models` checks this field against
 *    it.
 *  - `prefixes`. Some models were trained with a task instruction glued to the
 *    front of every input. Nomic is the one here that needs it, and omitting it
 *    costs accuracy silently.
 */
export interface EmbedModel extends TextModel {
  task: "feature-extraction";
  /** Embedding width, so the page can state it before anything downloads. */
  dim: number;
  /** How the token rows become one vector. See above — not a free choice. */
  pooling: "mean" | "cls";
  /** The repo whose `1_Pooling/config.json` is the authority for `pooling`. */
  upstream: string;
  /**
   * True only for a checkpoint trained with Matryoshka representation
   * learning, i.e. one whose *prefix* dimensions were explicitly optimised to
   * stand alone.
   *
   * It is catalogue data rather than an assumption because the truncation
   * control means two different things depending on it: a demonstration on an
   * MRL model, and a **measurement** on the others. Only one entry here is
   * MRL-trained, which is why the page reports the cost rather than asserting
   * there isn't one.
   */
  matryoshka?: boolean;
  /**
   * Task instructions this checkpoint expects, when it expects any.
   *
   * `symmetric` is for comparing two texts of the same kind (this category's
   * two pages); `query` / `document` are the asymmetric retrieval pair that
   * `/text-ranking` needs. Named by use rather than by string so a page cannot
   * pick the wrong one by accident.
   */
  prefixes?: { symmetric: string; query: string; document: string };
}

/**
 * §3.7's catalogue, and the cheapest floor in the app: the default is a
 * **22 MiB** download.
 *
 * `onnx-community/Qwen3-Embedding-0.6B-ONNX` measures 613.5 MB and is cut per
 * §3.7 — a 22 MB model does this page's job, and the roadmap's own bar is
 * about the floor.
 *
 * Both poolings and both prefix regimes are represented deliberately, one
 * click apart: a catalogue where every entry is mean-pooled and prefix-free
 * makes those two fields look like decoration.
 */
export const EMBED_MODELS: EmbedModel[] = [
  {
    id: "Xenova/all-MiniLM-L6-v2",
    label: "all-MiniLM-L6-v2",
    hint: "The default, and the whole argument for this page: 22 MB, 384-d, and good enough for most retrieval.",
    params: 23,
    dim: 384,
    pooling: "mean",
    upstream: "sentence-transformers/all-MiniLM-L6-v2",
    task: "feature-extraction",
    bytes: { webgpu: 45_297_825, wasm: 22_972_370 },
  },
  {
    id: "Xenova/bge-base-en-v1.5",
    label: "BGE base v1.5",
    hint: "Retrieval-tuned, 768-d — and CLS-pooled, not mean-pooled. Mean-pooling it looks fine and is wrong.",
    params: 109,
    dim: 768,
    pooling: "cls",
    upstream: "BAAI/bge-base-en-v1.5",
    task: "feature-extraction",
    bytes: { webgpu: 218_108_236, wasm: 110_083_337 },
  },
  {
    id: "Xenova/all-mpnet-base-v2",
    label: "all-mpnet-base-v2",
    hint: "The sentence-transformers workhorse. 768-d, mean-pooled, stronger than MiniLM on paraphrase.",
    params: 109,
    dim: 768,
    pooling: "mean",
    upstream: "sentence-transformers/all-mpnet-base-v2",
    task: "feature-extraction",
    bytes: { webgpu: 218_117_164, wasm: 110_086_122 },
  },
  {
    id: "nomic-ai/nomic-embed-text-v1.5",
    label: "Nomic Embed v1.5",
    hint: "The only Matryoshka-trained entry: its first 128 dimensions were optimised to stand alone. Needs a task prefix.",
    params: 137,
    dim: 768,
    pooling: "mean",
    upstream: "nomic-ai/nomic-embed-text-v1.5",
    matryoshka: true,
    // From the model card. Dropping these costs accuracy with nothing failing,
    // which is why the page shows the composed string rather than only the text
    // the user typed.
    prefixes: {
      symmetric: "clustering: ",
      query: "search_query: ",
      document: "search_document: ",
    },
    task: "feature-extraction",
    bytes: { webgpu: 273_859_028, wasm: 137_296_292 },
  },
  {
    id: "Alibaba-NLP/gte-modernbert-base",
    label: "GTE ModernBERT",
    hint: "The strongest here and the largest. CLS-pooled, 768-d, and past the large-download warning on WebGPU.",
    params: 149,
    dim: 768,
    pooling: "cls",
    upstream: "Alibaba-NLP/gte-modernbert-base",
    task: "feature-extraction",
    bytes: { webgpu: 298_363_618, wasm: 150_218_016 },
  },
];

export const DEFAULT_EMBED_MODEL = EMBED_MODELS[0].id;

/**
 * Texts for `/text-features` — short, so the vector strip is about the model
 * rather than about a wall of prose.
 */
export const FEATURE_TEXT_SAMPLES: TextSample[] = [
  {
    id: "webgpu",
    label: "A technical sentence",
    text: "WebGPU exposes the graphics card to a web page as a compute device.",
    hint: "The page's own subject, and a useful anchor for the similarity page next door.",
  },
  {
    id: "cooking",
    label: "A recipe line",
    text: "Fold the melted butter into the flour until no dry patches remain.",
    hint: "Nothing to do with the other samples — its vector should sit far from all of them.",
  },
  {
    id: "short",
    label: "Two words",
    text: "Heavy rain.",
    hint: "A very short input. Mean pooling over two tokens is a different thing from mean pooling over forty.",
  },
];

/** A pair for `/sentence-similarity`, and what it is a test of. */
export interface PairSample {
  id: string;
  label: string;
  a: string;
  b: string;
  /** What the honest answer looks like — including where the model is wrong. */
  hint: string;
}

/**
 * Pairs chosen so the page has something to say, including where these models
 * fail.
 *
 * `negation` is the important one. Embedding models score a sentence and its
 * negation as *very* similar — they share almost every token and the training
 * objective never had to separate them — so a high score there is a real
 * limitation of the whole approach rather than a bug in this page, and it is
 * the single most useful thing a similarity demo can show.
 */
export const PAIR_SAMPLES: PairSample[] = [
  {
    id: "paraphrase",
    label: "A paraphrase",
    a: "A man is playing a guitar on the street.",
    b: "A busker is performing with his guitar outdoors.",
    hint: "Almost no words in common, the same meaning. This is what an embedding buys you over keyword matching.",
  },
  {
    id: "unrelated",
    label: "Unrelated",
    a: "A man is playing a guitar on the street.",
    b: "The quarterly figures were restated after the audit.",
    hint: "The floor. If this does not sit far below the paraphrase, the embeddings have collapsed.",
  },
  {
    id: "negation",
    label: "A negation",
    a: "The flight to Berlin was cancelled.",
    b: "The flight to Berlin was not cancelled.",
    hint: "Opposite meanings, one word apart. Expect a very high score — that is a limitation of embeddings, not of this page.",
  },
  {
    id: "overlap",
    label: "Shared words, different sense",
    a: "The bank raised its interest rates again.",
    b: "We sat on the bank and watched the river.",
    hint: "Lexical overlap with no shared meaning — the case keyword search gets wrong and an embedding should not.",
  },
];

// --- Translation -------------------------------------------------------------

/**
 * One Marian language pair.
 *
 * **A pair is a model, so changing the pair is a LOAD.** There is no language
 * argument anywhere in this task: a Marian checkpoint carries its own direction
 * and `tr(text)` takes nothing else. So the direction control is a *model
 * selector*, it lives in SELECT, and the page says that switching direction is
 * another ~200 MB download. Getting that wrong would make en→de and de→en look
 * free, which is the single most likely misreading of the page.
 */
export interface TranslationModel extends TextModel {
  task: "translation";
  /** BCP-47-ish source and target, for the direction label and the samples. */
  source: string;
  target: string;
  /** "English → German". The row's name, rather than the repo id. */
  direction: string;
}

/**
 * §3.5's six pairs — en↔de, en↔fr, en→es, en→zh.
 *
 * Six rather than two so the "specialists beat one generalist" argument is
 * visible: the whole catalogue here is 1.25 GB across six directions, against
 * `Xenova/nllb-200-distilled-600M` at **894.6 MB for one download** (and
 * `mbart-large-50-many-to-many-mmt` at 872.5 MB). A user who wants one pair
 * pays a quarter of NLLB and gets a better translation for it. Neither
 * multilingual model is offered, because both are over §0's bar.
 *
 * **Every entry pins its WASM precision, and that is a measurement.** A Marian
 * decoder cannot be quantized on the WASM provider bundled with 4.2.0 — the
 * session simply does not open (`qdq_actions.cc:137 … Missing required scale:
 * model.shared.weight_merged_0_scale`, measured 2026-09-25 in Chromium). Marian
 * is the *third* family to hit it after Whisper and Donut, so the spec lives in
 * `model/backend.ts` as `SEQ2SEQ_WASM_DTYPES` rather than being written out
 * again here. It costs 2.7x on the CPU path — 101 MB at a uniform q8 against
 * 271 MB like this — and the alternative is no CPU path at all.
 *
 * That also settles the open question the plan left for Phase 2. The plan
 * worried that en↔de (199.6 MiB at fp16) would slip under `LARGE_MODEL_BYTES`
 * while en→es (213.1 MiB) crossed it, leaving one warning on a page of
 * identical models. With the pin, the WASM download is 271–289 MB for every
 * pair and `sizeEstimate` keys `large` off the **bigger** of the two — so every
 * pair warns, consistently, and the number it warns about is one the user will
 * actually pay. The inconsistency was an artefact of a WASM path that does not
 * exist.
 */
export const TRANSLATION_MODELS: TranslationModel[] = [
  {
    id: "Xenova/opus-mt-en-de",
    label: "English → German",
    direction: "English → German",
    source: "en",
    target: "de",
    hint: "The default. A Marian specialist: one direction, ~200 MB, and no language argument.",
    params: 74,
    task: "translation",
    graphs: ["encoder_model", "decoder_model_merged"],
    dtypes: { wasm: SEQ2SEQ_WASM_DTYPES },
    bytes: { webgpu: 209_271_625, wasm: 271_047_378 },
  },
  {
    id: "Xenova/opus-mt-de-en",
    label: "German → English",
    direction: "German → English",
    source: "de",
    target: "en",
    hint: "The reverse direction, and a second download — not a toggle on the one above.",
    params: 74,
    task: "translation",
    graphs: ["encoder_model", "decoder_model_merged"],
    dtypes: { wasm: SEQ2SEQ_WASM_DTYPES },
    bytes: { webgpu: 209_271_625, wasm: 271_047_378 },
  },
  {
    id: "Xenova/opus-mt-en-fr",
    label: "English → French",
    direction: "English → French",
    source: "en",
    target: "fr",
    hint: "Same architecture, a different pair of vocabularies.",
    params: 74,
    task: "translation",
    graphs: ["encoder_model", "decoder_model_merged"],
    dtypes: { wasm: SEQ2SEQ_WASM_DTYPES },
    bytes: { webgpu: 212_168_275, wasm: 274_670_310 },
  },
  {
    id: "Xenova/opus-mt-fr-en",
    label: "French → English",
    direction: "French → English",
    source: "fr",
    target: "en",
    hint: "The reverse of the pair above, and again a separate checkpoint.",
    params: 74,
    task: "translation",
    graphs: ["encoder_model", "decoder_model_merged"],
    dtypes: { wasm: SEQ2SEQ_WASM_DTYPES },
    bytes: { webgpu: 212_168_275, wasm: 274_670_310 },
  },
  {
    id: "Xenova/opus-mt-en-es",
    label: "English → Spanish",
    direction: "English → Spanish",
    source: "en",
    target: "es",
    hint: "A larger target vocabulary, so a slightly larger download for the same architecture.",
    params: 78,
    task: "translation",
    graphs: ["encoder_model", "decoder_model_merged"],
    dtypes: { wasm: SEQ2SEQ_WASM_DTYPES },
    bytes: { webgpu: 223_416_625, wasm: 288_738_978 },
  },
  {
    id: "Xenova/opus-mt-en-zh",
    label: "English → Chinese",
    direction: "English → Chinese",
    source: "en",
    target: "zh",
    hint: "A non-Latin script, which is where a specialist's own tokenizer earns its place.",
    params: 78,
    task: "translation",
    graphs: ["encoder_model", "decoder_model_merged"],
    dtypes: { wasm: SEQ2SEQ_WASM_DTYPES },
    bytes: { webgpu: 223_416_625, wasm: 288_738_978 },
  },
];

export const DEFAULT_TRANSLATION_MODEL = TRANSLATION_MODELS[0].id;

/**
 * What one Marian download buys against what a multilingual model would cost,
 * in bytes, so the page can state the trade-off rather than describe it.
 *
 * `Xenova/nllb-200-distilled-600M`, summing `encoder_model` +
 * `decoder_model_merged` only — never the alternative `decoder_model` /
 * `decoder_with_past_model` the same repo also publishes.
 *
 * **1.76 GB, not the 894.6 MB §3.5 quotes**, and the difference is §1.1's
 * finding one more time: 894.6 MB is NLLB's *q8* size, and q8 is not what this
 * page loads. `loadOpts()` asks WebGPU for fp16 — 1 760 444 340 bytes — and the
 * CPU path cannot use a quantized decoder at all (`SEQ2SEQ_WASM_DTYPES`), so
 * there is no configuration in which a browser pays 895 MB for it. The
 * multilingual option is further over §0's bar than the roadmap thought, which
 * makes the one-pair-at-a-time design more clearly right rather than less.
 */
export const NLLB_BYTES = 1_760_444_340;

/** Sample text per source language, so a pair always has something to run. */
export const TRANSLATION_SAMPLES: Record<string, TextSample[]> = {
  en: [
    {
      id: "en-webgpu",
      label: "Technical",
      text: "WebGPU lets a web page run compute shaders directly on the graphics card.",
      hint: "Domain vocabulary a general model tends to paraphrase away.",
    },
    {
      id: "en-idiom",
      label: "An idiom",
      text: "They decided to bite the bullet and rewrite the whole thing from scratch.",
      hint: "Idioms are where a small specialist and a large generalist part company.",
    },
    {
      id: "en-plain",
      label: "Plain",
      text: "The meeting has been moved to Thursday at eleven, in the small room.",
      hint: "The easy case, and the one worth checking first.",
    },
  ],
  de: [
    {
      id: "de-plain",
      label: "Plain",
      text: "Die Besprechung wurde auf Donnerstag um elf Uhr verlegt.",
      hint: "The reverse of the English plain sample — useful for checking a round trip.",
    },
    {
      id: "de-compound",
      label: "A compound",
      text: "Die Geschwindigkeitsbeschränkung auf der Autobahn wurde nicht aufgehoben.",
      hint: "German compounds test the tokenizer more than the model.",
    },
  ],
  fr: [
    {
      id: "fr-plain",
      label: "Plain",
      text: "La réunion a été déplacée à jeudi onze heures, dans la petite salle.",
      hint: "The reverse of the English plain sample.",
    },
    {
      id: "fr-negation",
      label: "A negation",
      text: "Il n'a jamais dit qu'il ne viendrait pas à la conférence.",
      hint: "Double negation, which small models routinely flatten.",
    },
  ],
};

// --- Summarization -----------------------------------------------------------

/**
 * A summarization checkpoint.
 *
 * `hint` carries the quality caveat where there is one, because on this page
 * "worse than three sentences of the article" is a real and common outcome and
 * the page is built to show it rather than hide it.
 */
export interface SummarizerModel extends TextModel {
  task: "summarization";
  /** What it was fine-tuned on — which is what its output will sound like. */
  domain: string;
}

/**
 * §3.6's catalogue, and the outcome of the plan's Phase 0 gate.
 *
 * **The gate: does distilbart open a q8 session on WebGPU?** Measured in
 * Chromium on 2026-09-25 — **yes**, and it produces a correct summary. Which is
 * the only reason this page exists, because every other configuration is over
 * §0's ~500 MB bar:
 *
 *   distilbart-cnn-6-6, q8 on WebGPU    283.9 MB   opens, correct output  ✓
 *   distilbart-cnn-6-6, fp16 on WebGPU  563.6 MB   over the bar
 *   distilbart-cnn-6-6, q8 on WASM      283.9 MB   **session will not open**
 *   distilbart-cnn-6-6, enc q8 + dec fp32 on WASM  742.8 MB   over the bar
 *
 * The third line is `SEQ2SEQ_WASM_DTYPES` again: BART is the **fourth** family
 * to hit the bundled WASM provider's quantized-decoder failure, after Whisper,
 * Donut and Marian. Unlike Marian, the fp32-decoder fallback does not fit — so
 * **distilbart has no CPU path at all** and declares `backends: ["webgpu"]`,
 * which `useBackendProbe` turns into a disabled row with the reason on it
 * rather than a failed download.
 *
 * That would have left the page with no floor, so the catalogue opens with
 * **`Xenova/t5-small`**, which the plan did not consider: 154.4 MB at fp16 on
 * WebGPU and 202.5 MB on WASM with the seq2seq pin, both inside the bar, and
 * measured working on the CPU path at ~220 ms a summary. It is a *much* weaker
 * summarizer than distilbart — and on this page that is not a drawback. The
 * page's subject is the lead-3 baseline, and a model that visibly loses to three
 * sentences of the article makes that lesson concrete rather than hypothetical.
 *
 * The plan's `distilbart-xsum-12-1` and `distilbart-cnn-12-6` are left out: both
 * are over the bar at fp16 and neither adds anything t5-small and
 * distilbart-cnn-6-6 do not already cover between them.
 */
export const SUMMARIZER_MODELS: SummarizerModel[] = [
  {
    id: "Xenova/t5-small",
    label: "T5-small",
    hint: "The default and the floor — 154 MB, runs on CPU too. Genuinely weak, which is this page's point.",
    domain: "a multi-task mixture (the `summarize:` prefix is one of its tasks)",
    params: 60,
    task: "summarization",
    graphs: ["encoder_model", "decoder_model_merged"],
    // A measurement: a quantized seq2seq decoder cannot open a session on the
    // bundled WASM provider (`SEQ2SEQ_WASM_DTYPES`). Verified working in this
    // configuration — 23.5 s to load, 218–277 ms a summary.
    dtypes: { wasm: SEQ2SEQ_WASM_DTYPES },
    bytes: { webgpu: 154_350_057, wasm: 202_472_616 },
  },
  {
    id: "Xenova/distilbart-cnn-6-6",
    label: "DistilBART CNN 6-6",
    hint: "A real news summarizer, and GPU-only: its CPU session cannot be quantized and the unquantized one is 743 MB.",
    domain: "CNN/DailyMail news articles",
    params: 306,
    task: "summarization",
    graphs: ["encoder_model", "decoder_model_merged"],
    // **A measurement, and the plan's Phase 0 gate.** q8 on WebGPU opens and
    // summarizes correctly; fp16 would be 563.6 MB, over §0's bar. Latency on a
    // real GPU is still unmeasured — the measurement box had no GPU with
    // `shader-f16`, only SwiftShader — so `just fe-e2e-summarize` re-measures it
    // where there is one.
    dtypes: { webgpu: "q8" },
    // No WASM path: q8 will not open, and encoder-q8 + decoder-fp32 is 742.8 MB.
    backends: ["webgpu"],
    bytes: { webgpu: 283_921_904 },
  },
  {
    id: "Xenova/bart-large-cnn",
    label: "BART-large CNN",
    hint: "The model everyone benchmarks against, at 463 MB and GPU-only. The quality ceiling this page can reach.",
    domain: "CNN/DailyMail news articles",
    params: 406,
    task: "summarization",
    graphs: ["encoder_model", "decoder_model_merged"],
    // Same pin, same reason, and it is worth being exact about what kind of
    // claim it is: q8-on-WebGPU was **measured on distilbart**, which is the
    // same architecture from the same export tooling, and is an *inference*
    // here rather than a second measurement. `just fe-e2e-summarize` on a real
    // GPU is what would turn it into one.
    dtypes: { webgpu: "q8" },
    backends: ["webgpu"],
    bytes: { webgpu: 462_535_837 },
  },
];

export const DEFAULT_SUMMARIZER = SUMMARIZER_MODELS[0].id;

/** Sentences of article to quote as the baseline. Three, as the name says. */
export const LEAD_N = 3;

/** Generation defaults. Both are **run** parameters — changing one re-runs. */
export const SUMMARY_MAX_TOKENS = 130;
export const SUMMARY_MIN_TOKENS = 30;

/**
 * The entailment model the faithfulness check borrows.
 *
 * Not a new download for the category — it is `/zero-shot-classification`'s
 * cheapest entry, reused, which is why the second opt-in here costs 27–50 MB
 * rather than another 300. Scoring a summary sentence against the article is
 * exactly a one-label NLI call: `softmaxEach` is true when `labels.length === 1`,
 * so the pipeline returns entailment-against-contradiction for that single
 * hypothesis, which is the number wanted.
 */
export const FAITHFULNESS_MODEL = "Xenova/mobilebert-uncased-mnli";

/**
 * Articles chosen so the lead-3 baseline is *hard*.
 *
 * News is the genre every one of these checkpoints was fine-tuned on, and the
 * inverted-pyramid convention puts the answer in the first three sentences — so
 * a sample set of rambling prose would make the neural summary look better than
 * it is. `fabricated` is the one that exists to be failed: it contains a number
 * and a name close enough together that small summarizers routinely attach the
 * wrong one to the other, which is what the faithfulness check is for.
 */
export interface ArticleSample {
  id: string;
  label: string;
  text: string;
  hint: string;
}

export const ARTICLE_SAMPLES: ArticleSample[] = [
  {
    id: "launch",
    label: "A news report",
    hint: "Inverted pyramid: the answer is in sentence one. The hardest case for a neural summarizer to beat.",
    text:
      "The European Space Agency confirmed on Tuesday that its Ariane 6 rocket had completed a " +
      "second successful commercial launch, placing four satellites into low Earth orbit. " +
      "Officials said the flight validated the upper-stage restart sequence that had failed " +
      "during a demonstration mission last year. The agency expects to raise the launch cadence " +
      "to roughly one flight a month by the end of next year. That would ease a shortage of " +
      "European launch capacity which had forced several operators to book rides on American " +
      "rockets. Arianespace said three further commercial payloads are already contracted for " +
      "the first half of next year, including two Earth-observation satellites for the " +
      "Copernicus programme.",
  },
  {
    id: "buried",
    label: "The point is buried",
    hint: "The lead is scene-setting and the news is in the fourth sentence — where lead-3 finally loses.",
    text:
      "The conference hall in Lisbon filled slowly on Thursday morning, delegates drifting in " +
      "with paper cups of coffee. Panels on grid storage and permitting had drawn modest " +
      "crowds all week. The mood had been one of cautious routine. Then, shortly before lunch, " +
      "the Portuguese energy minister announced that the government would abandon its planned " +
      "auction for two gigawatts of offshore wind, citing costs that had risen by more than " +
      "forty per cent since the tender was drafted. Developers who had spent two years " +
      "preparing bids learned of the decision from the stage. Shares in the two largest " +
      "bidders fell sharply within the hour.",
  },
  {
    id: "fabricated",
    label: "A name and a number",
    hint: "Two people and two figures, close together. Small summarizers attach the wrong number to the wrong name — which is what the faithfulness check is for.",
    text:
      "The audit found that Marta Reyes, the department's procurement lead, approved 14 " +
      "contracts above the delegated threshold during the period under review. Her deputy, " +
      "Tomas Keller, approved 3. Investigators said the 14 approvals accounted for 8.2 million " +
      "euros of the 9.1 million euros examined. Reyes told the panel she had believed the " +
      "threshold had been raised the previous year. Keller said he had escalated every case he " +
      "was unsure about. The report recommends that both approval limits be reset and that a " +
      "second signature be required above 250,000 euros.",
  },
];
