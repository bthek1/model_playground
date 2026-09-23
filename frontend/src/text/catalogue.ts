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
