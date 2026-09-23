# NLP Models in the Browser (WebGPU or CPU)

> The **Natural Language Processing** category of
> [`taskTaxonomy.ts`](../../frontend/src/components/layout/taskTaxonomy.ts), task by task:
> what runs client-side on the user's GPU (WebGPU) or CPU (WebAssembly), and which
> checkpoint to use. No Python server in the inference path.
>
> This file began as issue [#5](https://github.com/bthek1/model_playground/issues/5) and
> moved here when its first route shipped, the same way
> [`audio.md`](audio.md), [`vision.md`](vision.md), [`graph.md`](graph.md) and
> [`multimodal.md`](multimodal.md) did: a roadmap that documents *shipped* code has to be
> reviewable in the same pull request as the code it describes, which an issue body
> cannot be.

| Roadmap section | Route | Status |
|---|---|---|
| Text Classification (§3.1) | [`/text-classification`](../../frontend/src/routes/text-classification.tsx) | **Shipped** — three sentiment heads, the head-to-head, 68–250 MB |
| Token Classification (§3.2) | — | Planned — [#39](https://github.com/bthek1/model_playground/issues/39) |
| Table Question Answering (§3.11) | — | **Does not port** — no export for TAPAS or TAPEX; text-to-SQL needs ~1 GB *and* a database |
| Question Answering (§3.3) | — | Planned — [#40](https://github.com/bthek1/model_playground/issues/40) |
| Zero-Shot Classification (§3.4) | — | Planned — [#42](https://github.com/bthek1/model_playground/issues/42) |
| Translation (§3.5) | — | Planned — [#45](https://github.com/bthek1/model_playground/issues/45) |
| Summarization (§3.6) | — | Planned, **gated on a measurement** — [#46](https://github.com/bthek1/model_playground/issues/46) |
| Feature Extraction · Sentence Similarity (§3.7) | — | Planned — [#43](https://github.com/bthek1/model_playground/issues/43) |
| Text Generation (§3.8) | — | Planned — [#47](https://github.com/bthek1/model_playground/issues/47) |
| Fill-Mask (§3.9) | — | Planned — [#41](https://github.com/bthek1/model_playground/issues/41) |
| Text Ranking (§3.10) | — | Planned — [#44](https://github.com/bthek1/model_playground/issues/44) |

**The browser is an encoder paradise and a decoder compromise.** BERT-family encoders are
20 MB to 400 MB and run a single forward pass, so classification, NER, extractive QA,
embeddings and reranking are effectively free. Generative decoders are autoregressive: a
0.6B model at `q4f16` on WebGPU produces maybe 20 to 40 tokens per second on a laptop,
which is usable but is not a hosted API. Rule of thumb: **if the task ends in a single
forward pass over the input, build the page. If it generates tokens one at a time, cap
the model well under 0.6B and stream.**

---

## 0. The feasibility bar this file is filtered by

Every row has to clear two tests before it becomes a page, and this file has been swept
against both:

1. **It runs client-side** — on WebGPU where the operators are covered, on the WASM
   provider where they are not. A CPU-only path is fine when CPU is the right engineering
   answer. What is *not* fine is a row that needs a server.
2. **Its cheapest usable checkpoint is under ~500 MB**, measured off the Hub's blob
   listing rather than estimated, summing only the graphs a page actually loads —
   `encoder_model` + `decoder_model_merged`, never the alternative `decoder_model` /
   `decoder_with_past_model` a seq2seq repo also publishes.

**Six models were removed by that sweep**, and the measurements are the news, because
five of the six were quoted in the original roadmap at sizes that were simply wrong:

| Model | The roadmap said | Measured | § |
|---|---|---|---|
| `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | ~180 MB, **"the better default"** | **738.6 MB** (fp32-only; no q8 exists) | 3.4 |
| `Xenova/LaMini-Flan-T5-783M` | ~300 MB q8 | **822.7 MB** | 3.6 |
| `Xenova/nllb-200-distilled-600M` | ~650 MB q8 | **894.6 MB** | 3.5 |
| `Xenova/mbart-large-50-many-to-many-mmt` | ~700 MB q8 | **872.5 MB** | 3.5 |
| `onnx-community/Qwen3-Embedding-0.6B-ONNX` | ~600 MB | **613.5 MB** | 3.7 |
| `onnx-community/Qwen3-0.6B-ONNX` | ~400 MB q4f16 | **569.8 MB** | 3.1, 3.3, 3.8 |

**No page was lost.** Every affected section keeps a default between 22 MB and 284 MB,
which is the test that matters: a page needs a cheap floor, not a strong ceiling.
Translation loses its multilingual option and becomes one pair at a time — which is
*cheaper* per session, not merely smaller.

Two habits come out of that table. **An official in-repo export is not automatically a
quantized one** — read the blob listing. And **never infer a seq2seq download from the
parameter count**, because the repo publishes three alternative decoders and only one of
them is ever loaded.

---

## 1. Three findings that hold across the whole category

### 1.1 Every size table in the original roadmap was the *wrong backend's* number

The roadmap quoted `q8` throughout. `loadOpts()` asks for **fp16 on WebGPU**, which is
what any machine with a GPU adapter gets. The real default download is roughly double,
all the way down:

| model | q8 | **fp16** | § |
|---|---|---|---|
| `Xenova/distilbert-base-uncased-finetuned-sst-2-english` | 64.5 MiB | **127.9 MiB** | 3.1 |
| `Xenova/twitter-roberta-base-sentiment-latest` | 120.1 MiB | **238.1 MiB** | 3.1 |
| `Xenova/finbert` | 105.6 MiB | **209.2 MiB** | 3.1 |
| `Xenova/bert-base-NER` | 103.9 MiB | **205.8 MiB** | 3.2 |
| `Xenova/nli-deberta-v3-xsmall` | 83.2 MiB | **136.2 MiB** | 3.4 |
| `Xenova/opus-mt-en-de` | 101.1 MiB | **199.6 MiB** | 3.5 |
| `Xenova/distilbart-cnn-6-6` | 270.8 MiB | **537.5 MiB** | 3.6 |
| `Xenova/all-MiniLM-L6-v2` | 21.9 MiB | **43.2 MiB** | 3.7 |

Two consequences. Several entries cross `LARGE_MODEL_BYTES` (200 MB) on WebGPU while
reading as comfortably under it in the roadmap — the guardrail handles that correctly,
because `sizeEstimate` keys `large` off the **bigger** of the two downloads. And
**Summarization's floor moves from 283.9 MB to 563.6 MB**, over the bar, which is why
§3.6 exists only if its WebGPU dtype is pinned and measured.

So: **every entry in `text/catalogue.ts` carries measured `bytes` for both backends**, a
stricter rule than vision's "measure where an estimate would mislead", and
`just fe-e2e-models` re-checks the numbers against the Hub rather than trusting them.

### 1.2 `q4` is not a lever for an encoder

On every encoder measured for this category, `model_q4.onnx` is *larger* than
`model_quantized.onnx` (`distilbert-sst-2`: 118.9 MiB q4 against 64.5 MiB q8), and
`q4f16` is usually larger too. **4-bit is a decoder format.** No encoder page should
reach past the `loadOpts()` default; only §3.8 has anything to gain from it.

### 1.3 There is no `textLoadOpts()`, deliberately

The `asrLoadOpts` / `vlmLoadOpts` precedent is for a precision decision that holds across
a whole *family*, and the measurements do not support one here: q8-on-WebGPU is right for
a seq2seq summarizer and wrong by default for a 22 MB embedder. Precision that differs
from `loadOpts()` is expressed **per catalogue entry** with `dtypes`, as vision already
does — and per the repo's standing rule, each pin's comment says whether it is a
**measurement or a precaution**. Only one of those is evidence. Generalise into a helper
only if a third page needs the same override.

---

## 2. The module

`src/text/` is the NLP counterpart of `src/vision/`, and deliberately the same shape:

| file | what it is |
|---|---|
| [`types.ts`](../../frontend/src/text/types.ts) | the catalogue entry type and the worker protocol |
| [`engine.ts`](../../frontend/src/text/engine.ts) | a **pure** message handler — no runtime import, so it unit-tests without a download |
| [`pipeline.worker.ts`](../../frontend/src/text/pipeline.worker.ts) | the thin wrapper, the only file importing `@huggingface/transformers` |
| [`client.ts`](../../frontend/src/text/client.ts) | the worker factory, so hooks are mockable under happy-dom |
| [`catalogue.ts`](../../frontend/src/text/catalogue.ts) | every entry, with measured `bytes` for both backends |

Plus [`hooks/useTextPipeline.ts`](../../frontend/src/hooks/useTextPipeline.ts) over
`model/useModelWorker.ts`, and [`components/text/`](../../frontend/src/components/text/)
for the two shared output components.

**One worker for the whole modality, and the task travels in `load`** — the same rule as
`vision.worker.ts`. Text **generation** is the one exception and gets its own worker,
because it streams (the `useAsr` precedent: the task that owns a loop owns its worker).

**This is the cheapest module in the app to have built, and the reason is instructive.**
There is no text equivalent of `audio/io.ts` or `vision/image.ts`: the input is already a
string, so there is no decode step, no preprocessing, and no transport problem. A
`RawImage` does not survive `postMessage` and a `Tensor` throws outright; a string crosses
the wire as itself. That absence is why §3.1 is the category's first page rather than a
more impressive one.

`engine.ts` owes the three behaviours every engine in this repo owes: **one model live at
a time** (null the reference *first*, then `disposeQuietly`), **warm-up before `ready`**
(one throwaway inference, posting `{ status: "warmup" }`, never failing the load), and
**never block the main thread**.

### 2.1 The two shared components

- **[`ScoreList`](../../frontend/src/components/text/ScoreList.tsx)** — a sorted bar per
  label with the score printed. **It refuses to render a single row**, and that is the
  whole design: a classifier's argmax is the least informative thing it produces, because
  "POSITIVE" alone looks identical at 0.99 and at 0.51. Callers pass the full label set,
  and a near-tie is called out in words rather than left to be read off two bar widths.
- **`SpanOverlay`** + `highlight()` — built in §3.2, where it can be tested against a real
  model's character offsets rather than against our expectations, and reused by §3.3 and
  §3.9. Its one rule: **slice the original string by character offset, never rebuild the
  text from tokens.** Concatenated subwords lose the original whitespace and the highlight
  lands a character or two off — a wrong result that reads as a styling problem.

### 2.2 Nothing is debounced, anywhere in this category

The original roadmap's §5 wants live classification on a 200–300 ms pause. The
page-pattern rule wins and is absolute: **typing is INPUT, and only GENERATE spends.** A
debounced auto-run is the five-samples-five-inferences failure with a timer in front of
it.

---

## 3. Task by task

### 3.1 Text Classification — **shipped**

Taxonomy task **Text Classification** ·
[`/text-classification`](../../frontend/src/routes/text-classification.tsx) ·
[#38](https://github.com/bthek1/model_playground/issues/38).

The smallest useful page in the app: a string in, a score list out, no decode step at all.

| entry | q8 | fp16 | domain |
|---|---|---|---|
| `Xenova/distilbert-base-uncased-finetuned-sst-2-english` | 64.5 MiB | 127.9 MiB | film reviews (SST-2) — the default, 2 classes |
| `Xenova/twitter-roberta-base-sentiment-latest` | 120.1 MiB | 238.1 MiB | tweets — 3 classes, the only one with `neutral` |
| `Xenova/finbert` | 105.6 MiB | 209.2 MiB | financial news — 3 classes |

**`onnx-community/ModernBERT-base-ONNX` was cut from this page's catalogue, against the
original roadmap's own table**, and the reason is the roadmap's own note beside it: *"the
base model, so fine-tune before it classifies anything"*. That note is the
disqualification. A base encoder has no trained classification head, so on a
classification page it emits `LABEL_0` / `LABEL_1` from randomly initialised weights — a
confident, fluent, meaningless answer with nothing failing on the way there. It is the
exact class of silent wrongness this repo pins tests against, and a page is not the place
to demonstrate it. `Xenova/finbert` takes the slot: a real fine-tune, a third domain, and
it makes the head-to-head demonstrable. ModernBERT keeps its place on §3.9, where a base
model is exactly what is wanted.

**The head-to-head is the page, and it is a second LOAD rather than a free toggle.**
Running one sentence through two of these shows that "domain-matched" is a claim about a
*specific* domain, and that a model outside its domain is confidently wrong rather than
uncertain. A second model is a second download and a second model in memory, so the page
quotes the cost before the click. The samples are chosen so the three models **disagree**
— a sample set every model gets right demonstrates nothing about any of them.

`just fe-e2e-text` asserts a known label on a known sentence, and pins the head-to-head
*structurally*: SST-2's head has two classes and FinBERT's has three, so a comparison that
quietly renders one model's answer twice cannot pass. Asserting that the two *rankings*
differ would pin a property neither model promises.

### 3.2 Token Classification — planned ([#39](https://github.com/bthek1/model_playground/issues/39))

NER with the entities highlighted in the user's own text, and a redact button. The
redaction is a real argument for client-side inference rather than a demonstration of one:
the document never leaves the tab.

| entry | q8 | fp16 | role |
|---|---|---|---|
| `Xenova/bert-base-NER` | 103.9 MiB | 205.8 MiB | the default — 4 types, English |
| `Xenova/bert-base-multilingual-cased-ner-hrl` | 178.5 MiB | 354.9 MiB | 10 languages |

**The original roadmap is wrong that multilingual NER is lost.** It sends the task to a
server on the strength of two repos with no export; the runtime's own *default* for
`token-classification` is the multilingual entry above, which is under the bar.

`aggregation_strategy: "simple"` is load-bearing and fails as a rendering bug: without it
the pipeline returns one result per subword token and the page paints a highlight per
word-piece, which looks like a broken overlay rather than a missing option.

### 3.3 Question Answering — planned ([#40](https://github.com/bthek1/model_playground/issues/40))

Extractive QA at **62.8 MiB q8 / 124.5 MiB fp16**
(`Xenova/distilbert-base-cased-distilled-squad`) — the only exported entry, so this is a
one-entry page, and §0's second question is satisfied by the floor being 63 MB rather than
by there being an alternative.

**The model cannot abstain, and saying so is a first-class requirement.** SQuAD 1.1 models
always answer; the squad2 checkpoints that can say "no answer" have no export, so the
behaviour is not merely unshipped but unavailable. The page ships a sample whose question
is unanswerable from its passage, and the model answering it anyway — with a score — is
the lesson.

### 3.4 Zero-Shot Classification — planned ([#42](https://github.com/bthek1/model_playground/issues/42))

| entry | q8 | fp16 | role |
|---|---|---|---|
| `Xenova/nli-deberta-v3-xsmall` | 83.2 MiB | 136.2 MiB | the default |
| `Xenova/mobilebert-uncased-mnli` | 25.7 MiB | 47.8 MiB | the smallest that works at all |
| `Xenova/distilbert-base-uncased-mnli` | 67.6 MiB | 134.1 MiB | the runtime's own default |
| `Xenova/bart-large-mnli` | 392.2 MiB | **778.1 MiB** | the classic — gated on the fp16 number |

**N labels cost N forward passes**, because the model runs once per label as an NLI
premise–hypothesis pair. The pass count is shown next to GENERATE, derived from the label
list as it is edited.

### 3.5 Translation — planned ([#45](https://github.com/bthek1/model_playground/issues/45))

One Marian pair at a time, ~101 MiB q8 / ~200 MiB fp16 each. **A pair is a model, so
changing the pair is a LOAD** — a Marian checkpoint carries its own language pair and
`tr(text)` takes nothing else. Getting that wrong would make en→de and de→en look free,
which is the single most likely misreading of the page.

### 3.6 Summarization — planned, and **gated on a measurement** ([#46](https://github.com/bthek1/model_playground/issues/46))

Every BART summarizer exported for the browser is over the bar at fp16
(`distilbart-cnn-6-6`: 537.5 MiB). The page exists only if its WebGPU entries pin
`dtype: "q8"` — 283.9 MB, comfortably inside — **and that pin is a measurement rather than
a precaution.** q8-on-WebGPU is exactly the combination `/super-resolution` measured as
*worse than not running the model at all*.

The lead-3 baseline (`text.split(/(?<=[.!?])\s/).slice(0, 3)`) travels beside every
summary, per `/graph-classification`'s rule that a metric owes its null model on screen.

### 3.7 Feature Extraction · Sentence Similarity — planned ([#43](https://github.com/bthek1/model_playground/issues/43))

| entry | dim | q8 | fp16 |
|---|---|---|---|
| `Xenova/all-MiniLM-L6-v2` | 384 | **21.9 MiB** | 43.2 MiB |
| `Xenova/all-mpnet-base-v2` | 768 | 105.0 MiB | 208.0 MiB |
| `Xenova/bge-base-en-v1.5` | 768 | 105.0 MiB | 208.0 MiB |
| `Alibaba-NLP/gte-modernbert-base` | 768 | 143.3 MiB | 284.5 MiB |

The cheapest floor in the category. **`{ pooling: "mean", normalize: true }` is not
optional**, and omitting it is the page's one silent failure: raw BERT is not an embedding
model, and without the two arguments every similarity collapses into a narrow band near
0.9 — a page that appears to work, with a number that means nothing. The E2E assertion is
a **spread**, not a value.

### 3.8 Text Generation — planned ([#47](https://github.com/bthek1/model_playground/issues/47))

The category's only streaming page, and the payoff for putting `partial` in the **shared**
`ModelResponse` envelope rather than in a private VLM protocol.

**`Xenova/gpt2`'s "128.3 MB at q4f16" is not a file that exists.** Its `model_q4f16.onnx`
is byte-for-byte the size of its fp16 build (GPT-2's `Conv1D` weights are skipped by the
exporter), it publishes no `model_quantized.onnx` at all so `loadOpts("wasm")`'s q8 404s,
and the 128.3 MB file the roadmap quoted belongs to a legacy graph family 4.2.0 only
requests via `model_file_name`. `HuggingFaceTB/SmolLM2-360M-Instruct` at **272.7 MB**
genuinely quantized is the default instead.

### 3.9 Fill-Mask — planned ([#41](https://github.com/bthek1/model_playground/issues/41))

Four entries across three tokenizer families, specifically so the mask-token trap is one
click away rather than theoretical: **always insert `tokenizer.mask_token`, never a
literal string.** Hard-coding `[MASK]` breaks the moment the user switches to RoBERTa, and
it does not error — the model treats the literal characters as words and returns fluent,
wrong predictions.

### 3.10 Text Ranking — planned ([#44](https://github.com/bthek1/model_playground/issues/44))

The most complete page available in the category: **all four retrieval stages run
client-side** over a corpus the user pastes in, and two of the four are arithmetic with no
model at all (BM25, RRF). The cross-encoder is what makes it worth building — it scores a
*pair*, so it cannot be precomputed per document, which is exactly why it reranks the top
20 rather than the whole corpus. That cost is visible in a browser, so the architecture
teaches itself.

**This page holds two models live at once, and that is a declared exception** to the
one-model rule, on `/pose`'s precedent: an embedder plus a reranker is 128 MiB together on
WASM. It inherits `/pose`'s two obligations — one catalogue entry naming both models with
the **combined** download quoted, and `model/progress.ts` keyed on **repo + file** so two
repos publishing the same filename do not overwrite each other's progress row.

### 3.11 Table Question Answering — **does not port**

Taxonomy task **Table Question Answering** · no route, and it should stay that way. The
row falls through to `/tasks/$slug`, pinned by an assertion in `taskTaxonomy.test.ts` so it
cannot be re-mapped without a model to point at.

- **TAPAS** has no ONNX export, and its table-aware position embeddings mean a generic
  encoder is not a substitute.
- **TAPEX** likewise.
- **Text-to-SQL** needs Qwen2.5-Coder-1.5B (~1 GB at `q4`, and slow), and the SQL
  execution step needs a database.

There is a real page here that is worth building one day, just not that one: run the SQL
half in the browser with `sql.js` and let a small LLM write the query. Only the generation
is weak, and a page that shows the generated SQL *before* running it turns that weakness
into the interesting part.

---

## 4. Feasibility summary

| Taxonomy task | In-browser? | Recommended model | Floor |
|---|---|---|---|
| Text Classification | Yes, excellent | `Xenova/distilbert-base-uncased-finetuned-sst-2-english` | 64.5 MiB q8 / 127.9 MiB fp16 |
| Token Classification | Yes | `Xenova/bert-base-NER` | 103.9 MiB q8 |
| Table QA | **No** | — | server, or `sql.js` plus a small LLM |
| Question Answering | Yes, extractive only | `Xenova/distilbert-base-cased-distilled-squad` | 62.8 MiB q8 |
| Zero-Shot Classification | Yes | `Xenova/nli-deberta-v3-xsmall` | 25.7 MiB q8 (mobilebert) |
| Translation | Yes, **one pair at a time** | `Xenova/opus-mt-en-de` per pair | 101.1 MiB q8 |
| Summarization | **Only if q8-on-WebGPU measures well** | `Xenova/distilbart-cnn-6-6` | 270.8 MiB q8 |
| Feature Extraction | Yes, excellent | `Xenova/all-MiniLM-L6-v2` | **21.9 MiB q8** |
| Text Generation | Yes, small models, streamed | `HuggingFaceTB/SmolLM2-360M-Instruct` | 260.1 MiB q4f16 |
| Fill-Mask | Yes, excellent | `Xenova/distilbert-base-uncased` | 64.6 MiB q8 |
| Sentence Similarity | Yes, excellent | `Xenova/all-MiniLM-L6-v2` | 21.9 MiB q8 |
| Text Ranking | Yes, all four stages | `bge-base-en-v1.5` + `ms-marco-MiniLM-L-6-v2` | 127.1 MiB q8 combined |

---

## 5. Memory and performance notes

- **Encoders can share the tab.** The one-model-live rule is about hundreds of megabytes;
  two ~25 MB encoders coexist fine, which is what makes §3.10 possible at all. It is an
  exception a catalogue entry has to *declare*, not one a page takes quietly.
- **Embed the corpus once**, keyed by text. §3.7 and §3.10 cache embeddings in memory, and
  the cache is **cleared on a model change** — a 384-dim MiniLM vector and a 768-dim BGE
  one are not comparable, and a stale cache across a switch is a wrong answer with no
  error attached.
- **Tokenizers are downloaded too.** A 22 MB model still fetches a tokenizer and a config,
  so the LOAD bar reports the aggregate from `model/progress.ts` rather than the weights
  alone.
- **Warm up on load.** One throwaway inference on a short string. On WASM this is JIT time
  the first real query should not pay.
- **Weights cache after the first download**, so a second visit is instant and works
  offline — which changes the LOAD button's words, never its necessity.

---

## 6. Reference

- **Transformers.js pipelines used here**: `text-classification`, `token-classification`,
  `question-answering`, `zero-shot-classification`, `translation`, `summarization`,
  `feature-extraction`, `text-generation`, `fill-mask`. All nine are in `SUPPORTED_TASKS`
  in 4.2.0 — unlike the Multimodal category, nothing here needs a model class instead.
  (There is no `sentence-similarity` or `text-ranking` entry, which is correct: both are
  built from `feature-extraction` and a cross-encoder `text-classification`.)
- **`TextStreamer`** for token-by-token output on §3.8.
- **Page construction**: [`adding-a-task-page.md`](../guides/adding-a-task-page.md) —
  read §0 first; it is cheap and skipping it is not.
- **In-repo standards**: [`model-page-pattern.md`](../standards/model-page-pattern.md)
  (the four-slot contract), [`adding-a-model.md`](../guides/adding-a-model.md) §8.
