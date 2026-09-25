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
| Token Classification (§3.2) | [`/token-classification`](../../frontend/src/routes/token-classification.tsx) | **Shipped** — NER + redaction, and the multilingual head the roadmap thought was lost |
| Table Question Answering (§3.11) | — | **Does not port** — no export for TAPAS or TAPEX; text-to-SQL needs ~1 GB *and* a database |
| Question Answering (§3.3) | [`/question-answering`](../../frontend/src/routes/question-answering.tsx) | **Shipped** — extractive, the answer marked in the passage, 63–125 MB |
| Zero-Shot Classification (§3.4) | [`/zero-shot-classification`](../../frontend/src/routes/zero-shot-classification.tsx) | **Shipped** — your own labels, N labels cost N passes, 26–816 MB |
| Translation (§3.5) | [`/translation`](../../frontend/src/routes/translation.tsx) | **Shipped** — six pairs, one at a time; a pair is a model, 209–289 MB |
| Summarization (§3.6) | — | Planned, **gated on a measurement** — [#46](https://github.com/bthek1/model_playground/issues/46) |
| Feature Extraction · Sentence Similarity (§3.7) | [`/text-features`](../../frontend/src/routes/text-features.tsx) · [`/sentence-similarity`](../../frontend/src/routes/sentence-similarity.tsx) | **Shipped** — one engine, two rows; the pooling is catalogue data, 22–284 MB |
| Text Generation (§3.8) | — | Planned — [#47](https://github.com/bthek1/model_playground/issues/47) |
| Fill-Mask (§3.9) | [`/fill-mask`](../../frontend/src/routes/fill-mask.tsx) | **Shipped** — four base encoders, three tokenizer families, 68–300 MB |
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
- **[`SpanOverlay`](../../frontend/src/components/text/SpanOverlay.tsx)** +
  [`highlight()`](../../frontend/src/text/highlight.ts) — built by §3.2, against a real
  model's character offsets rather than against our expectations, and reused by §3.3 and
  §3.9. Its one rule: **slice the original string by character offset, never rebuild the
  text from tokens.** Concatenated subwords lose the original whitespace and the highlight
  lands a character or two off — a wrong result that reads as a styling problem. The
  assertion that pins it is that the concatenation of every slice equals the input
  **exactly**, whitespace included.
- **[`mask.ts`](../../frontend/src/text/mask.ts)** — added by §3.9, and the same rule in a
  third disguise: **the filling is spliced into the user's own string**, never taken from
  the pipeline's `sequence`, which is a `tokenizer.decode(…)` and on an uncased model comes
  back as "the capital of france is paris." Everything here takes the mask token as an
  *argument*: nothing in this module, or in any caller, writes `[MASK]`.
  - Overlapping spans are **reported, not interleaved**. Two spans claiming the same
    characters cannot both be drawn, and picking one quietly is how a page ends up showing
    a confident highlight over a range no model proposed. `highlight()` returns them in
    `dropped` and the component says so on screen — rather than throwing mid-render.
  - **Every span carries its type as visible text, not as colour alone**, and that is a
    correctness decision. The four `--entity-*` theme hues are validated all-pairs in both
    themes, but their worst colour-vision separation (ΔE 6.9) sits in the band that is
    legal only *with* a secondary encoding. **Five hues do not pass**: no 5-subset of the
    reference categorical order clears the normal-vision floor with all pairs in play. Four
    is enough because a checkpoint has four entity types and only one model is live at a
    time — which is why `MISC` and `DATE` share slot 4. An unrecognised type renders
    neutral rather than getting a generated hue, since off-palette colour is unvalidated
    colour.
- **[`offsets.ts`](../../frontend/src/text/offsets.ts)** — built by §3.3, and the reason
  the rule above is not free. **Transformers.js 4.2.0 produces no character offsets at
  all.** Its tokenizers return `input_ids` and `attention_mask` and nothing else — there is
  no `return_offsets_mapping` — and *both* pipelines that would carry offsets ship with the
  work unwritten: `question-answering` and `token-classification` each declare `start` /
  `end` as **optional** in their types and each has a literal `// TODO` where they would be
  filled in. A caller reading `result.start` type-checks cleanly and receives `undefined`
  at runtime. Measured against 4.2.0 under `onnxruntime-node`, not inferred from the types.
  - `wordPieceOffsets(text, pieces)` rebuilds the alignment by walking the pieces along the
    source, and **returns `null` rather than guessing** when one does not match — an
    `[UNK]`, a lowercasing tokenizer, an accent-stripping normaliser. A page's honest
    fallback is "no highlight", because a near-miss mark reads as a styling bug and is
    invisible to everyone but a careful reader.
  - It is safe for the cased WordPiece checkpoints this category uses
    (`do_lower_case: false`, `strip_accents: null`) — a property of the *checkpoint*, so
    read its `tokenizer_config.json` before adding an entry that depends on a highlight.

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

### 3.2 Token Classification — **shipped**

Taxonomy task **Token Classification** ·
[`/token-classification`](../../frontend/src/routes/token-classification.tsx) ·
[#39](https://github.com/bthek1/model_playground/issues/39).

NER with the entities marked in the user's own text, and a redact button. **This is the
page where "it runs in your browser" stops being a performance claim and becomes the
point**: redacting a document you are not allowed to upload is a real reason to want the
model on this side of the wire, rather than a demonstration of one.

| entry | q8 | fp16 | types | role |
|---|---|---|---|---|
| `Xenova/bert-base-NER` | 103.9 MiB | 205.8 MiB | PER ORG LOC MISC | the default — English CoNLL-2003 |
| `Xenova/bert-base-multilingual-cased-ner-hrl` | 170.2 MiB | 338.5 MiB | PER ORG LOC **DATE** | 10 languages |

**The original roadmap is wrong that multilingual NER is lost.** It sends the task to a
server on the strength of two repos with no ONNX export
(`Babelscape/wikineural-multilingual-ner`, `Jean-Baptiste/roberta-large-ner-english`) —
but the runtime's own *default* for `token-classification` is the multilingual entry
above, comfortably under the bar. A page was planned without it.

**`aggregation_strategy: "simple"` is pinned in the engine, not passed by the hook.**
Without it the pipeline returns one result per *subword token*, so "Wellington" comes back
as `Well` / `##ing` / `##ton` with three sets of offsets and the page paints three
highlights across one word. That is a **rendering bug rather than an error** — it looks
like a broken overlay, not a missing option — and it is one forgetful call site away at
every future caller. Pinning it in `engine.ts` means there is one call site to keep right
instead of all of them.

**Transformers.js 4.2.0 returns no character offsets, and this is the finding of the
page.** The `token-classification` pipeline hands back `entity_group`, `score` and `word` —
that is all. It ships with the work unwritten (`// TODO add start and end?`) and declares
`start`/`end` as **optional** in its own types, so a caller reading `result.start`
type-checks cleanly and receives `undefined` at runtime. Measured against 4.2.0 on
`Xenova/bert-base-NER`, not inferred:

```
[{ entity_group: "PER", score: 0.998, word: "P" },
 { entity_group: "PER", score: 0.984, word: "##riya Raman" },
 { entity_group: "LOC", score: 0.998, word: "Wellington" }, …]
```

Without offsets every span is dropped as invalid and the overlay renders the user's text
with **nothing marked** — a page indistinguishable from a model that found nothing. It
shipped past a green unit suite, because the mock supplied the offsets the real pipeline
never produces, and past the mocked E2E run, which never loads a byte. **Only
`just fe-e2e-text` caught it**, which is the entire argument for that suite.

`locateEntities()` recovers them by walking the source **forward**, matching each returned
word from where the previous one ended. Three things that walk has to get right, each of
which otherwise produces a working-looking page:

- **Forward, never `indexOf` from zero.** A passage naming the same person twice would
  otherwise mark the first occurrence twice.
- **Re-spaced punctuation.** WordPiece decodes `Jones-Smith` as `Jones - Smith`, so an
  exact search returns −1 on text that plainly contains the entity; the fallback makes
  whitespace flexible.
- **`aggregation_strategy: "simple"` does not always merge a word it split.** "Priya
  Raman" comes back as `P` + `##riya Raman`, two PER groups — two marks across one name,
  and two half-names for redaction to remove. Contiguous same-label spans are merged; ones
  separated by whitespace are **not**, because that would join two genuinely separate
  mentions ("Berlin Munich" as one LOC).

An entity that cannot be placed is **reported on screen**, never dropped quietly: a
silently shorter list of highlights looks exactly like a model that found less.
`offsets.ts` solves the adjacent problem for a caller that owns its tokenizer and has a
complete, contiguous piece list; this one is for a caller reading the *pipeline's* output,
which is aggregated and skips every `O` token, so the pieces arrive with arbitrary gaps.

**Redaction is a pure derivation**, so toggling it — or changing which types it removes —
re-derives from the spans already in hand and costs nothing. Same rule as `/vad`'s
threshold and detection's confidence floor: only GENERATE spends. The redacted text is
offered as a **clipboard copy** rather than a download, which is the right affordance for
a paragraph.

`just fe-e2e-text` asserts the **offsets**, not the count: each mark's own text must be
the entity exactly — no leading space, no truncated final character, no `##` fragment. A
count-based assertion passes happily while every highlight sits two characters to the left
of the word it means.

### 3.3 Question Answering — **shipped**

Taxonomy task **Question Answering** ·
[`/question-answering`](../../frontend/src/routes/question-answering.tsx) ·
[#40](https://github.com/bthek1/model_playground/issues/40).

Extractive QA at **62.8 MiB q8 / 124.5 MiB fp16**
(`Xenova/distilbert-base-cased-distilled-squad`) — the only exported entry, so this is a
one-entry page, and §0's second question is satisfied by the floor being 63 MB rather than
by there being an alternative. `deepset/roberta-base-squad2` and
`deepset/deberta-v3-large-squad2` have no ONNX export, and #5's suggested Qwen3-0.6B
reader is 569.8 MB for the same `{span, score}` a 63 MB encoder produces.

**The pipeline could not carry this page, and that is the finding.** The plan was written
around `question-answering` being in `SUPPORTED_TASKS` and returning
`{ answer, score, start, end }` with character offsets. It is in `SUPPORTED_TASKS`, and it
returns `{ answer, score }` — `start` and `end` are declared *optional* in its types and
**never populated**, past a literal `// TODO add start and end?` in the pipeline's own
source (4.2.0, measured under `onnxruntime-node`). So a caller reading `result.start`
type-checks cleanly and gets `undefined`. `token-classification` has the same unwritten
TODO in the same place.

This is the fourth page planned around a pipeline that could not carry it — after MusicGen,
Florence-2 and `/image-text-to-text` — and the first where the check that would have caught
it is not "is the task in `SUPPORTED_TASKS`". **Check the fields a pipeline actually
populates, not only that the task exists**; an optional field in a `.d.ts` is a claim about
the type, not about the runtime.

So the route owns an engine ([`src/text/qa/`](../../frontend/src/text/qa/)), on the repo's
standing criterion — it is not a plain `pipeline()` call — and drives `AutoTokenizer` +
`AutoModelForQuestionAnswering` directly.

**Render the answer as a span, not as a string.** An extractive model's answer *is* a
range, and showing it as one is what makes a wrong answer look wrong: quoted alone it reads
as authoritative, shown where it came from it is obviously not there. The score sits beside
it always — it is the only signal there is, and hiding it makes a 0.03 answer look like a
0.99 one.

**Owning the span means owning the alignment**, and that is where this page can silently
fail. [`text/offsets.ts`](../../frontend/src/text/offsets.ts) walks the WordPiece pieces
along the passage and returns one character range per token; it is exact, and it returns
**null rather than guessing** when a piece does not match (an `[UNK]`, a lowercasing
tokenizer, an accent-stripping normaliser). A near-miss highlight lands beside the word it
means and reads as a styling bug, so "no highlight, answer quoted" is the honest fallback
and the page has that branch.

**The obvious shortcut — `context.indexOf(answer)` — is not merely fragile, it is wrong on
this page's own sample.** Asked what WebGPU supports that WebGL does not, the model answers
`general-purpose compute shaders`; the tokenizer decodes those same ids as
`general - purpose compute shaders`, which does not occur in the passage at all, so a
substring search returns -1. Slicing [213, 244) returns the passage's characters with the
hyphen intact. The search also picks the *first* occurrence, which on a passage that names
someone twice highlights the wrong one.

**The selection is a transcription of the pipeline's, on purpose.**
[`qa/select.ts`](../../frontend/src/text/qa/select.ts) keeps the same masking, the same two
softmaxes and the same `p(start=i)·p(end=j)` sweep over every `i ≤ j` — no maximum answer
length (HF's Python pipeline caps at 15 tokens; Transformers.js does not cap, and a cap
changes the answer on exactly the unsure questions this page is about), and CLS left in the
softmax denominator before its score is zeroed. Measured against the pipeline on nine
question/passage pairs: **same answer, same score to six decimals, all nine** — so the
offsets are added without the answers moving.

**The model cannot abstain, and saying so is a first-class requirement.** SQuAD 1.1 models
always answer; the squad2 checkpoints that can say "no answer" have no export, so the
behaviour is not merely unshipped but unavailable. The note lives in OUTPUT's *description*
rather than beside the result, so it is on screen before the first answer — a caveat that
appears only once you already believe the answer has arrived too late — and it is keyed off
`QaModel.canAbstain` so an abstaining export would retire it without a rewrite. A route test
and a mocked E2E spec pin the copy, the `/video-classification` precedent.

**The disclaimer ships with its demonstration.** One sample asks "Who won the 1998 World
Cup?" of the Eiffel Tower passage. The model answers "Gustave Eiffel" **at 0.94** — chosen
over an obviously off-topic pair that scores 0.03, because the lesson is not that the model
answers but that it is *confident*: the score is not a usable "do I know this" signal
either.

`just fe-e2e-qa` asserts a **character range**, not a string — "a span appeared" passes
while the alignment is off by a token, and "the span reads Gustave Eiffel" passes while it
marks the second mention of a name.

### 3.4 Zero-Shot Classification — **shipped**

Taxonomy task **Zero Shot Classification** ·
[`/zero-shot-classification`](../../frontend/src/routes/zero-shot-classification.tsx) ·
[#42](https://github.com/bthek1/model_playground/issues/42).

The first page in the app whose **label set is the user's** rather than the checkpoint's.
An NLI model is asked, once per label, whether the text entails a hypothesis built from
that label; the answers are normalised into a score list.

| entry | q8 | fp16 | role |
|---|---|---|---|
| `Xenova/nli-deberta-v3-xsmall` | 83.2 MiB | 136.2 MiB | the default |
| `Xenova/mobilebert-uncased-mnli` | **25.7 MiB** | 47.8 MiB | the floor — the smallest that works at all |
| `Xenova/distilbert-base-uncased-mnli` | 67.6 MiB | 134.1 MiB | the runtime's own default, and **absent from #5's table** |
| `Xenova/bart-large-mnli` | 392.2 MiB | **778.1 MiB** | the classic. Gated on the fp16 number |

**`MoritzLaurer/deberta-v3-base-zeroshot-v2.0` is cut, and it was this page's recommended
default.** It publishes exactly one ONNX file — `onnx/model.onnx` at fp32, 738.6 MB — so
the "~180 MB" the original roadmap quoted was an estimate of a file that does not exist.
The replacement is the fourth row above, which the original table also missed: it is what
`pipeline("zero-shot-classification")` loads when you name no model at all.

**`Xenova/bart-large-mnli` is not "411 MB, under the bar, but gate it".** 411 MB is its q8
size; on WebGPU `loadOpts()` asks for fp16 and the download is 815,853,432 bytes. It still
ships, because **§0's bar is about a page's floor and this page's floor is a 26 MB
model** — a cheap default beside a gated heavy option is the shape §0 allows, and the
failure it forbids is a page where every entry is heavy. But the entry carries measured
bytes for both backends so the picker quotes the number the user will actually pay.

**`isHeavy` moved out of `/depth` rather than being copied.** The threshold and the
predicate now live in `model/size.ts` as `HEAVY_MODEL_BYTES` / `isHeavyDownload`, which is
the same move `backend.ts` and `size.ts` made out of `audio/` when vision arrived. It is
still a **size test and not a model id** — that is what let it survive Depth Pro being
cut, and what let this page inherit it instead of writing a second gate.

**N labels cost N forward passes**, and there is no batching anywhere in the pipeline —
it is a `for` loop over the hypotheses with `await this.model(inputs)` inside it. Ten
labels is ten inferences on one press. The pass count sits next to GENERATE and is derived
from the label list as it is edited, so editing labels still spends nothing and the cost
is on screen before the click.

**The hypothesis template is part of the input, so it is on screen and editable.** This is
the `hypothesis_template` finding from `/zero-shot-image-classification` transplanted: the
pipeline applies `"This example is {}."` unless told otherwise, so a page that templates
silently compares a prompt the user cannot read. The route shows the composed hypothesis
for the first label, sends the template explicitly on every run, and carries it into
OUTPUT beside the result. Labels are stored as **bare nouns** (`billing`, not
`a billing issue`) so the template composes.

**A template with no `{}` is the silent failure this page can have.** Every label would
compose to the *same* hypothesis, so every label would get the same logits and the ranking
would be whatever order the ties resolve in — nothing throws, and the bar chart is
perfectly ordinary. `templateProblem()` refuses it and says why.

**`multi_label` changes the arithmetic, not the model**, and cannot re-derive from scores
already in hand: single-label is one softmax across every label's entailment logit, while
multi-label is a softmax of entailment against contradiction per label. So flipping it
runs nothing and the next GENERATE is a real second inference, which the page says beside
the switch — the same contract as `/video-text-to-text`'s reverse toggle. One label is
**always** scored independently (`softmaxEach = multi_label || labels.length === 1`),
whatever the toggle says, and the page states that rather than rendering a lone 1.00 as
certainty.

**Every entry declares `entailment` in its `label2id`, and that is load-bearing.** The
pipeline looks the index up by name and falls back to `2` with only a console warning if
it is missing — and the right index is 1 for the DeBERTa head, 0 for MobileBERT and
DistilBERT, and 2 for BART. A checkpoint without the mapping would be scored on the wrong
logit and return a full, confident, wrongly-ordered list. `just fe-e2e-models` checks the
mapping on the Hub, because no run would tell you.

`just fe-e2e-zeroshot-text` asserts a **known ranking on a known sentence** — a billing
complaint must put `billing` first against labels the model was never trained on — then
re-runs the same premise under a bare `{}` template and asserts the **scores move**. Two
templates producing identical numbers is exactly what a page that lets the pipeline apply
its own default looks like, and nothing else can catch it. It asserts the scores move
rather than that the ranking flips, which would pin a property the model does not promise.

### 3.5 Translation — **shipped**

Taxonomy task **Translation** · [`/translation`](../../frontend/src/routes/translation.tsx) ·
[#45](https://github.com/bthek1/model_playground/issues/45). The category's first seq2seq
page, and the one that turned a suspicion in `model/backend.ts` into a rule.

| pair | fp16 (WebGPU) | WASM — enc q8 + dec fp32 |
|---|---|---|
| `Xenova/opus-mt-en-de` · `opus-mt-de-en` | 199.6 MiB | **258.5 MiB** |
| `Xenova/opus-mt-en-fr` · `opus-mt-fr-en` | 202.3 MiB | **261.9 MiB** |
| `Xenova/opus-mt-en-es` · `opus-mt-en-zh` | 213.1 MiB | **275.4 MiB** |

**A pair is a model, so changing the pair is a LOAD.** A Marian checkpoint carries its own
language pair and `tr(text)` takes nothing else — there is no language argument anywhere in
the task. So the direction control is a *model selector*, it lives in SELECT beside the
download it costs, and the page says so before the click. A hook that accepted `{ from, to }`
and dropped them would look like it worked, because the model would keep translating in the
direction it was built for; `useTranslate` therefore takes no language at all. The route
test asserts that a SELECT change calls neither `load` nor `run`, and `just fe-e2e-translate`
asserts the reverse pair really reverses on a real load — a control that changed the label
without changing the checkpoint is invisible to every other test in the repo.

**The plan said no `dtypes` pin was needed here. That was wrong, and it is the page's most
useful finding.** A Marian decoder cannot be quantized on the WASM provider bundled with
Transformers.js 4.2.0: the session does not open at all, with

```
Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: qdq_actions.cc:137
TransposeDQWeightsForMatMulNBits Missing required scale:
model.shared.weight_merged_0_scale for node: model.shared.weight_transposed_DequantizeLinear
```

— the same error from the same line as Whisper's and Donut's. Marian is the **third** family
to hit it (BART, in §3.6, is the fourth), which is the point at which `model/backend.ts`'s
own note said to generalise rather than copy the literal again. It is now
`SEQ2SEQ_WASM_DTYPES` there, `asrLoadOpts` is expressed in terms of it, and every seq2seq
catalogue entry references it. Measured in Chromium on 2026-09-25; `encoder_model` quantizes
fine, so only the decoder pays full precision (101 MiB → 258.5 MiB per pair), and the
alternative is no CPU path at all. The fallback configuration was verified end to end: 26 s
to load, 126–163 ms per translation, correct German out.

**That also settles the question the plan left open for Phase 2.** The plan worried that
en↔de (199.6 MiB at fp16) would slip under `LARGE_MODEL_BYTES` while en→es (213.1 MiB)
crossed it, leaving one warning on a page of otherwise identical models — an inconsistency
to explain or document. It does not arise: `sizeEstimate` keys `large` off the **bigger** of
the two downloads, and with the pin the WASM side is 271–289 MB for every pair, so every
pair warns, consistently, about a number the user will actually pay. The inconsistency was
an artefact of a WASM path that does not exist. A test pins it so a future un-pinning cannot
reintroduce it quietly.

**NLLB is further over the bar than the roadmap thought — §1.1's finding, one more time.**
`Xenova/nllb-200-distilled-600M` was quoted at 894.6 MB, which is its **q8** size; it is a
seq2seq, so `loadOpts()` asks WebGPU for fp16 (**1 760 444 340 bytes, 1.68 GiB**) and its
CPU path cannot use a quantized decoder either. There is no configuration in which a browser
pays 895 MB for it. So the page states the comparison **per pair** — one specialist is about
an eighth of one NLLB — rather than summing the catalogue, which comes to 1.6 GB and reads
as an argument against the design rather than for it. `mbart-large-50-many-to-many-mmt` is
1.62 GiB at fp16 and stays cut for the same reason.

### 3.6 Summarization — planned, and **gated on a measurement** ([#46](https://github.com/bthek1/model_playground/issues/46))

Every BART summarizer exported for the browser is over the bar at fp16
(`distilbart-cnn-6-6`: 537.5 MiB). The page exists only if its WebGPU entries pin
`dtype: "q8"` — 283.9 MB, comfortably inside — **and that pin is a measurement rather than
a precaution.** q8-on-WebGPU is exactly the combination `/super-resolution` measured as
*worse than not running the model at all*.

The lead-3 baseline (`text.split(/(?<=[.!?])\s/).slice(0, 3)`) travels beside every
summary, per `/graph-classification`'s rule that a metric owes its null model on screen.

### 3.7 Feature Extraction · Sentence Similarity — **shipped**

Taxonomy tasks **Feature Extraction** → [`/text-features`](../../frontend/src/routes/text-features.tsx)
(named for `/image-features`, not for the slug) and **Sentence Similarity** →
[`/sentence-similarity`](../../frontend/src/routes/sentence-similarity.tsx) ·
[#43](https://github.com/bthek1/model_playground/issues/43).

Two taxonomy rows, one engine, one catalogue, one hook — and two routes, because they are
two questions and two Hub tags (the same argument that keeps `/visual-question-answering`
out of `/image-text-to-text`). The cheapest floor in the app: the default download is
**22 MiB**.

| entry | dim | pooling | q8 (WASM) | fp16 (WebGPU) |
|---|---|---|---|---|
| `Xenova/all-MiniLM-L6-v2` | 384 | mean | **21.9 MiB** | 43.2 MiB |
| `Xenova/bge-base-en-v1.5` | 768 | **cls** | 105.0 MiB | 208.0 MiB |
| `Xenova/all-mpnet-base-v2` | 768 | mean | 105.0 MiB | 208.0 MiB |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | mean | 130.9 MiB | 261.2 MiB |
| `Alibaba-NLP/gte-modernbert-base` | 768 | **cls** | 143.3 MiB | 284.5 MiB |

Re-measured off the Hub and re-checked by `just fe-e2e-models`.
`onnx-community/Qwen3-Embedding-0.6B-ONNX` measures 613.5 MB and stays cut — a 22 MB model
does this page's job, and §0's bar is about the floor.

**The pooling is a property of the checkpoint, and the plan was wrong to pin it.** The plan
said to pin `{ pooling: "mean", normalize: true }` in the engine, and the second half is
right: normalising is not a preference, every consumer here compares by cosine, so it is
pinned and cannot be turned off. The first half is not. A sentence embedding *is* a pooling
of the token rows, and which pooling is part of how the model was trained — mean for
all-MiniLM, all-mpnet and Nomic; **CLS** for BGE and gte-modernbert, read from each
upstream repo's own `1_Pooling/config.json`. Mean-pooling a CLS-trained checkpoint returns
a vector of the right width that ranks plausibly and is wrong, with nothing failing
anywhere. So `pooling` is catalogue data (`EmbedModel.pooling`), it is on screen in SELECT
before a byte downloads, and it travels with `upstream` because **the `Xenova/*` ONNX
mirrors do not publish that file** — there is no way to discover it at load time.
`just fe-e2e-models` reads it back from the upstream repo and fails on a mismatch; the
catalogue ships both poolings one click apart so the hazard is reachable rather than
theoretical.

**The plan's silent failure is real, and the guard is a spread rather than a threshold.**
Without pooling and normalisation the similarities collapse into a narrow band near 0.9 and
every pair looks alike — a page that appears to work with a number that means nothing. "The
paraphrase scores above 0.5" passes comfortably on exactly those collapsed vectors, so
`just fe-e2e-embed` asserts the **gap** between a paraphrase and an unrelated pair
(measured on all-MiniLM: ~0.62 against ~0.02). Omitting `pooling` now also fails *loudly*
one layer down — `toVector` throws on the unpooled `[1, T, D]` hidden state rather than
taking row 0, which is a real vector that would rank plausibly.

**Matryoshka truncation is a measurement here, not a demonstration, because four of the
five entries are not Matryoshka-trained.** MRL is a claim about checkpoints whose *prefix*
dimensions were explicitly optimised to stand alone; all-MiniLM, all-mpnet, BGE and
gte-modernbert never were. Rather than assert the cut is free, the pages report what it
cost: `truncate()` returns `kept`, the fraction of the vector's length the prefix held, and
that is exactly the factor a *missing* renormalisation would scale every similarity by.
`nomic-embed-text-v1.5` is in the catalogue so the genuine MRL case is one click away — and
it is the entry that needs a **task prefix**, which is why `EmbedModel.prefixes` is not
dead code. Prefixes are named by use (`symmetric` / `query` / `document`) rather than by
string, so a page cannot reach for the wrong one; the composed string is on screen before
the click, the `/zero-shot-classification` hypothesis-template rule one modality over.

**Two modules moved rather than being copied.** `vision/serialize.ts` →
[`model/serialize.ts`](../../frontend/src/model/serialize.ts) and `vision/similarity.ts` →
[`model/similarity.ts`](../../frontend/src/model/similarity.ts): a `Tensor` is a
Transformers.js fact and a unit vector has no modality, and this category needed all of
both. Same move `backend.ts` and `size.ts` made out of `audio/`. That matters here because
`feature-extraction` is the **first text task whose result is a `Tensor`** — which does not
merely arrive stripped of its methods, it refuses to be cloned at all
(`#<_Tensor> could not be cloned`), so the text engine now flattens every result exactly as
vision's does.

**Embed once, keyed by the exact string — not by a hash of it.** A hash is the obvious
reach and is strictly worse: a collision serves *another sentence's* embedding with nothing
failing, which on a similarity page is a confident wrong number. The strings are the user's
own input, in a Map, in one tab. The cache is keyed on the **composed** string (prefix
included), because the same sentence as a query and as a document is two different vectors.
A model change clears it, asserted directly — a 384-d MiniLM vector served for a 768-d BGE
query is a wrong answer with no error attached.

### 3.8 Text Generation — planned ([#47](https://github.com/bthek1/model_playground/issues/47))

The category's only streaming page, and the payoff for putting `partial` in the **shared**
`ModelResponse` envelope rather than in a private VLM protocol.

**`Xenova/gpt2`'s "128.3 MB at q4f16" is not a file that exists.** Its `model_q4f16.onnx`
is byte-for-byte the size of its fp16 build (GPT-2's `Conv1D` weights are skipped by the
exporter), it publishes no `model_quantized.onnx` at all so `loadOpts("wasm")`'s q8 404s,
and the 128.3 MB file the roadmap quoted belongs to a legacy graph family 4.2.0 only
requests via `model_file_name`. `HuggingFaceTB/SmolLM2-360M-Instruct` at **272.7 MB**
genuinely quantized is the default instead.

### 3.9 Fill-Mask — **shipped**

Taxonomy task **Fill-Mask** ·
[`/fill-mask`](../../frontend/src/routes/fill-mask.tsx) ·
[#41](https://github.com/bthek1/model_playground/issues/41).

Four entries across three tokenizer families, specifically so the mask-token trap is one
click away rather than theoretical: **always insert `tokenizer.mask_token`, never a
literal string.**

| entry | q8 | fp16 | mask | role |
|---|---|---|---|---|
| `Xenova/bert-base-uncased` | 105.7 MiB | 209.2 MiB | `[MASK]` | the default — the original MLM |
| `Xenova/distilbert-base-uncased` | 64.6 MiB | 127.9 MiB | `[MASK]` | the cheapest, and the least factual |
| `Xenova/roberta-base` | 120.3 MiB | 238.2 MiB | `<mask>` | **the second tokenizer family** |
| `onnx-community/ModernBERT-base-ONNX` | 144.1 MiB | 285.8 MiB | `[MASK]` | 2024 corpus, the sharpest |

This is the one page where a *base* model is the qualification rather than the
disqualification §3.1 made it: masked language modelling is the objective these encoders
were pretrained on, so the head is the real one, not a randomly initialised
`LABEL_0`/`LABEL_1`.

**The plan's premise about the trap was wrong, and the correction is worth keeping.** It
expected a hard-coded `[MASK]` on RoBERTa to return "fluent, wrong predictions". Measured
against the real pipeline, it **throws**: `FillMaskPipeline` looks `mask_token_id` up in
the token ids and raises `Mask token (<mask>) not found in text.` The bug is loud. It is
still a failure the user did nothing to cause, so the page keeps all three defences — the
token is inserted by a button, rewritten in place when the model changes, and reconciled
against the loaded tokenizer in the engine (`text/mask.ts`, `text/engine.ts`). A run
therefore stays correct even if a catalogue entry drifts from its repo, and the page says
on screen when the two disagree.

**The silent failure on this page is a *second* mask.** The pipeline takes `findIndex`
over the ids: it fills the first and drops the rest with no error at all. Measured —
"The `[MASK]` of France is `[MASK]`." comes back as "the border of france is.": one
filling, and a sentence quietly missing a word. So the route refuses to run on anything
but exactly one mask, with the reason on the trigger rather than a disabled button and
nothing else.

**A model change rewrites the mask already in the box**, which is the decision the plan
left open for Phase 2. Rewriting beats refusing because the user's sentence is the part
worth keeping and the token is punctuation they did not type — and the page says it
happened, because the alternative to rewriting silently is not refusing, it is saying
nothing.

**The page's own lesson is that DistilBERT does not know the capital of France.** BERT
says *paris* at 0.33 and ModernBERT at 0.88; DistilBERT says *marseille*, and at fp32 it
is **worse** (marseille, nantes, toulouse — no *paris* in the top three), so this is
distillation rather than quantization damage. One click, two models, and the cost of
halving an encoder is on screen. That also settles the E2E's known answer: it asserts
*paris* on BERT and RoBERTa, never on DistilBERT.

Bias probing is framed as **evidence about the corpus, not about the world**, per this
file's original §3.9 — three paired prompts differing by a single word, run as one
batched GENERATE, rendered side by side under the prompts that produced them, with the
framing sentence travelling *inside* the result rather than near it.

`just fe-e2e-fillmask` is the guard, and **the RoBERTa half is the test**: the same
question through a different tokenizer, which a page that hard-codes the literal cannot
pass. `just fe-e2e-models` additionally reads each repo's own `tokenizer_config.json` and
compares it to the declared `maskToken`, so a drifted entry is a red test rather than a
note on a page.

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
| Token Classification | Yes, **and multilingual too** | `Xenova/bert-base-NER` | 103.9 MiB q8 |
| Table QA | **No** | — | server, or `sql.js` plus a small LLM |
| Question Answering | **Shipped**, extractive only | `Xenova/distilbert-base-cased-distilled-squad` | 62.8 MiB q8 / 124.5 MiB fp16 |
| Zero-Shot Classification | Yes, **shipped** | `Xenova/nli-deberta-v3-xsmall` | 25.7 MiB q8 (mobilebert) |
| Translation | Yes, **one pair at a time** | `Xenova/opus-mt-en-de` per pair | 101.1 MiB q8 |
| Summarization | **Only if q8-on-WebGPU measures well** | `Xenova/distilbart-cnn-6-6` | 270.8 MiB q8 |
| Feature Extraction | Yes, excellent | `Xenova/all-MiniLM-L6-v2` | **21.9 MiB q8** |
| Text Generation | Yes, small models, streamed | `HuggingFaceTB/SmolLM2-360M-Instruct` | 260.1 MiB q4f16 |
| Fill-Mask | **Shipped** | `Xenova/bert-base-uncased` — DistilBERT is cheaper and misses facts | 105.7 MiB q8 / 209.2 MiB fp16 |
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
