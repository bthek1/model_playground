# NLP Models in the Browser (WebGPU or CPU)

> The **Natural Language Processing** category of
> `components/layout/taskTaxonomy.ts`, task by task: what runs client-side on the
> user's GPU (WebGPU) or CPU (WebAssembly), and which checkpoint to use. No Python
> server in the inference path.

**Nothing in this category has a real route yet.** Text Generation maps to
`/playground`, which is a WebGPU demo surface rather than an LLM page; the other
eleven tasks render the `/tasks/$slug` placeholder. This file is the research a
plan gets written from. The procedure is
[`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md); the shared plumbing is
shipped and described in
[`Audio_Models_in_React_WebGPU_and_CPU.md`](Audio_Models_in_React_WebGPU_and_CPU.md)
§1–2.

NLP is the category that ports best. Eleven of the twelve tasks become working
pages, most of them with models under 100 MB, and several of them are *better*
in the browser than in a notebook because the interesting output is
interactive: a highlighted entity span, a live-updating similarity score, a
token stream. **Text Classification (§3.1) is the one to build first** — it is
the smallest useful page in the app, and it establishes `src/text/` for the ten
that follow.

There is one thing to be honest about up front. **The browser is an encoder
paradise and a decoder compromise.** BERT-family encoders are 20 MB to 400 MB
and run a single forward pass, so classification, NER, extractive QA, embeddings
and reranking are effectively free. Generative decoders are autoregressive: a
0.6B model at `q4f16` on WebGPU produces maybe 20 to 40 tokens per second on a
laptop, which is usable but is not a hosted API. The research behind these tables
leans on Qwen3 throughout; the browser cannot follow all the way.

Every model id below was checked against the Hugging Face API. Re-check before
shipping with `just fe-e2e-models`.

---

## 1. The core stack

`@huggingface/transformers` is already a dependency. Nothing needs installing,
and nothing else is needed: there is no text equivalent of the audio guide's
decode helpers, because the input is already a string. That makes NLP the
cheapest category to start.

Backend selection and dtype are **shared, not re-derived** — import `pickBackend`
and `loadOpts` from `@/audio/backend` (see the Computer Vision guide §1 on
renaming that module to `src/model/backend.ts` when the first non-audio page
lands). One worker, `src/text/pipeline.worker.ts`, serves every task below except
generation; the task string travels in the `load` message exactly as
`audio/pipeline.worker.ts` does.

One dtype note specific to text, and it is the most consequential setting on
these pages, and it is the one place `loadOpts()` is not enough — a decoder page
must pass its own dtype rather than take the shared default:

| Model kind | WebGPU | WASM | Why |
|---|---|---|---|
| Encoder (BERT, RoBERTa, DeBERTa, ModernBERT) | `fp16` | `q8` | small enough that quantization buys little |
| Seq2seq (BART, Marian, NLLB, T5) | `fp16` | `q8` | the decoder runs per token, so precision costs time |
| Decoder LLM (Qwen3, SmolLM2) | `q4f16` | `q4` | 4-bit weights with fp16 compute. The only setting that makes a 0.6B model pleasant |

`q4f16` is not a compromise on an LLM, it is the default. At `fp16` a 0.6B model
is a 1.2 GB download that most users will abandon.

---

## 2. The two output shapes, and the components they imply

Almost every page in this category renders one of two things, so build both
components once.

**A score list.** Classification, zero-shot, NER confidence, reranking. A
horizontal bar per label, sorted, with the score printed. The point about the
aggregate number hiding the failure applies here directly: show the
top few with scores, never a single label, so a near-tie is visible.

**A span overlay.** Token classification, extractive QA, fill-mask. The input
text re-rendered with highlighted ranges. This is the one piece of real work in
the category, and it has a trap:

```ts
// Character offsets, not token indices. Transformers.js returns `start` and
// `end` in characters when the tokenizer is fast, which every model here uses.
export function highlight(text: string, spans: { start: number; end: number; label: string }[]) {
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const out: { text: string; label?: string }[] = [];
  let at = 0;
  for (const s of sorted) {
    if (s.start > at) out.push({ text: text.slice(at, s.start) });
    out.push({ text: text.slice(s.start, s.end), label: s.label });
    at = s.end;
  }
  if (at < text.length) out.push({ text: text.slice(at) });
  return out;
}
```

**Never reconstruct the text from tokens.** This is the token-versus-word trap,
and it is worse in a UI than in a notebook: subword tokens rebuilt by
concatenation lose the original whitespace, and the highlight lands one or two
characters off. Slice the original string by character offset instead.

Both components belong in `src/components/text/`, built once and shared, in the
same way `components/model/` is shared by every audio route.

Everything below assumes the worker skeleton from
[`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md) section 5. Text
models are fast enough that the main thread survives one call, but not fast
enough that it survives a user typing.

---

## 3. Task by task

### 3.1 Text Classification, the smallest useful page in the repo

Taxonomy task **Text Classification** · not built. Upstream research: DistilBERT SST-2, twitter-roberta, ModernBERT, Qwen3.

| Upstream (PyTorch) | Browser model | Size (q8) | Notes |
|---|---|---|---|
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | `Xenova/distilbert-base-uncased-finetuned-sst-2-english` | ~67 MB | the default, binary sentiment |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | `Xenova/twitter-roberta-base-sentiment-latest` | ~125 MB | the domain-matched model, 3 classes |
| `answerdotai/ModernBERT-base` | `onnx-community/ModernBERT-base-ONNX` | ~150 MB | the base model, so fine-tune before it classifies anything |
| `Qwen/Qwen3-0.6B` as a classifier | `onnx-community/Qwen3-0.6B-ONNX` | ~400 MB q4f16 | see 3.8. Works, and is a hundred times the cost |

```ts
const clf = await pipeline("text-classification",
  "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
  { ...loadOpts(backend), topk: null });         // topk: null returns all labels
const out = await clf("the film was a triumph of tedium");
```

Build the head-to-head into the page: run the general and
the domain-matched model on the same sentence and put the two score lists side
by side. The gap on a tweet is the whole lesson.

### 3.2 Token Classification, the page that most wants to be interactive

Taxonomy task **Token Classification** · not built. Upstream research: bert-base-NER, roberta-large NER, wikineural.

| Upstream (PyTorch) | Browser model | Notes |
|---|---|---|
| `dslim/bert-base-NER` | `Xenova/bert-base-NER` | ~110 MB, 4 entity types, the default |
| `Jean-Baptiste/roberta-large-ner-english` | none | 355M params, no export |
| `Babelscape/wikineural-multilingual-ner` | none | no export |

```ts
const ner = await pipeline("token-classification", "Xenova/bert-base-NER",
                           { ...loadOpts(backend), aggregation_strategy: "simple" });
const ents = await ner(text);   // [{ entity_group, word, start, end, score }, ...]
```

`aggregation_strategy: "simple"` is what merges `B-PER` and `I-PER` back into
one span. Without it the page renders one highlight per subword token and looks
broken.

There is a genuinely good page hiding here: **tag and redact**. Highlight
the entities, then offer a button that replaces every `PER` and `LOC` span with
a placeholder. Redaction that never leaves the browser is a real argument for
client-side inference rather than a demo of one.

### 3.3 Question Answering, extractive is easy, generative is a choice

Taxonomy task **Question Answering** · not built. Upstream research: DistilBERT SQuAD, roberta-squad2, deberta-v3-large, Qwen3.

| Upstream (PyTorch) | Browser model | Notes |
|---|---|---|
| `distilbert/distilbert-base-cased-distilled-squad` | `Xenova/distilbert-base-cased-distilled-squad` | ~65 MB, the extractive baseline |
| `deepset/roberta-base-squad2` | none | no export. The abstention behaviour is lost |
| `deepset/deberta-v3-large-squad2` | none | 435M params, no export |
| `Qwen/Qwen3-0.6B` as a reader | `onnx-community/Qwen3-0.6B-ONNX` | generative, see 3.8 |

```ts
const qa = await pipeline("question-answering", "Xenova/distilbert-base-cased-distilled-squad",
                          loadOpts(backend));
const { answer, score, start, end } = await qa(question, context);
```

Highlight `start` to `end` in the passage rather than printing the answer alone.
An extractive model's answer is a span, and showing it as a span makes the
failure mode visible when the model picks a confidently wrong sentence.

**Abstention** cannot be demonstrated with the only exported model, since SQuAD
1.1 models never abstain — the squad2 checkpoints that can say "no answer" have
no export. Say so on the page instead of quietly implying the model knows when it
does not know.

### 3.4 Zero-Shot Classification, the "your own labels" page for text

Taxonomy task **Zero Shot Classification** · not built. Upstream research: bart-large-mnli, deberta-v3 zeroshot, mDeBERTa, Qwen3.

| Upstream (PyTorch) | Browser model | Size | Notes |
|---|---|---|---|
| `facebook/bart-large-mnli` | `Xenova/bart-large-mnli` | ~400 MB q8 | the classic, and heavy for a tab |
| `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | ~180 MB | official ONNX in the same repo. The better default |
| - | `Xenova/nli-deberta-v3-xsmall` | ~70 MB | the fast option when the page must load quickly |
| - | `Xenova/mobilebert-uncased-mnli` | ~25 MB | the smallest that works at all |
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | none | - | cross-lingual is lost |

```ts
const zs = await pipeline("zero-shot-classification",
  "MoritzLaurer/deberta-v3-base-zeroshot-v2.0", loadOpts(backend));
const out = await zs(text, ["billing", "outage", "feature request"], { multi_label: false });
```

Cost scales with the number of labels: the model runs once per label as an NLI
premise-hypothesis pair. Ten labels is ten forward passes. Say that in the UI,
because a user who pastes thirty labels and waits will otherwise assume the page
has hung.

### 3.5 Translation, the seq2seq case

Taxonomy task **Translation** · not built. Upstream research: Marian / OPUS-MT, NLLB-200, Qwen3.

| Upstream (PyTorch) | Browser model | Size (q8) | Notes |
|---|---|---|---|
| `Helsinki-NLP/opus-mt-en-de` | `Xenova/opus-mt-en-de` | ~75 MB | one pair per model, and tiny. Swap the id for other pairs |
| `facebook/nllb-200-distilled-600M` | `Xenova/nllb-200-distilled-600M` | ~650 MB q8 | 200 languages in one model, and the download shows it |
| - | `Xenova/mbart-large-50-many-to-many-mmt` | ~700 MB q8 | the other multilingual option |
| `Qwen/Qwen3-1.7B` | none at that size | - | too heavy for a tab |

```ts
const tr = await pipeline("translation", "Xenova/nllb-200-distilled-600M", loadOpts(backend));
await tr(text, { src_lang: "eng_Latn", tgt_lang: "deu_Latn" });
```

The bilingual-versus-multilingual trade-off becomes a concrete UI
decision here, and the browser sharpens it: a Marian pair is a 75 MB download
and a NLLB is 650 MB. If the page knows the language pair, ship the specialist.

### 3.6 Summarization, works, and the baseline matters more than the model

Taxonomy task **Summarization** · not built. Upstream research: lead-3, distilBART, BART-large-CNN, Qwen3.

| Upstream (PyTorch) | Browser model | Size (q8) | Notes |
|---|---|---|---|
| `sshleifer/distilbart-cnn-12-6` | `Xenova/distilbart-cnn-6-6` | ~230 MB | the smaller distil variant, exported |
| `facebook/bart-large-cnn` | `Xenova/bart-large-cnn` | ~400 MB | the full model |
| - | `Xenova/LaMini-Flan-T5-783M` | ~300 MB q8 | instruction-tuned T5, summarises by prompt |

**Ship the lead-3 baseline as a toggle.** It is the baseline that must be
beaten, and it is the most useful thing on the page: three
sentences of `text.split(/(?<=[.!?])\s/).slice(0, 3)` cost nothing, and seeing
the neural summary fail to beat them on a news article is the lesson.

The faithfulness section has no browser path: the NLI entailment checker the
check would need is another 180 MB model. It is possible to load both, since
`MoritzLaurer/deberta-v3-base-zeroshot-v2.0` is already in the catalogue, but
that is two models live and belongs behind an explicit toggle.

### 3.7 Feature Extraction and Sentence Similarity, one engine, two pages

Taxonomy tasks **Feature Extraction** and **Sentence Similarity** · neither built.

These share a worker. Both are embedding models, and the similarity page is the
feature page with a cosine on the end.

| Upstream (PyTorch) | Browser model | Dim | Size (q8) |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | `Xenova/all-MiniLM-L6-v2` | 384 | ~23 MB |
| `sentence-transformers/all-mpnet-base-v2` | `Xenova/all-mpnet-base-v2` | 768 | ~110 MB |
| `BAAI/bge-base-en-v1.5` | `Xenova/bge-base-en-v1.5` | 768 | ~110 MB |
| `Alibaba-NLP/gte-modernbert-base` | `Alibaba-NLP/gte-modernbert-base` | 768 | ~150 MB, official ONNX in-repo |
| `Qwen/Qwen3-Embedding-0.6B` | `onnx-community/Qwen3-Embedding-0.6B-ONNX` | 1024 | ~600 MB. Strong, and heavy |
| `cross-encoder/stsb-roberta-base` | none | - | use the reranker in 3.10 instead |

```ts
const embed = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", loadOpts(backend));
const a = await embed(textA, { pooling: "mean", normalize: true });
const b = await embed(textB, { pooling: "mean", normalize: true });
const cosine = a.data.reduce((s, x, i) => s + x * b.data[i], 0);   // both normalized
```

`pooling: "mean", normalize: true` is not optional. **Raw BERT is not an
embedding model**, and forgetting these two arguments is exactly how you get raw
BERT: the similarities collapse
into a narrow band near 0.9 and every pair looks alike.

Matryoshka truncation ports cleanly and makes a good control:
truncate the 768-dim vector to 256 or 128, renormalise, and watch the retrieval
quality barely move while the index gets three times smaller.

### 3.8 Text Generation, the honest one

Taxonomy task **Text Generation** · currently maps to `/playground`, which is a WebGPU demo surface, not this page. Upstream research: Qwen3, decoding strategies, KV cache, chat templates.

| Upstream (PyTorch) | Browser model | Size (q4f16) | Notes |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | `onnx-community/Qwen3-0.6B-ONNX` | ~400 MB | the reference model, in the only dtype that fits |
| - | `HuggingFaceTB/SmolLM2-360M-Instruct` | ~250 MB | faster, weaker, loads in seconds |
| - | `onnx-community/Qwen2.5-0.5B-Instruct` | ~350 MB | the alternative |
| `openai-community/gpt2` | `Xenova/gpt2` | ~125 MB | for the decoding-strategy demo specifically, see below |
| `Qwen/Qwen3-1.7B` | none practical | - | around 1.2 GB even at q4. Server |

```ts
const gen = await pipeline("text-generation", "onnx-community/Qwen3-0.6B-ONNX",
                           { device: "webgpu", dtype: "q4f16" });
const streamer = new TextStreamer(gen.tokenizer, {
  skip_prompt: true,
  callback_function: (t) => self.postMessage({ type: "token", t }),
});
await gen(messages, { max_new_tokens: 256, temperature: 0.7, top_p: 0.9, streamer });
```

**Stream, always.** The KV cache is what makes this viable; in a tab the
consequence is that the user sees the first token in about a second and the
rest arrives steadily. Waiting for a complete 256-token response feels like a
hang even when the throughput is identical.

The best page here is not a chatbot, it is **the decoding strategies made
interactive**. Greedy against sampling, temperature and `top_p` as live
sliders, and the repetition loop visible on the same prompt. GPT-2 at 125 MB is
the right model for that page precisely because it loops so readily: the lesson
is the decoding strategy, not the model quality, and a small model teaches it in
one screen.

### 3.9 Fill-Mask, small, exact, and quietly the most instructive

Taxonomy task **Fill Mask** · not built. Upstream research: BERT, DistilBERT, RoBERTa, DeBERTa, ModernBERT.

| Upstream (PyTorch) | Browser model | Mask token |
|---|---|---|
| `google-bert/bert-base-uncased` | `Xenova/bert-base-uncased` | `[MASK]` |
| `distilbert/distilbert-base-uncased` | `Xenova/distilbert-base-uncased` | `[MASK]` |
| `FacebookAI/roberta-base` | `Xenova/roberta-base` | `<mask>` |
| `answerdotai/ModernBERT-base` | `onnx-community/ModernBERT-base-ONNX` | `[MASK]` |

```ts
const fm = await pipeline("fill-mask", "Xenova/bert-base-uncased", loadOpts(backend));
await fm(`the capital of France is ${fm.tokenizer.mask_token}.`);
```

**Always insert `tokenizer.mask_token`, never a literal string.** Different
tokenizers use different mask tokens, and hard-coding `[MASK]` breaks silently the moment the user switches to
RoBERTa. The model does not error, it just treats the literal text as words.

Bias probing ports directly and is worth building
carefully. It reads the training corpus back out of the model, which is a
serious result presented as a one-line demo, so the page should frame it as
evidence about the corpus rather than as a party trick.

### 3.10 Text Ranking, retrieval that fits entirely in a tab

Taxonomy task **Text Ranking** · not built. Upstream research: BM25, dense retrieval, hybrid RRF, cross-encoder reranking.

The most complete page available in the category, because **all four stages run
client-side**, over a corpus the user pastes in.

| Stage | Browser implementation |
|---|---|
| BM25 | pure TypeScript, roughly 40 lines. No model at all |
| Dense retrieval | `Xenova/bge-base-en-v1.5` from 3.7, embed the corpus once |
| Hybrid RRF | arithmetic over the two rank lists, no model |
| Reranking | `Xenova/ms-marco-MiniLM-L-6-v2`, ~23 MB, one pass per query-document pair |

```ts
const rerank = await pipeline("text-classification", "Xenova/ms-marco-MiniLM-L-6-v2",
                              { ...loadOpts(backend), topk: 1 });
// A cross-encoder scores a pair. Feed it { text, text_pair }, not a single string.
const scored = await Promise.all(candidates.map(async (d) => ({
  doc: d, score: (await rerank({ text: query, text_pair: d }))[0].score,
})));
```

The cross-encoder is the piece that makes the page worth building: it runs once
per candidate rather than once per query, which is exactly why it reranks the top
20 rather than the whole corpus. That cost shows up
visibly in a browser, so the architecture teaches itself.

`mixedbread-ai/mxbai-rerank-xsmall-v1` is the alternative reranker, also
exported, if you want a second row in the comparison.

### 3.11 Table Question Answering, the one that does not port

Taxonomy task **Table Question Answering** · not built, and it should stay that way. Upstream research: TAPAS, TAPEX, text-to-SQL with Qwen2.5-Coder.

No browser path for any of the three sections:

- **TAPAS** has no ONNX export, and its table-aware position embeddings mean a
  generic encoder is not a substitute.
- **TAPEX** likewise.
- **Text-to-SQL** needs Qwen2.5-Coder-1.5B, which is around 1 GB at `q4` and
  slow, and the SQL execution step needs a database.

There is a real page here that is worth building anyway, just not that one: **run the SQL half in the browser with `sql.js` and let a small LLM write
the query.** SQLite compiled to WebAssembly is a mature, well-supported library,
so the execution and the table rendering are solved. Only the generation is
weak, and a page that shows the generated SQL before running it turns that
weakness into the interesting part.

---

## 4. Feasibility summary

| Taxonomy task | In-browser? | Recommended model | Best backend | If not |
|---|---|---|---|---|
| **Text Classification** | Yes, excellent | `Xenova/distilbert-base-uncased-finetuned-sst-2-english` | WASM or WebGPU | - |
| **Token Classification** | Yes | `Xenova/bert-base-NER` | WASM or WebGPU | large and multilingual NER to server |
| **Table QA** | No | - | - | server, or sql.js plus a small LLM |
| **Question Answering** | Yes, extractive | `Xenova/distilbert-base-cased-distilled-squad` | WASM or WebGPU | squad2 abstention to server |
| **Zero-Shot Classification** | Yes | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | WebGPU | cross-lingual to server |
| **Translation** | Yes | `Xenova/opus-mt-en-de` per pair, or NLLB-200 | WebGPU | LLM translation to server |
| **Summarization** | Yes | `Xenova/distilbart-cnn-6-6` | WebGPU | faithfulness check is a second model |
| **Feature Extraction** | Yes, excellent | `Xenova/all-MiniLM-L6-v2` | WASM or WebGPU | - |
| **Text Generation** | Yes, small models | `onnx-community/Qwen3-0.6B-ONNX` at `q4f16` | WebGPU | anything past 1B to server |
| **Fill-Mask** | Yes, excellent | `Xenova/bert-base-uncased` | WASM | - |
| **Sentence Similarity** | Yes, excellent | `Xenova/bge-base-en-v1.5` | WASM or WebGPU | cross-encoder STS to server |
| **Text Ranking** | Yes, all four stages | `Xenova/bge-base-en-v1.5` plus `Xenova/ms-marco-MiniLM-L-6-v2` | WebGPU | - |

Rule of thumb: **encoders are free, decoders are a budget.** If the task ends in
a single forward pass over the input, build the page. If it generates tokens one
at a time, cap the model at about 0.6B and stream.

---

## 5. Memory and performance notes

- **`q4f16` for every decoder LLM.** Not a fallback, the default. `fp16` on a
  0.6B model is a 1.2 GB download.
- **Encoders can share the tab.** The one-model-live rule is about hundreds of
  megabytes; two 25 MB encoders such as MiniLM and the MS-MARCO reranker
  coexist fine, which is what makes the ranking page possible.
- **Embed the corpus once.** The ranking and similarity pages should cache
  document embeddings in memory keyed by text hash. Re-embedding on every
  keystroke is the difference between instant and unusable.
- **Debounce live input.** A page that classifies as the user types must wait
  for a pause of 200 to 300 ms, and must cancel the in-flight request rather
  than queueing. `useModelWorker` derives `running` from an in-flight count for
  exactly this reason.
- **Tokenizers are downloaded too.** A 25 MB model still fetches a tokenizer and
  a config, so the progress bar should report the aggregate from
  `model/progress.ts` rather than the weights alone.
- **Warm up on load.** One throwaway inference on a short string. On WASM this
  is JIT time the first real query should not pay.
- **Weights cache after the first download**, so the second visit is instant and
  works offline.

---

## 6. Reference

- **Transformers.js pipelines used here**: `text-classification`,
  `token-classification`, `question-answering`, `zero-shot-classification`,
  `translation`, `summarization`, `feature-extraction`, `text-generation`,
  `fill-mask`.
- **TextStreamer** for token-by-token output on the generation page.
- **sql.js** for the browser-side half of the table QA page.
- **Page construction**: [`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md).
- **In-repo standards**:
  [`../../standards/model-page-pattern.md`](../../standards/model-page-pattern.md)
  (the four-slot contract),
  [`../../guides/adding-a-model.md`](../../guides/adding-a-model.md) §8.
- **The shipped precedent**: the five audio routes, mapped in
  [`Audio_Models_in_React_WebGPU_and_CPU.md`](Audio_Models_in_React_WebGPU_and_CPU.md).
  A text page is the same worker protocol with a `string` instead of a
  `Float32Array`, and no decode step at all.
- Model recommendations here come from a companion collection of Python
  notebooks, which is a separate project and not a dependency of this repo. Every
  browser id above was checked against the Hugging Face API — re-check with
  `just fe-e2e-models` once a catalogue module exists to check.
