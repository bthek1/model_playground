# Multimodal Models in the Browser (WebGPU or CPU)

> The **Multimodal** category of
> [`taskTaxonomy.ts`](../../frontend/src/components/layout/taskTaxonomy.ts), task by task:
> what runs client-side on the user's GPU (WebGPU) or CPU (WebAssembly), and which
> checkpoint to use. No Python server in the inference path.
>
> This file began as issue [#4](https://github.com/bthek1/model_playground/issues/4) and
> moved here when its first route shipped, the same way
> [`audio.md`](audio.md), [`vision.md`](vision.md) and [`graph.md`](graph.md) did: a
> roadmap that documents *shipped* code has to be reviewable in the same pull request as
> the code it describes, which an issue body cannot be.

| Roadmap section | Route | Status |
|---|---|---|
| Image-Text-to-Text (§3.1) | [`/image-text-to-text`](../../frontend/src/routes/image-text-to-text.tsx) | **Shipped** — SmolVLM 256M / 500M, streaming |
| Visual Question Answering (§3.2) | [`/visual-question-answering`](../../frontend/src/routes/visual-question-answering.tsx) | **Shipped** — the same engine, no new download |
| Document Question Answering (§3.3) | — | **Cut** — shipped, then removed for size. 411 MB / 597 MB, one checkpoint |
| Video-Text-to-Text (§3.4) | [`/video-text-to-text`](../../frontend/src/routes/video-text-to-text.tsx) | **Shipped** — SmolVLM2 256M Video, 189 MB, a frame sampler over §3.1 |
| Audio-Text-to-Text (§3.5) | — | **Cut** — [#34](https://github.com/bthek1/model_playground/issues/34) closed: blocked, and ~800 MB of two live models |
| Visual Document Retrieval (§3.6) | — | **Cut** — [#35](https://github.com/bthek1/model_playground/issues/35) closed: 953 MB, fp32-only, no `config.json` |
| the four that stay on a server (§3.7) | — | **Done** — documented, with a reason each |

**Three of nine are built, and that is the whole of the category.** §3.1, §3.2
and §3.4 all ship, over **one** engine — one worker, one `engine.ts`, one
`useVlm`, one `VlmRun` envelope — and two downloads between them, because §3.2
adds none at all. The other six are server-side or cut, each with its measured
reason below. The build-out is recorded in
[#30](https://github.com/bthek1/model_playground/issues/30),
[#31](https://github.com/bthek1/model_playground/issues/31) and
[#33](https://github.com/bthek1/model_playground/issues/33).

**Three routes over one engine is the category's design and also its hazard.**
They must not drift into three catalogues, three hooks and three sets of copy: a
change this category needs belongs in the shared hook, and every route gets it.
That is what §3.4 did when it needed N images — `VlmRun.image` became
`VlmRun.images`, additively, and §3.1's own route tests are what proved the
shipped page unchanged.

### What was cut, and why

The browser budget is the whole story of this category, so the cuts belong at
the top of it rather than buried in their sections.

| | download | why it went |
|---|---|---|
| `/document-question-answering` (§3.3) | **411 MB** WebGPU · **597 MB** WASM | shipped and then removed. One checkpoint, and no second one is possible: the pipeline hardcodes Donut's prompt, so another architecture would be prompted with tokens it has never seen |
| `/audio-text-to-text` (§3.5) | ~**800 MB** combined | blocked on NLP ([#5](https://github.com/bthek1/model_playground/issues/5)) *and* two models live at once — the largest standing exception to the one-model rule, for a page whose own point is that the architecture fails |
| `/visual-document-retrieval` (§3.6) | **953.5 MB**, fp32 only | its own plan opened with a spike that was allowed to end it. `colSmol-256M-ONNX` publishes no `config.json`, so no `Auto*` class can load it, and the three-stage comparison needs CLIP and Florence-2 live beside it |

Two of those three were never built, so cutting them costs nothing but the
plan. The third was built, shipped, and measured — and the measurement is what
condemned it. That is §0 of
[`adding-a-task-page.md`](../guides/adding-a-task-page.md) working as intended,
one step later than it should have.

Multimodal is where the browser budget bites hardest. A vision-language model is an image
encoder bolted to a language decoder, so it pays both costs: hundreds of image tokens per
picture, then autoregressive generation over them. The reference models run 2B to 7B on a
12 GB card. A tab has perhaps a tenth of that.

The good news is that the sub-billion VLM is a real category. SmolVLM at 256M and 500M has
maintained ONNX exports, accepts images and video frames, and runs on WebGPU.

**WebGPU is not optional here.** Every model in this file is a decoder, and a decoder on
WASM is measured in seconds per token. Both shipped catalogue entries declare
`backends: ["webgpu"]`, so `useBackendProbe` disables the row on a machine without an
adapter and says why — the honest move rather than a CPU path nobody will wait for.

**And "has a GPU" is not the gate — `shader-f16` is.** An adapter without that feature
loads `q4f16` weights perfectly happily, reports `ready`, and then fails on the **first
operator** of every run:

```
Non-zero status code returned while running Gather node. Name:'/Gather'
Status Message: shader_helper.cc:401 GenerateSourceCode
  Program Gather requires f16 but the device does not support it.
```

That is the worst of the three possible outcomes, because the user pays for the download
first. `supportsShaderF16()` in [`model/backend.ts`](../../frontend/src/model/backend.ts)
is the real probe, and `useBackendProbe({ requireShaderF16: true })` folds it into the
answer so the picker disables the row **before** anything is fetched. The page adds a
sentence naming the missing feature, because "this machine resolved to wasm" is misleading
on a machine that plainly has a GPU.

Found by `just fe-e2e-vlm` on Chromium's SwiftShader fallback — which is exactly what a CI
runner with no `/dev/dri` gets, and the reason that spec needs real hardware.

Every model id below was checked against the Hugging Face API. Re-check with
`just fe-e2e-models`, which also verifies the q4f16 graphs and the quoted sizes.

---

## 1. The core stack

`@huggingface/transformers` is already a dependency at 4.2.0. Nothing needs installing.

### There is no `image-text-to-text` pipeline

The original roadmap said 4.2.0 "carries the `image-text-to-text` pipeline". It does not.
`SUPPORTED_TASKS` has 25 entries — `image-to-text` and `document-question-answering` among
them — and `image-text-to-text` is not one. A VLM page therefore drives
`AutoModelForImageTextToText` + `AutoProcessor` directly.

**This is the third time this repo has planned a page around a pipeline that could not
carry it**, and the rule is now well earned:

| Page | Planned as | Actually needed | Why |
|---|---|---|---|
| `/text-to-audio` | `pipeline("text-to-audio")` | `MusicgenForConditionalGeneration` | the pipeline throws "Missing the following inputs: input_ids" |
| `/image-to-text` | `pipeline("image-to-text")` | `Florence2ForConditionalGeneration` | resolves via `AutoModelForVision2Seq`, whose registry has no `florence2` |
| `/image-text-to-text` | `pipeline("image-text-to-text")` | `AutoModelForImageTextToText` | the task does not exist |

**Check `SUPPORTED_TASKS` before planning around a pipeline**, not after.

The runtime *does* support the models: 4.2.0 registers `idefics3`, `smolvlm`, `qwen2_vl`
and `qwen3_vl`, exports `TextStreamer`, and `DATA_TYPES` includes `q4f16`.

### `q4f16`, via `vlmLoadOpts()`

`loadOpts()` returns fp16/q8 and is wrong for every model in this file. The VLM override
lives in [`model/backend.ts`](../../frontend/src/model/backend.ts) beside `asrLoadOpts`,
which is the precedent for a per-family precision decision living next to the default
rather than as a literal in a worker:

```ts
vlmLoadOpts("webgpu")  // { device: "webgpu", dtype: "q4f16" }
vlmLoadOpts("wasm")    // { device: "wasm",   dtype: "q4"    }
```

The `f16` half is dropped on WASM because fp16 activations are a GPU format. That path
exists to be *typed*, not recommended.

Measured, by summing every graph each repo publishes:

| | q4f16 | fp16 | saving |
|---|---|---|---|
| SmolVLM-256M | **189 MB** | 514 MB | 2.7x |
| SmolVLM-500M | **358 MB** | 1017 MB | 2.8x |
| Qwen3-VL-2B | 1373 MB | 3380 MB | 2.5x |

### A q4f16 size estimate is wrong, and not by a rounding error

`estimateBytes` assumes one precision across the whole model. At `q4f16` that is false in a
way that gets **worse the smaller the model is**: SmolVLM-256M's
`embed_tokens_q4f16.onnx` is **56.8 MB, the same size as its fp16 build** — the embedding
table is not 4-bit quantized at all, and it is 30% of the download.

So every VLM entry carries **measured `bytes`**, and `just fe-e2e-models` re-checks them
against the Hub. `BYTES_PER_PARAM` still gains `q4`/`q4f16` entries (0.55, not 0.5 — a q4
export leaves embeddings, norms and biases at higher precision): a widened `Dtype` without
them renders "NaN MB" on a real page with nothing failing on the way there.

### Two other size claims that did not survive

- **Qwen3-VL-2B keeps its weights in external `.onnx_data` files.** Summing only the
  `.onnx` stubs measures a 1373 MB model at 1.2 **MB**.
- **The roadmap's Qwen2-VL-2B alternative is 2668 MB at q4f16, not ~1.1 GB.** Qwen3-VL is
  the better heavy model by a wide margin, not a toss-up.

---

## 2. The shape of a VLM page

Two things distinguish these pages from every other task page in the repo.

### The input is a conversation, not a value

`apply_chat_template` is not decoration. Each checkpoint has its own image placeholder
token and its own turn markers, and a hand-built prompt string produces output that is
**subtly degraded rather than obviously broken** — fluent, confident, and not an answer to
the question. Same class of bug as Florence-2's `construct_prompts`, which this repo has
already paid for once.

```ts
const text = processor.apply_chat_template(
  [{ role: "user", content: [{ type: "image" }, { type: "text", text: prompt }] }],
  { add_generation_prompt: true },
);
const inputs = await processor(text, [image]);
```

Three ways to get this wrong silently: `{ type: "image" }` is a **slot** filled
positionally from the image list, so the content array and the list must line up;
`add_generation_prompt` is what makes the model *answer* rather than continue the
question; and `generate` returns **prompt + answer**, so decoding the whole sequence hands
the user back their own question with the answer glued on.

### The output streams, and the first token is slow

Image encoding runs to completion before a single token exists. That pause is seconds, and
an unlabelled pause is indistinguishable from a hang.

```
LOAD:  idle -> loading -> ready           (Machine A, untouched)
RUN:   encoding -> generating -> done     (inside one run)
```

The encode/generate split is **not a fifth status**. It travels on the shared envelope's
`partial` variant, added in
[`model/types.ts`](../../frontend/src/model/types.ts) for this route:

```ts
export type ModelResponse<TResult, TPartial = never> =
  | { type: "progress"; progress: ModelProgress }
  | { type: "ready"; model: string; backend: LoadOpts["device"] }
  | { type: "partial"; id: number; partial: TPartial }   // ← in-run progress
  | { type: "result"; id: number; result: TResult }
  | { type: "error"; id?: number; error: string };
```

Machine A stays `ready` throughout and `running` stays an inflight count — a partial
neither opens nor closes a request. `TPartial` defaults to `never`, which is what made the
change free: the arm is uninhabited for every worker that does not stream, so no existing
engine, switch or test was touched.

`useModelWorker` surfaces it as `partial`, cleared when a run starts and when its result or
error lands, and **dropped if its request has already settled** so a late chunk cannot
repaint OUTPUT after the finished answer is up. It went in the shared envelope rather than
a private protocol because NLP text-generation
([#5](https://github.com/bthek1/model_playground/issues/5)) needs exactly the same thing.

### Image tokens are the budget, and 512 is not a round number

`SmolVLM`'s `preprocessor_config.json` declares `size.longest_edge: 2048`,
`max_image_size.longest_edge: 512` and **`do_image_splitting: true`**: the processor
resizes up to 2048 and then cuts the picture into **512px tiles**, each encoded separately
at its own token cost, plus a global view. A 2048px input is up to a 4x4 grid — seventeen
encodes for one question.

So `MAX_INFERENCE_SIDE = 512` is the model's own tile size, and handing it an image already
at 512 produces **one** tile. That is the difference between ~64 image tokens and over a
thousand before the user's question is even appended.

This caps the **source**; `AutoProcessor` still does the real resize and normalisation from
the model's own config. Never resize *for* the model.

---

## 3. Task by task

### 3.1 Image-Text-to-Text — **shipped**

[`/image-text-to-text`](../../frontend/src/routes/image-text-to-text.tsx) ·
[`src/multimodal/`](../../frontend/src/multimodal/)

| Browser model | q4f16 | Notes |
|---|---|---|
| `HuggingFaceTB/SmolVLM-256M-Instruct` | **189 MB** | the default; loads in seconds and is visibly the weaker |
| `HuggingFaceTB/SmolVLM-500M-Instruct` | **358 MB** | noticeably better answers; the comparison the page is for |
| `onnx-community/Qwen3-VL-2B-Instruct-ONNX` | 1373 MB | **not shipped** — see below |
| `Xenova/nanoLLaVA`, `onnx-community/Qwen2-VL-2B-Instruct` | ~1 GB / 2668 MB | older, and the second is far heavier than advertised |
| `OpenGVLab/InternVL3-2B-hf`, `google/gemma-3-4b-it` | — | no export |

Both shipped entries are `idefics3` (`Idefics3ForConditionalGeneration`), both publish a
`chat_template`, and both are gated to WebGPU by the picker.

**Why Qwen3-VL-2B is absent.** The original roadmap wanted it as the heavy end of a
head-to-head, and at 1373 MB it is past the ~1 GB ceiling
[`adding-a-task-page.md`](../guides/adding-a-task-page.md) §0 sets and past the budget
`model/size.test.ts` asserts over every shipped catalogue. Two of this repo's own documents
disagreed; the ceiling won, and the 2B gets its own plan and its own decision rather than
arriving as a side effect of this one. SmolVLM-500M is the second rung instead — same
family, same chat template, 358 MB.

**What the page settles.** The question is held INPUT: typing it, tapping a preset, or
choosing a picture all run nothing, and only GENERATE spends. The answer streams, the
encode is named, and the result is labelled with the question it was **actually** asked —
captured inside the run, so editing the box afterwards cannot relabel a result on screen.
The page also says, next to the output, that a small VLM answers confidently whether or not
it can see what you asked about.

**A dead `data-testid`, found on the way.** `components/Markdown.tsx` accepts only
`{ children, className }` and silently drops everything else, so a `data-testid` passed to
it never reaches the DOM. `/image-to-text` passed one and it never resolved — its test
queried `slot-4` instead. This route puts the testid on a wrapper.

### 3.2 Visual Question Answering — **shipped** ([`/visual-question-answering`](../../frontend/src/routes/visual-question-answering.tsx))

VQA is now prompting a general VLM, and the browser confirms it: neither classical model
has a usable export (`dandelin/vilt-b32-finetuned-vqa`, `Salesforce/blip-vqa-base` — no
ONNX), so the 3129-answer classification approach is simply unavailable in a tab. This is
§3.1's engine behind a **distinct route** — a user looking for VQA does not think to click
"image-text-to-text", and the taxonomy has both rows because the Hub has both tags. It is
the cheapest route in the repo: **no new download, no new worker, no new catalogue**.

**The whole page is [`multimodal/prompt.ts`](../../frontend/src/multimodal/prompt.ts), 80
lines of pure function.** `composePrompt(question, { terse })` returns the exact string
that will be sent and the cap that travels with it — off, the question as typed at 128
tokens; on, the question plus `Answer in one word.` at 16. The same weights and the same
picture produce visibly different output, which is the most legible lesson about prompting
a VLM that fits on one screen.

Three decisions inside those 80 lines:

- **The cap is 16, not 1 or 2.** The instruction is the mechanism and the cap is a
  backstop. A model told to answer in one word and then truncated mid-word has been cut
  off, not instructed — and the demonstration would be indistinguishable from the
  scissors. The `@slow` assertion is "materially shorter in words", which is only
  meaningful because the model had room to be verbose.
- **A question that already asks for brevity is not instructed again.** Doubly ordering a
  small decoder makes it answer the *instruction*, and a user who typed the instruction
  themselves would have no way to see why.
- **The composed prompt is on screen before the click, and labels the answer after it.** A
  page that rewrites the prompt behind the user's back is the `hypothesis_template`
  problem `/zero-shot-image-classification` was caught by: the user compares two prompts
  while the model is shown two others.

**The toggle looks like a filter, which is why it is this page's §1.6 hazard.** Flipping
it runs nothing — it changes what the next GENERATE sends, and the line beside it says so.
A route test, a mocked E2E spec and the `@slow` spec each assert it separately.

### 3.3 Document Question Answering — **built, then cut for size**

[#32](https://github.com/bthek1/model_playground/issues/32) shipped this route and
it was removed in the same breath as `/text-to-audio` and `/image-to-text`. The
reason is the table at the top of this file: **411 MB on WebGPU, 597 MB on
WASM**, one checkpoint, and no possibility of a lighter second entry — the
pipeline hardcodes Donut's prompt (below), so there is nothing else to offer.

The section is kept in full rather than deleted, because almost everything it
records is about the *runtime* rather than about this page, and the next
encoder-decoder will need all of it. The ORT quantized-decoder finding in
particular now lives in [`model/backend.ts`](../../frontend/src/model/backend.ts)
beside `asrLoadOpts`, which is the function it generalises.

**It was the one route in this category that rode a real pipeline.** 4.2.0 carries
`document-question-answering`; its registry maps exactly one entry,
`vision-encoder-decoder → VisionEncoderDecoderModel`, which is Donut's model type; and
`Xenova/donut-base-finetuned-docvqa` is the pipeline's *own default model*. Checked before
planning rather than after a failed load — the step §3.1 above exists to teach.

| | |
|---|---|
| download | **410.7 MB fp16** (WebGPU) · **596.7 MB** (WASM), measured |
| graphs | `encoder_model` + `decoder_model_merged` |
| backends | **both**, ungated |

**The repo totals ~4 GB at fp32, and that number is a trap**: it publishes three
*alternative* decoders (`decoder_model`, `decoder_with_past_model`,
`decoder_model_merged`) and only the merged one is ever loaded. Sum the graphs the entry
declares, not the repo.

#### The WASM decoder must stay fp32, and the ORT bug is not ASR-specific

A uniform q8 cannot open a session in the browser at all:

```
Can't create a session. ERROR_CODE: 1, ERROR_MESSAGE: qdq_actions.cc:137
TransposeDQWeightsForMatMulNBits Missing required scale:
decoder.model.decoder.embed_tokens.weight_merged_0_scale
```

That is **the same ONNX Runtime bug `asrLoadOpts` was written for**, and finding it here is
the news. Its note says the fault "reproduces on every ASR repo tried" — but Donut is not
an ASR model, so it is not ASR-specific. Read it as: **any encoder-decoder whose decoder is
quantized** fails to open a session on the WASM provider bundled with
`@huggingface/transformers` 4.2.0. The encoder quantizes fine; only the decoder pays full
precision, expressed per entry as
`dtypes: { wasm: { encoder_model: "q8", decoder_model_merged: "fp32" } }`.

Cost: **596.7 MB on WASM instead of 218.7 MB** — the same ~3x the ASR catalogue pays, and
justified the same way: the alternative is not a cheaper CPU path but *no CPU path at all*.

**Nothing but a real browser load catches it.** The unit suite mocks the runtime away, the
mocked E2E run never loads weights, `just fe-e2e-models` confirms the files exist (they
do), and the identical call loads cleanly under `onnxruntime-node` — the fault is specific
to the WASM provider in the browser. `just fe-e2e-docvqa` is what found it; that
recipe is gone with the route, and this finding is the reason the section is kept.

**WebGPU keeps fp16, unpinned.** `q4f16` would cut it to 241.0 MB, but that would be a
*precaution, not a measurement*, and document QA is where quantization error lands directly
on small print. The machine that found the WASM bug had no adapter, so pinning the GPU path
either way would still be a guess.

**Both backends are offered, deliberately.** Unlike every model in §3.1 this is not an
autoregressive chat decoder but an encoder plus a short extractive decode, so the CPU path
is real. Declaring `backends: ["webgpu"]` without measuring would be the guess this
catalogue avoids elsewhere.

#### The resolution control was wrong, and the opposite is right

The plan for this page proposed exposing resolution as a slider, on the reasoning that a
document needs more pixels than a photo. The reasoning was right; the mechanism was not.

`preprocessor_config.json` is `do_resize` + `do_thumbnail` + `do_pad` at a fixed
`{ height: 2560, width: 1920 }`, and `thumbnail()` in 4.2.0 **never upscales** — it shrinks
to fit preserving aspect, then pads to exactly 2560x1920. So **the encoder always sees a
2560x1920 tensor**, and two things follow:

- **Inference cost is constant.** A slider would save nothing at the model.
- **A smaller source is padded, not enlarged.** Fewer real pixels of print at identical
  compute — legibility given away for no return.

So there is no slider. The correct expression is the repo's standing rule stated plainly:
**never resize for the model** — `AutoProcessor` reads the model's own config and that file
*is* the input contract. This is the one vision-family route that does not cap its source,
and the page says so, because an unexplained inconsistency with a dozen sibling routes
reads as an oversight. The `MAX_SOURCE_SIDE = 2560` that remains is a **memory bound at the
processor's own dimension** — lossless with respect to what the model sees, and there only
to keep a 12-megapixel phone photo out of a structured clone.

That is the same lesson `/link-prediction` records: reuse that looks like a decision is
often an inheritance, so run it.

#### `answer` can be null, and nothing errors

The pipeline extracts with `decoded.match(/<s_answer>(.*?)<\/s_answer>/)` and returns
`[{ answer: null }]` when that misses. So "the model found nothing" is an ordinary, silent
outcome, and it arrives on precisely the documents the model found hardest. The engine
passes `null` through rather than flattening it to `""` — "found nothing" and "found an
empty span" are different things to say — and OUTPUT renders it explicitly instead of
showing a blank panel.

#### Two things the page has to say

**The privacy argument**, because this is the one page where a user can feel why the whole
architecture was chosen: people photograph payslips, medical letters and bank statements,
and this answers questions about them without the image leaving the device.

**That the answer is extracted, not reasoned.** Donut copies a span off the page; it cannot
add up a column or compare two figures. Framing that is a correctness requirement, not
decoration — the `/video-classification` precedent — and an E2E spec asserts the copy.

The alternatives remain unbuildable: `impira/layoutlm-document-qa` needs OCR boxes as
*input*, which the browser would have to produce first, and `stepfun-ai/GOT-OCR-2.0-hf` has
no export. (Florence-2 covered plain OCR on `/image-to-text` until that route was
cut for size; with both gone, **OCR has no page** — asking `/image-text-to-text`
to read a sign is the nearest thing, and it is a VLM answering a question rather
than a transcription.)

### 3.4 Video-Text-to-Text — **shipped** ([`/video-text-to-text`](../../frontend/src/routes/video-text-to-text.tsx))

`HuggingFaceTB/SmolVLM2-256M-Video-Instruct`, **189.2 MB at q4f16** measured, three graphs,
`model_type: smolvlm` — and in 4.2.0 `SmolVLMForConditionalGeneration extends
Idefics3ForConditionalGeneration` with the processor re-exported unchanged, so it is a
catalogue entry on the existing engine rather than a new module. It is within 0.2% of
SmolVLM-256M's size. The 500M video variant (357.6 MB) is structurally identical and is the
second rung if it earns one.

**A video model in a tab is a frame sampler plus an image model, and the page says so
next to the result.** That is a correctness requirement, not decoration — the same one
`/video-classification` carries, and an E2E spec asserts the copy for the same reason.
It is also simply true: there is no temporal attention in SmolVLM2 to hide.

| | |
|---|---|
| download | **189.2 MB** q4f16 (WebGPU) · 264.2 MB q4 (WASM, not offered) |
| graphs | `embed_tokens` + `vision_encoder` + `decoder_model_merged` |
| backends | **WebGPU only**, and `shader-f16` gated — a decoder on WASM is seconds per token, and this page hands it eight frames' worth of image tokens |
| frames | 1–8, default 4 |

**What it added to the engine, and how.** `VlmRun.image` became `VlmRun.images`, an
**ordered list**, and the worker's chat template grew from one `{ type: "image" }` slot to
N. Strictly additive: a single-image run is a one-element list, `useVlm.run` takes
`RawImage | RawImage[]`, and `/image-text-to-text`'s own route tests are what proved the
shipped page unchanged. The slots are filled **positionally** from the array, so the count
is derived from the list's own length at the call site rather than passed beside it — the
only way that goes wrong is if the two are allowed to be two facts.

**Frames multiply the tile problem; they do not add to it.** §3.1 learned that the
processor splits anything over 512px into tiles, each costing its own image tokens plus a
global view. With N frames that multiplies by N — a 640x360 frame becomes five tiles, so
eight of them is forty encodes for one question. `MAX_FRAME_SIDE = 512` is the fix, and it
is the model's own `video_sampling.video_size.longest_edge` rather than a number inherited
from the sibling page.

**The frame count is the cost dial, capped at 8.** Each frame at 512px is one tile and
`processor_config.json` sets `image_seq_len: 64`, so N frames is 64N image tokens *before*
the question is appended, attended over for every generated word. The model's own config
allows 64 frames; that is a number for a server. Moving the slider runs nothing.

**Sampling is uniform over the clip, and shown.** [`multimodal/frames.ts`](../../frontend/src/multimodal/frames.ts)
is the pure half — `uniformFrameTimes(duration, count)` samples the **centre of each of N
equal slices**, because 0 is usually a black frame or a fade and `duration` is past the
last decodable frame on a good many encodes, and both failures look like the model
ignoring the video. The filmstrip in OUTPUT is the frames the model was actually given: a
page that samples invisibly makes every wrong answer unattributable.

Note what this is *not*: `vision/video.ts`'s `frameTimes` samples at a fixed **rate**,
which is right for `/video-classification` where each frame is an independent pass and a
longer clip should produce more points. Here the frames share one prompt, so the cost is
the **count** and eight means eight whether the clip is six seconds or six minutes.
The decode itself is still `sampleVideo` — it grew a `times` option rather than a second
copy, and that option is a *function of the duration*, because only the decoder has read
it by then.

**The reverse toggle is the experiment, and it is the one control on these pages that
legitimately spends.** Feed the frames backwards and see whether the answer changes. Often
it does not, and that is a real property of a small video VLM — it is describing a picture,
not reading a sequence — rather than a gotcha. Unlike a threshold it cannot re-derive: the
model has to actually see the other order. So flipping it runs nothing, the next GENERATE
is a genuine second inference, and the page says that before the click. The `@slow` spec
asserts the **re-run**, not a difference in the answer: asserting a difference would pin a
property the model does not have.

**Decoding is cached on (clip, frame count).** Frame extraction is a seek per frame, so
pressing GENERATE twice must cost one decode and two inferences —
[`useVideoPick`](../../frontend/src/hooks/useVideoPick.ts) owns that, along with the
one-object-URL rule `useImagePick` and `useAudioPick` already follow. It deliberately does
**not** cache by frame *order*: reversing is a property of the run, and a hook that cached
it would decode the same clip twice to produce the same pictures.

### 3.5 Audio-Text-to-Text — **cut** ([#34](https://github.com/bthek1/model_playground/issues/34) closed)

Two independent reasons, either of which would have been enough. It was
**blocked** on a text-generation worker that does not exist yet (NLP,
[#5](https://github.com/bthek1/model_playground/issues/5)), and the cascade is
**~800 MB of two models live at once** — Whisper-base plus whichever decoder #5
eventually picks, which at `Qwen3-0.6B` is 569.8 MB on its own. That is the
largest standing exception to the one-model rule, and it would have been paid
for a page whose own thesis is that the architecture loses information.

If NLP ships something materially smaller than 0.6B, this is worth reopening —
the wiring is two existing workers and the plan below is still correct. What
follows is that plan's substance, kept for that case.

The cascade — `onnx-community/whisper-base` into a small text-generation model — needs a
text-generation worker, which arrives with the NLP category
([#5](https://github.com/bthek1/model_playground/issues/5)). Nothing new downloads once it
does. `ibm-granite/granite-speech-3.3-2b` has no export and `Qwen/Qwen2-Audio-7B-Instruct`
is not a browser model in any quantization.

The point of the page is **what the cascade cannot hear**: play a clip with laughter, a
raised voice or two speakers, and the transcript loses everything except the words. Show
the transcript and the answer side by side. Two models are live at once here — a deliberate
exception to the one-model rule, affordable only because both are small, and one to note in
the code.

### 3.6 Visual Document Retrieval — **cut** ([#35](https://github.com/bthek1/model_playground/issues/35) closed)

The plan opened with a spike that was explicitly allowed to end it, and the
evidence that spike was meant to gather was already in the plan: **953.5 MB,
fp32 only, and no `config.json`**. Three stages would have had to be live at
once (colSmol, CLIP, Florence-2). This is the cleanest possible instance of
[`adding-a-task-page.md`](../guides/adding-a-task-page.md) §0 — a documented
"server-side, and here is the reason" is a finished piece of work, and it is
worth more than three days of rediscovery.

What would have made it worth building is recorded below, because the argument
is a good one and the day a quantized ColPali export appears it becomes live
again. The comparison is the page:
`onnx-community/colSmol-256M-ONNX` for late interaction, `Xenova/clip-vit-base-patch32` as
the single-vector control, and Florence-2 `<OCR>` + BM25 as the lexical baseline — all
three in the browser over dropped-in page images. MaxSim over patch vectors is real
algorithm work, not a dot product, and a wrong reduction axis still ranks plausibly.

**But "an export exists, which makes this buildable" does not survive the repo contents.**
`colSmol-256M-ONNX` ships one `onnx/model.onnx` at **fp32 only, 953.5 MB**, a tokenizer, a
preprocessor config and a chat template — and **no `config.json`**. Every `Auto*` class
resolves the architecture from that file, so Transformers.js cannot load this repo at all.
Three consequences: it is a **bare-ONNX route** (§9, like `audio/vad/` and
`audio/enhance/`), `vlmLoadOpts()` does not apply because there is nothing but fp32, and
driving it bare means reading its input contract off the graph itself. #35 therefore opens
with a **spike that is allowed to end the plan** — a documented "server-side, and here is
why" is a finished piece of work.

### 3.7 The four that stay on a server

| Taxonomy task | Why not in a tab |
|---|---|
| Image Text to Image | InstructPix2Pix, MagicBrush, ControlNet — all SD 1.5 derivatives. Multi-gigabyte weights, tens of denoising steps |
| Image Text to Video | AnimateDiff + SparseCtrl, LTX-Video. The same problem with a frame count on top |
| Any to Any | `onnx-community/Janus-Pro-1B-ONNX` exists (~1 GB at q4f16) but only its **understanding half** ports; the generation half is a diffusion decoder |
| the native half of Audio Text to Text | Qwen2-Audio-7B is not a browser model in any quantization |

**Any to Any is the closest call.** A Janus-understanding page is legitimate *if it is
labelled for what it is*. Do not ship a page called "Any to Any" that only does one
direction — rename the route, or leave the placeholder. Same rule `/super-resolution` and
`/image-to-3d` follow.

Rule of thumb: **if the output is text, there is probably a page; if the output is pixels
or audio, there is not.**

---

## 4. Feasibility summary

| Taxonomy task | In-browser? | Recommended model | Backend | If not |
|---|---|---|---|---|
| **Image Text to Text** | **Shipped** | SmolVLM-256M / 500M | WebGPU only | Qwen3-VL-2B past the size ceiling |
| **Visual Question Answering** | Yes, via a VLM | the §3.1 engine | WebGPU only | ViLT and BLIP-VQA have no export |
| **Document Question Answering** | **Cut** — built, measured at 411/597 MB, one checkpoint | `Xenova/donut-base-finetuned-docvqa` | — | LayoutLM, GOT-OCR to server |
| **Video Text to Text** | Yes, sampled frames | `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | WebGPU only | long-video understanding to server |
| **Audio Text to Text** | **Cut** — blocked on #5, and ~800 MB of two live models | Whisper-base + a small LLM | — | native audio LLMs to server |
| **Visual Document Retrieval** | **Cut** — bare ONNX, fp32 only, 953 MB, no `config.json` | `onnx-community/colSmol-256M-ONNX` | — | full ColQwen2 to server |
| **Any-to-Any** | Understanding half only | `onnx-community/Janus-Pro-1B-ONNX` | WebGPU | generation half to server |
| **Image Text to Image** | No | — | — | server API |
| **Image Text to Video** | No | — | — | server API |

---

## 5. Memory and performance notes

The audio and vision rules apply, plus five of their own.

- **`q4f16` on WebGPU, always** — via `vlmLoadOpts()`, never a literal in a worker.
- **Quote measured bytes**, because a q4f16 estimate is wrong by up to 30%.
- **Downscale to the model's tile size** (512 for SmolVLM). Documents were the one
  deliberate exception, and that route is gone — but the rule it proved stands: a
  page sets its own number from the processor's own config rather than inheriting a
  sibling's.
- **Cap the frame count** at 4–8 for video. A 32-frame clip is not a slower page, it is a
  broken one.
- **Give the encode its own state**, and stream every generation.
- **One VLM live at a time, no exception.** These are the largest downloads in the app; a
  leaked session ends the tab. Null the reference *first*, then dispose.
- **The large-model warning is not optional here.** Every model in this file is past
  `LARGE_MODEL_BYTES` several times over: state the cost, download nothing until the
  click, and assert zero Hub requests before it in an E2E spec. `/text-to-audio` was
  the gating precedent and has itself since been cut — which is the sharper lesson.
  **A gate is not a substitute for a size that fits.** Three routes in this repo were
  correctly gated, honest about the cost, and removed anyway, because a page whose
  every checkpoint is several hundred megabytes is a page nobody clicks twice.

---

## 6. Testing

Unit tests mock the network and the runtime away, so they **cannot catch a broken chat
template** — and that failure has no error attached to it. It is a fluent, confident
sentence that does not answer the question, which is exactly what "a result appeared" would
accept.

- `just fe-test` — the engine, the catalogue, the hook, and the route's §8 checklist.
- `just fe-e2e` — the mocked shell, and the gate: **zero Hub requests before the click**.
- `just fe-e2e-models` — every id resolves, every entry publishes its three q4f16 graphs,
  and the quoted sizes still match the Hub.
- `just fe-e2e-vlm` — a real SmolVLM-256M load and a real generation, asserting a **known
  answer on a known image**, plus §3.2's terse toggle as a **property**: a materially
  shorter answer, in words, to the same question on the same picture. Needs a real GPU with
  `shader-f16`: on SwiftShader the model loads in 29 s and then every run fails on the first
  Gather, which is how that gate was found in the first place.
- `just fe-e2e-videovlm` — the same for §3.4, and the only guard on the **multi-image**
  template. It asserts a known answer about a known clip, that the frame count reaches the
  model, and that reversing the frames is a genuine second inference. It deliberately does
  **not** assert that the answer changes: at this size it usually does not, and that is the
  page's finding rather than its failure.

---

## 7. Reference

- **Transformers.js**: `AutoModelForImageTextToText`, `AutoProcessor.apply_chat_template`,
  `TextStreamer`. (The `document-question-answering` pipeline is real and works —
  §3.3 — but no route uses it any more.)
- **Image helpers**: [`vision.md`](vision.md) §2 — `useImagePick`, `ImageSourcePanel`,
  `downscale`, `toPayload`/`fromPayload`. **Audio helpers**: [`audio.md`](audio.md) §2.
- **Page construction**: [`docs/guides/adding-a-task-page.md`](../guides/adding-a-task-page.md).
- **The page contract**: [`docs/standards/model-page-pattern.md`](../standards/model-page-pattern.md).
- **Recipes**: [`docs/guides/adding-a-model.md`](../guides/adding-a-model.md) §8
  (Transformers.js), §9 (a bare ONNX graph), §10 (a chat-templated VLM).
- **What shipped**: plan issue
  [#30](https://github.com/bthek1/model_playground/issues/30).
