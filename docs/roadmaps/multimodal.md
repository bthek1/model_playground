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
| Visual Question Answering (§3.2) | — | Planned — [#31](https://github.com/bthek1/model_playground/issues/31): same engine, its own route |
| Document Question Answering (§3.3) | [`/document-question-answering`](../../frontend/src/routes/document-question-answering.tsx) | **Shipped** — Donut, extractive, either backend |
| Video-Text-to-Text (§3.4) | — | Planned — [#33](https://github.com/bthek1/model_playground/issues/33): a frame sampler over §3.1 |
| Audio-Text-to-Text (§3.5) | — | [#34](https://github.com/bthek1/model_playground/issues/34) — **blocked** on NLP ([#5](https://github.com/bthek1/model_playground/issues/5)) |
| Visual Document Retrieval (§3.6) | — | [#35](https://github.com/bthek1/model_playground/issues/35) — **starts with a spike that may end it** |
| the four that stay on a server (§3.7) | — | **Done** — documented, with a reason each |

**Two of nine are built**, and every remaining section has a plan issue. The build-out so far
is recorded in [#30](https://github.com/bthek1/model_playground/issues/30); the category
closes when the last route ships, the way
[#1](https://github.com/bthek1/model_playground/issues/1),
[#2](https://github.com/bthek1/model_playground/issues/2) and
[#3](https://github.com/bthek1/model_playground/issues/3) did.

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
it never reaches the DOM. `/image-to-text` passes one and it has never resolved — its test
queries `slot-4` instead. This route puts the testid on a wrapper.

### 3.2 Visual Question Answering — planned

VQA is now prompting a general VLM, and the browser confirms it: neither classical model
has a usable export (`dandelin/vilt-b32-finetuned-vqa`, `Salesforce/blip-vqa-base` — no
ONNX). So this is §3.1's engine behind a **distinct route** — a user looking for VQA does
not think to click "image-text-to-text" — with a question box, a short `max_new_tokens`,
and the "answer in one word" toggle that makes what a prompt is worth visible.

### 3.3 Document Question Answering — **shipped**

[`/document-question-answering`](../../frontend/src/routes/document-question-answering.tsx) ·
[`src/multimodal/docvqa/`](../../frontend/src/multimodal/docvqa/) · [#32](https://github.com/bthek1/model_playground/issues/32)

**The one route in this category that rides a real pipeline.** 4.2.0 carries
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
to the WASM provider in the browser. `just fe-e2e-docvqa` is what found it.

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
no export (Florence-2 on `/image-to-text` already covers plain OCR).

### 3.4 Video-Text-to-Text — planned

`HuggingFaceTB/SmolVLM2-256M-Video-Instruct` (~189 MB at q4f16, also `idefics3`) is the
practical choice. A video model in a tab is a **frame sampler plus an image model**. Sample
4 to 8 frames, be explicit about the sampling in the UI, and put the frame count on a
slider — each frame carries its own image tokens, so 4 against 16 is a four-times latency
difference the user feels. The temporal-blindness toggle (feed the frames reversed and see
whether the answer changes) is the honest companion.

### 3.5 Audio-Text-to-Text — **blocked**

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

### 3.6 Visual Document Retrieval — planned, and less certain than the roadmap implied

[#35](https://github.com/bthek1/model_playground/issues/35). The comparison is the page:
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
| **Document Question Answering** | **Shipped** | `Xenova/donut-base-finetuned-docvqa` | WebGPU **or** WASM | LayoutLM, GOT-OCR to server |
| **Video Text to Text** | Yes, sampled frames | `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | WebGPU only | long-video understanding to server |
| **Audio Text to Text** | Yes, as a cascade | Whisper-base + a small LLM | WebGPU | native audio LLMs to server |
| **Visual Document Retrieval** | **Unproven** — bare ONNX, fp32 only, 953 MB, no `config.json` | `onnx-community/colSmol-256M-ONNX` | WebGPU | full ColQwen2 to server |
| **Any-to-Any** | Understanding half only | `onnx-community/Janus-Pro-1B-ONNX` | WebGPU | generation half to server |
| **Image Text to Image** | No | — | — | server API |
| **Image Text to Video** | No | — | — | server API |

---

## 5. Memory and performance notes

The audio and vision rules apply, plus five of their own.

- **`q4f16` on WebGPU, always** — via `vlmLoadOpts()`, never a literal in a worker.
- **Quote measured bytes**, because a q4f16 estimate is wrong by up to 30%.
- **Downscale to the model's tile size** (512 for SmolVLM), with documents the deliberate
  exception.
- **Cap the frame count** at 4–8 for video. A 32-frame clip is not a slower page, it is a
  broken one.
- **Give the encode its own state**, and stream every generation.
- **One VLM live at a time, no exception.** These are the largest downloads in the app; a
  leaked session ends the tab. Null the reference *first*, then dispose.
- **The large-model warning is not optional here.** Every model in this file is past
  `LARGE_MODEL_BYTES` several times over. `/text-to-audio` is the gating precedent: state
  the cost, download nothing until the click, and assert zero Hub requests before it in an
  E2E spec.

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
  answer on a known image**. Needs a real GPU with `shader-f16`: on SwiftShader the model
  loads in 29 s and then every run fails on the first Gather, which is how that gate was
  found in the first place.
- `just fe-e2e-docvqa` — a real Donut load and a real extraction, pinned by a **known
  answer on a known document**: `invoice.png` is the Transformers.js docs' own DocVQA
  example and its invoice number is `us-001`. Needs **no GPU** — the WASM path is real
  here — so unlike the VLM spec it runs anywhere. It is deliberately **one test**: a fresh
  context per test re-downloads 597 MB, which turned an 8-minute file into a 25-minute one.
  It also waits for the **trigger** to come back rather than for the answer text to change,
  because the previous answer stays on screen during a re-run and polling it cannot tell
  "still running" from "same answer".

---

## 7. Reference

- **Transformers.js**: `AutoModelForImageTextToText`, `AutoProcessor.apply_chat_template`,
  `TextStreamer`, and the `document-question-answering` pipeline (§3.3).
- **Image helpers**: [`vision.md`](vision.md) §2 — `useImagePick`, `ImageSourcePanel`,
  `downscale`, `toPayload`/`fromPayload`. **Audio helpers**: [`audio.md`](audio.md) §2.
- **Page construction**: [`docs/guides/adding-a-task-page.md`](../guides/adding-a-task-page.md).
- **The page contract**: [`docs/standards/model-page-pattern.md`](../standards/model-page-pattern.md).
- **Recipes**: [`docs/guides/adding-a-model.md`](../guides/adding-a-model.md) §8
  (Transformers.js), §9 (a bare ONNX graph), §10 (a chat-templated VLM).
- **What shipped**: plan issue
  [#30](https://github.com/bthek1/model_playground/issues/30).
