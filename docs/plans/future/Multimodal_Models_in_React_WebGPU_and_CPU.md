# Multimodal Models in the Browser (WebGPU or CPU)

> The **Multimodal** category of `components/layout/taskTaxonomy.ts`, task by
> task: what runs client-side on the user's GPU (WebGPU) or CPU (WebAssembly),
> and which checkpoint to use. No Python server in the inference path.

**Nothing in this category is built, and it should be built last.** All nine
Multimodal tasks render the `/tasks/$slug` placeholder. These pages need both the
image helpers from the Computer Vision guide and the audio helpers that are
already shipped, they are the heaviest downloads in the app, and they are the
only ones where WebGPU is a hard requirement — so they are cheapest to build
*after* one vision page and one text-generation page exist. The procedure is
[`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md).

Multimodal is where the browser budget bites hardest. A vision-language model is
an image encoder bolted to a language decoder, so it pays both costs: hundreds of
image tokens per picture, then autoregressive generation over them. The reference
models run 2B to 7B on a 12 GB card. A tab has perhaps a tenth of that.

The good news is that the sub-billion VLM is now a real category. SmolVLM at
256M and Qwen3-VL at 2B both have maintained ONNX exports, they accept images
and video frames, and they run on WebGPU. Five of the nine tasks become pages.
The other four stay on a server, and the reason is always the same: they generate
images, video or speech rather than text.

**WebGPU is not optional here.** Every model in this file is a decoder, and a
decoder on WASM is measured in seconds per token. If the adapter probe fails,
the honest move is to say so rather than to fall back to a CPU path nobody will
wait for.

Every model id below was checked against the Hugging Face API. Re-check before
shipping with `just fe-e2e-models`.

---

## 1. The core stack

`@huggingface/transformers` is already a dependency at 4.2.0, which carries the
`image-text-to-text` pipeline. Nothing needs installing.

Backend selection as in
[`Audio_Models_in_React_WebGPU_and_CPU.md`](Audio_Models_in_React_WebGPU_and_CPU.md)
§1 — but **`loadOpts()` is wrong for every model in this file**, and that matters
more here than anywhere else:

```ts
// VLMs are decoders. q4f16 is the default, not the fallback.
const opts = { device: "webgpu" as const, dtype: "q4f16" as const };
```

At `fp16` a 2B VLM is roughly 4 GB of download. At `q4f16` it is closer to
1.2 GB, which is the difference between a page people use and a page people
close. SmolVLM at 256M is around 250 MB at `q4f16` and is the one to default to.

`loadOpts()` returns `fp16`/`q8`, so a VLM page passes its own dtype rather than
taking the shared default. Add a `q4f16` branch to `audio/backend.ts` (by then
`model/backend.ts`) when the first decoder page lands, rather than scattering
literals through the workers — `asrLoadOpts` is the precedent for a per-family
override living next to the default.

Image helpers (`RawImage`, camera capture, canvas drawing) are specified in
[`Computer_Vision_Models_in_React_WebGPU_and_CPU.md`](Computer_Vision_Models_in_React_WebGPU_and_CPU.md)
§2 and do not exist yet. Audio helpers do —
[`audio/io.ts`](../../../frontend/src/audio/io.ts) is shipped. Multimodal pages
need both, which is the first sign that these are the most expensive pages to
build.

---

## 2. The shape of a VLM page

Two things distinguish these pages from every other task page in the repo.

**The input is a conversation, not a value.** Transformers.js takes the same
chat-message structure PyTorch does, so build the InputPanel around it rather
than around a text field:

```ts
const messages = [{
  role: "user",
  content: [
    { type: "image" },                              // the slot the image fills
    { type: "text", text: "What is written on the sign?" },
  ],
}];
const prompt = processor.apply_chat_template(messages, { add_generation_prompt: true });
const inputs = await processor(prompt, [image]);
const ids = await model.generate({ ...inputs, max_new_tokens: 128, streamer });
```

`apply_chat_template` is not decoration. Each model has its own image
placeholder token and its own turn markers, and a hand-built prompt string
produces output that is subtly degraded rather than obviously broken. Let the
processor build it.

**The output streams, and the first token is slow.** Image encoding happens
before generation starts, so there is a visible pause with no tokens in it. Give
that pause its own state. A page that shows a spinner for four seconds and then
streams text reads as broken; a page that says "encoding image" and then
"generating" reads as working.

```
LOAD:     idle -> loading -> ready          (Machine A, unchanged)
RUN:      idle -> encoding -> generating -> done
```

Machine A is untouched — this is extra detail *inside* a single run, reported by
the worker as run progress, not a fifth stage and not a new `status` value. Model
Playground's `running` is an inflight count; the encode/generate split is
something the OUTPUT slot renders, not something the load machine knows about.

The image-token cost is the recurring theme and it is a real budget: a single
1024px image can be 1000 or more tokens before the user's question is even
appended. Downscale aggressively. 512px on the long edge is usually enough
for VQA and captioning, and it roughly quarters the encode time.

---

## 3. Task by task

### 3.1 Image-Text-to-Text, the flagship page

Taxonomy task **Image Text to Text** · not built. Upstream research: SmolVLM2, Qwen3-VL, InternVL3, Gemma 3.

| Upstream (PyTorch) | Browser model | Size (q4f16) | Notes |
|---|---|---|---|
| `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `HuggingFaceTB/SmolVLM-256M-Instruct` | ~250 MB | the 2.2B does not fit; the 256M is the token-efficiency argument taken to its conclusion |
| `Qwen/Qwen3-VL-2B-Instruct` | `onnx-community/Qwen3-VL-2B-Instruct-ONNX` | ~1.2 GB | the current default, and a heavy but real browser model |
| - | `onnx-community/Qwen2-VL-2B-Instruct` | ~1.1 GB | the previous generation, well tested in the browser |
| - | `Xenova/nanoLLaVA` | ~1 GB | the older 1B option |
| `OpenGVLab/InternVL3-2B-hf`, `google/gemma-3-4b-it` | none | - | no export |

Ship both ends of the range on the same page. SmolVLM-256M loads in seconds and
is visibly weaker; Qwen3-VL-2B is a 1.2 GB download and is visibly better. That
comparison is the usual head-to-head, made concrete by a progress bar the user
actually waits on.

Gate the 2B behind the large-model warning, quoted once by `ModelPicker`, and
default the selection to the 256M.

### 3.2 Visual Question Answering, the same engine with a different prompt

Taxonomy task **Visual Question Answering** · not built. Upstream research: ViLT, BLIP-VQA, Qwen3-VL.

VQA is now just prompting a general VLM, and the browser confirms it: neither classical VQA model has a
usable export.

| Upstream (PyTorch) | Browser model | Notes |
|---|---|---|
| `dandelin/vilt-b32-finetuned-vqa` | none | no export. The 3129-answer classification approach is unavailable |
| `Salesforce/blip-vqa-base` | none | no export |
| `Qwen/Qwen3-VL-2B-Instruct` | `onnx-community/Qwen3-VL-2B-Instruct-ONNX` | prompt it; there is no VQA-specific model worth shipping |

So VQA is not a separate worker or a separate engine. It is the 3.1 page with a
question box and a short `max_new_tokens`. Keep it as a distinct **route**,
since a user looking for VQA will not think to click "image-text-to-text", but
share the engine underneath.

**What the prompt is worth** is the thing to build in: the same image and question with and without "answer in one word" produces
visibly different output, and a toggle shows it.

### 3.3 Document Question Answering, the one classical model that survived

Taxonomy task **Document Question Answering** · not built. Upstream research: LayoutLM, Donut, Qwen3-VL.

| Upstream (PyTorch) | Browser model | Size (q8) | Notes |
|---|---|---|---|
| `naver-clova-ix/donut-base-finetuned-docvqa` | `Xenova/donut-base-finetuned-docvqa` | ~500 MB | OCR-free, end to end, and it has a real export |
| `impira/layoutlm-document-qa` | none | - | needs OCR boxes as input anyway, which the browser would have to produce |
| `Qwen/Qwen3-VL-2B-Instruct` | `onnx-community/Qwen3-VL-2B-Instruct-ONNX` | ~1.2 GB | the general model, better on hard documents |
| `stepfun-ai/GOT-OCR-2.0-hf` | none | - | Florence-2 from the vision guide covers plain OCR |

```ts
const dqa = await pipeline("document-question-answering",
  "Xenova/donut-base-finetuned-docvqa", { device: "webgpu", dtype: "fp16" });
const out = await dqa(image, "What is the invoice total?");
```

This is the page with the strongest privacy argument in the whole app. People
photograph payslips, medical letters and bank statements, and this page answers
questions about them without the image leaving the device. Say that on the page,
plainly, because it is the reason the architecture was chosen.

**What resolution costs** matters more here than elsewhere: a document needs more pixels than a photo before the text becomes
legible to the encoder, so the usual "downscale to 512px" advice is wrong for
this page. Expose the resolution as a control and let the user trade speed for
legibility.

### 3.4 Video-Text-to-Text, frames in, text out

Taxonomy task **Video Text to Text** · not built. Upstream research: SmolVLM2-500M-Video, Qwen3-VL.

| Upstream (PyTorch) | Browser model | Size (q4f16) | Notes |
|---|---|---|---|
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | ~250 MB | the video-tuned 256M, and the only practical choice |
| `Qwen/Qwen3-VL-2B-Instruct` | `onnx-community/Qwen3-VL-2B-Instruct-ONNX` | ~1.2 GB | handles longer clips and timestamps |

A video model in a tab is a **frame sampler plus an image model**, and that is
all it is in PyTorch too. Sample 4 to 8 frames, pass them as a list, and
be explicit about the sampling in the UI:

```ts
// 8 frames is roughly the ceiling in a tab: each one costs its own image tokens.
const frames = await sampleFrames(videoEl, 8);        // uniform over the clip
const messages = [{ role: "user", content: [
  ...frames.map(() => ({ type: "image" })),
  { type: "text", text: "What happens in this clip?" },
]}];
```

**How many frames do you actually need** is the open question, and the browser
makes it urgent rather than academic: each frame is hundreds
of image tokens, so 4 frames against 16 is a four times difference in latency
the user feels. Put the frame count on a slider and let them find out.

The temporal-blindness test is the honest companion to that: feed the
frames in reverse and see whether the answer changes. Often it does not, which
tells you the model is describing a picture rather than a sequence. Build it as
a toggle.

### 3.5 Audio-Text-to-Text, build the cascade, not the native model

Taxonomy task **Audio Text to Text** · not built. Upstream research: Whisper plus an LLM, Granite Speech, Qwen2-Audio.

| Upstream (PyTorch) | Browser model | Notes |
|---|---|---|
| the cascade, Whisper plus Qwen3 | `onnx-community/whisper-base` plus `onnx-community/Qwen3-0.6B-ONNX` | both exported, both already in the audio and NLP catalogues |
| `ibm-granite/granite-speech-3.3-2b` | none | no export |
| `Qwen/Qwen2-Audio-7B-Instruct` | none | 7B. Not a browser model in any quantization |

The cascade is two models the app already ships, chained: transcribe with the
ASR worker, then feed the transcript to the text-generation worker. Nothing new
is downloaded.

**Then build "what the cascade cannot hear" as the point of the page.** It is
the best experiment in this category: play a clip with
laughter, a raised voice, or two speakers, and the transcript loses everything
except the words. Show the transcript and the LLM's answer side by side and let
the user hear what was discarded. A page that demonstrates its own architecture
failing is worth more than one that hides it.

Two models are live at once here. Both are small, roughly 150 MB and 400 MB, so
it fits, but it is a deliberate exception to the one-model rule and should be
noted in the code.

### 3.6 Visual Document Retrieval, the surprising one

Taxonomy task **Visual Document Retrieval** · not built. Upstream research: OCR plus BM25, CLIP, ColQwen2, ColPali.

An export exists for the small ColPali-family model, which makes this buildable:

| Upstream (PyTorch) | Browser model | Notes |
|---|---|---|
| ColQwen2 / ColPali | `onnx-community/colSmol-256M-ONNX` | late interaction over page patches, at 256M |
| CLIP single-vector control | `Xenova/clip-vit-base-patch32` | the single-vector baseline |
| OCR plus BM25 baseline | Florence-2 `<OCR>` plus 40 lines of TypeScript | the section 8 baseline |

All three stages of the comparison therefore run in the tab, over a
handful of page images the user drops in. That is a genuinely novel demo:
late-interaction retrieval is normally a server story, and seeing the
multi-vector scores beat the single-vector CLIP on the same pages is the whole
argument for ColPali in one screen.

Be careful with the scoring. Late interaction is a MaxSim over patch vectors,
not a cosine over one vector, so the ranking code is real work rather than a dot
product. Budget for it: it is the one piece of real algorithm work on the page.

### 3.7 The four that stay on a server

| Taxonomy task | Why not in a tab |
|---|---|
| Image Text to Image | InstructPix2Pix, MagicBrush and ControlNet are all SD 1.5 derivatives. Multi-gigabyte weights, tens of denoising steps |
| Image Text to Video | AnimateDiff plus SparseCtrl, and LTX-Video. Same problem with a frame count on top |
| Any to Any | Janus-Pro-1B has an ONNX export, but the generation half is a diffusion decoder; Qwen2.5-Omni-3B has none. See below |
| the native half of Audio Text to Text | Qwen2-Audio-7B is not a browser model in any quantization |

**Any to Any** deserves a note because it is the closest call.
`onnx-community/Janus-Pro-1B-ONNX` exists and is around 1 GB at `q4f16`, so the
**understanding half** is buildable: images and text in, text out. The
generation half is the part that does not port.

That makes a legitimate page, as long as it is labelled for what it is: Janus
understanding only, with the round-trip experiment demoted to prose. Do not ship
a page called "Any to Any" that only does one direction — rename the route, or
leave the placeholder.

---

## 4. Feasibility summary

| Taxonomy task | In-browser? | Recommended model | Best backend | If not |
|---|---|---|---|---|
| **Audio Text to Text** | Yes, as a cascade | Whisper-base plus Qwen3-0.6B | WebGPU | native audio LLMs to server |
| **Image Text to Text** | Yes | `HuggingFaceTB/SmolVLM-256M-Instruct`, or Qwen3-VL-2B gated | WebGPU only | InternVL3, Gemma 3 to server |
| **Image Text to Image** | No | - | - | server API |
| **Image Text to Video** | No | - | - | server API |
| **Visual Question Answering** | Yes, via a VLM | `onnx-community/Qwen3-VL-2B-Instruct-ONNX` | WebGPU only | ViLT and BLIP-VQA have no export |
| **Document Question Answering** | Yes | `Xenova/donut-base-finetuned-docvqa` | WebGPU | LayoutLM, GOT-OCR to server |
| **Video Text to Text** | Yes, sampled frames | `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` | WebGPU only | long-video understanding to server |
| **Visual Document Retrieval** | Yes, all three stages | `onnx-community/colSmol-256M-ONNX` | WebGPU | full ColQwen2 to server |
| **Any-to-Any** | Understanding half only | `onnx-community/Janus-Pro-1B-ONNX` | WebGPU | generation half to server |

Rule of thumb: **if the output is text, there is probably a page; if the output
is pixels or audio, there is not.** Every "no" in that table generates something
other than tokens.

---

## 5. Memory and performance notes

Multimodal pages are the tightest budget in the app, so the audio and vision
rules apply, plus four of their own.

- **`q4f16` on WebGPU, always.** A 2B VLM at `fp16` is a 4 GB download that no
  progress bar can make acceptable.
- **Downscale the image before the encoder**, with documents as the deliberate
  exception. Image tokens dominate the latency, and they scale with area.
- **Cap the frame count.** 4 to 8 frames for video. Each frame carries its own
  image tokens, so a 32-frame clip is not a slower page, it is a broken one.
- **Give the encode its own state.** The pause before the first token is
  seconds, and an unlabelled pause is indistinguishable from a hang.
- **Stream every generation.** `TextStreamer`, first token visible as early as
  possible, and a stop button that actually aborts.
- **One VLM live at a time**, no exception. These are the only models in the app
  large enough that a leaked session ends the tab. Null the reference, then
  `await model.dispose()` — `disposeQuietly` in the audio engines is the pattern.
- **`ModelPicker`'s large-model warning is not optional here.** Every model in
  this file is past `LARGE_MODEL_BYTES` several times over, and the
  `/text-to-audio` route is the precedent for how to gate one: state the cost,
  download nothing until the click, and assert zero Hub requests before it in an
  E2E spec.
- **The audio cascade is the one place two models coexist**, and only because
  both are small. Document it in the code so nobody "fixes" it later.

---

## 6. Reference

- **Transformers.js**: the `image-text-to-text` pipeline, plus
  `AutoProcessor.apply_chat_template`, `TextStreamer`, and
  `document-question-answering`.
- **Image helpers**: the vision guide, section 2.
  **Audio helpers**: the audio guide, section 2.
- **Page construction**: [`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md).
- **In-repo standards**:
  [`../../standards/model-page-pattern.md`](../../standards/model-page-pattern.md)
  (the four-slot contract — §7 covers where the pattern is allowed to bend),
  [`../../guides/adding-a-model.md`](../../guides/adding-a-model.md) §8.
- **The gating precedent**: `/text-to-audio` (MusicGen) is the shipped example of
  a model too heavy to load on arrival; see
  [`Audio_Models_in_React_WebGPU_and_CPU.md`](Audio_Models_in_React_WebGPU_and_CPU.md)
  §3.4.
- Model recommendations here come from a companion collection of Python
  notebooks, which is a separate project and not a dependency of this repo. Every
  browser id above was checked against the Hugging Face API — re-check with
  `just fe-e2e-models` once a catalogue module exists to check.
