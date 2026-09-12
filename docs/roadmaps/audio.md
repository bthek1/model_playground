# Audio Models in the Browser (WebGPU or CPU)

> The **Audio** category of `components/layout/taskTaxonomy.ts`, task by task: what
> runs client-side on the user's GPU (WebGPU) or CPU (WebAssembly), which checkpoint
> to use, and — for the six tasks already built — where the code lives. No Python
> server in the inference path.
>
> This is the reference the other category roadmaps cite. Sections 1 and 2
> (backend selection, audio I/O) are the shared plumbing they all build on.
> The remaining categories are still research and live as issues labelled
> [`roadmap`](https://github.com/bthek1/model_playground/issues?q=is%3Aissue+label%3Aroadmap);
> each moves here, as a file, once its first route ships.

**Audio is the category Model Playground finished first — and it is now complete.**
All six tasks in the taxonomy have working routes, so this file is a map of
shipped code rather than a proposal. Read it that way: where a section says *shipped*, the design
questions are settled and the files named are the answer. Where it says *not built*,
the model research is done and the page is not.

| Taxonomy task | Route | Status |
|---|---|---|
| Text to Speech | [`/text-to-speech`](../../frontend/src/routes/text-to-speech.tsx) | Shipped — Kokoro, MMS-VITS, SpeechT5 |
| Text to Audio | [`/text-to-audio`](../../frontend/src/routes/text-to-audio.tsx) | Shipped — MusicGen, gated behind its size |
| Automatic Speech Recognition | [`/asr`](../../frontend/src/routes/asr.tsx) | Shipped — Whisper, Moonshine, live mic |
| Audio to Audio | [`/audio-to-audio`](../../frontend/src/routes/audio-to-audio.tsx) | Shipped — DeepFilterNet3 on bare ONNX |
| Audio Classification | [`/audio-classification`](../../frontend/src/routes/audio-classification.tsx) | Shipped — AST, wav2vec2-KS, CLAP |
| Voice Activity Detection | [`/vad`](../../frontend/src/routes/vad.tsx) | Shipped — Silero VAD v5 on bare ONNX, plus an energy baseline |

The build-out is recorded in the git history and, for the last route, in the
closed plan issue [#9](https://github.com/bthek1/model_playground/issues/9).
This page began as issue #1 and was moved here when the category completed: a
roadmap that documents *shipped* code has to be reviewable in the same pull
request as the code it describes, which an issue body cannot be.

Every model id below is in the catalogue and is checked against the Hugging Face API
by `just fe-e2e-models` — re-run it before adding one.

---

## 1. The runtime, and how a backend is chosen

Both runtimes are already dependencies (`frontend/package.json`); nothing needs
installing:

- **`@huggingface/transformers`** — the Transformers.js pipelines, running ONNX
  exports of the same Hugging Face checkpoints on ONNX Runtime Web.
- **`kokoro-js`** — the purpose-built browser package for Kokoro TTS.
- **`onnxruntime-web`** — bare ONNX, for the two tasks with no Transformers.js
  pipeline: speech enhancement (§3.5) and VAD (§3.6).

Two execution backends:

- **WebGPU** — runs the model on the GPU, 5–30x faster than CPU. Needs a secure
  context (HTTPS or `localhost`) and, on Firefox, `dom.webgpu.enabled`.
- **WASM (CPU)** — WebAssembly + SIMD + threads. Slower, works everywhere, and is
  the fallback that must never be allowed to break.

The probe and the precision table live in
[`frontend/src/model/backend.ts`](../../frontend/src/model/backend.ts) — do not
re-derive them per task. (They were `audio/backend.ts` until Computer Vision
arrived; nothing in them was ever audio-specific, so they moved to `model/`
rather than being copied. The size guardrail moved with them, to
[`model/size.ts`](../../frontend/src/model/size.ts).)

```ts
import { pickBackend, loadOpts, asrLoadOpts } from "@/model/backend";

const backend = await pickBackend();  // "webgpu" | "wasm", never throws
const opts = loadOpts(backend);       // { device: "webgpu", dtype: "fp16" } | { device: "wasm", dtype: "q8" }
```

`pickBackend()` requests an *adapter* rather than merely checking `navigator.gpu`,
because some browsers expose the API with no usable GPU behind it. The backend is
resolved **once inside the worker, before `ready`**, and is fixed for that worker's
life (model-page-pattern §6).

**`asrLoadOpts()` is the documented exception**: on WASM it keeps the decoder at
`fp32`, because the quantized Whisper/Moonshine decoders cannot open a session on
the ONNX Runtime bundled with `@huggingface/transformers` 4.2.0. A uniform `q8`
breaks the universal fallback entirely. The comment in `backend.ts` carries the full
reasoning — don't simplify it away.

Model weights are downloaded once and held in Cache Storage, so a second load is
free and works offline. `model/cache.ts` probes that bucket, which is what lets the
picker show a `Cached` badge and a refresh resume without spending bandwidth.

---

## 2. Audio I/O — the browser's `soundfile` + `librosa`

Everything is in [`frontend/src/audio/io.ts`](../../frontend/src/audio/io.ts).
The Web Audio API does the decode and the resample; there is no DSP to write.

```ts
import { decodeToMono, recordMic } from "@/audio/io";

const audio = await decodeToMono(await file.arrayBuffer());  // mono Float32 @ 16 kHz
const take  = await recordMic(5);                            // 5 s from the mic
```

`decodeToMono` renders the decoded buffer through an `OfflineAudioContext` created
at the target rate, which resamples for free. It accepts any format the browser can
decode — wav, mp3, ogg, flac.

**16 kHz is the default, not a law.** ASR and classification want 16 kHz mono
Float32; speech enhancement wants **48 kHz** and passes `SAMPLE_RATE` explicitly. A
16 kHz assumption leaking into that route silently discards the band the model
exists to repair (§3.5).

**Routes do not call these directly any more — `useAudioPick` does.**
[`hooks/useAudioPick.ts`](../../frontend/src/hooks/useAudioPick.ts), with
[`AudioSourcePanel`](../../frontend/src/components/audio/AudioSourcePanel.tsx), is the
audio counterpart of `useImagePick` / `ImageSourcePanel`, and it exists because all four
audio routes had the same bug: `onFile` decoded **and ran the model** in one function, as
did the record button and each sample clip. There was no INPUT stage at all, so browsing
the samples cost an inference per click (one of them is 60 seconds long), and re-running
the clip you already had meant uploading it again. The hook holds a decoded clip and
nothing else; the route's GENERATE button is the only caller of `run`
([model-page-pattern.md §1.6](../standards/model-page-pattern.md)). Pass `sampleRate` to
it the same way you would to `decodeToMono` — `/audio-to-audio` passes 48 000.

Its one non-obvious rule: **`take()` returns a copy.** Every audio worker receives the
`Float32Array`'s buffer as a transfer and detaches it, so handing over the stored clip
would blank the input waveform and leave the second GENERATE with nothing to send.

Playback and WAV export for generated audio are in the same module (`play`,
`toWavBlob`), and `audio/waveform.ts` reduces a `Float32Array` to the min/max
envelope the output panels draw.

### Everything runs in a Web Worker

Inference blocks the thread it runs on, so nothing model-shaped touches the UI
thread. The worker plumbing exists once, in
[`frontend/src/model/useModelWorker.ts`](../../frontend/src/model/useModelWorker.ts):
worker lifecycle keyed on a `key` string, the id-correlated pending table, and the
two state machines. A task hook is a thin typed wrapper over it.

**One worker per modality, not per task.** The audio category has five, and the
split is deliberate:

| Worker | Owns |
|---|---|
| [`audio/pipeline.worker.ts`](../../frontend/src/audio/pipeline.worker.ts) | every discriminative pipeline task — the task string travels in the `load` message |
| [`audio/asr.worker.ts`](../../frontend/src/audio/asr.worker.ts) | ASR, because it drives the real-time capture loop |
| [`audio/tts.worker.ts`](../../frontend/src/audio/tts.worker.ts) | the whole text→audio modality: Kokoro, MMS/SpeechT5 *and* MusicGen behind one `TtsSynthesizer` interface |
| [`audio/enhance/enhance.worker.ts`](../../frontend/src/audio/enhance/enhance.worker.ts) | bare ONNX Runtime with hand-written DSP |
| [`audio/vad/vad.worker.ts`](../../frontend/src/audio/vad/vad.worker.ts) | bare ONNX Runtime, recurrent frame loop — *and* the no-weights energy baseline |

Each `*.worker.ts` is a thin wrapper around a pure engine module
(`asrEngine`/`pipelineEngine`/`ttsEngine`/`enhanceEngine`/`vadEngine`) that imports nothing from
`self` — which is what makes the engines unit-testable. **Every engine owes three
behaviours**: one model live at a time (null the reference *first*, then dispose via
`disposeQuietly`), warm-up on load (one throwaway inference posting
`{ status: "warmup" }`, which must never fail the load), and never blocking the main
thread.

---

## 3. Task by task

### 3.1 Automatic Speech Recognition — shipped

Route [`/asr`](../../frontend/src/routes/asr.tsx) · catalogue
[`audio/types.ts`](../../frontend/src/audio/types.ts) · hooks `useAsr`,
`useLiveAsr`.

ASR is the flagship in-browser task: Whisper and Moonshine have first-class ONNX and
WebGPU support, run faster than real time on a laptop GPU, and stream.

| Model | Params | Download | Notes |
|---|---|---|---|
| `onnx-community/whisper-base` | 74M | 140 MB WebGPU · **221 MB WASM** | timestamps, 99 languages, translate |
| `onnx-community/moonshine-tiny-ONNX` | 27M | 82 MB WASM | English, low latency, the live-caption choice |

The WASM download is roughly 3x the params estimate because of the fp32 decoder
(§1), so both entries carry **measured `bytes` per backend** rather than letting
`model/size.ts` estimate. Do the same for any model whose estimate would mislead.

**Live transcription re-transcribes only the tail 30 s**, so the model's own
timestamps restart at 0 on a longer take. `useLiveAsr`'s `shiftChunks()` offsets them
by the window start before the route renders `m:ss` — never render `chunks` straight
from the worker.

Qwen3-ASR and Parakeet have no ONNX export; use Whisper or Moonshine.

### 3.2 Audio Classification — shipped

Route [`/audio-classification`](../../frontend/src/routes/audio-classification.tsx)
· catalogue [`audio/classification.ts`](../../frontend/src/audio/classification.ts)
· hook `useAudioClassifier` over `usePipeline`.

Two flavours through the one generic pipeline worker: fixed-label tagging, and
open-set scoring against prompts the user types.

| Model | Pipeline task | Notes |
|---|---|---|
| `Xenova/ast-finetuned-audioset-10-10-0.4593` | `audio-classification` | 527 AudioSet sound-event tags |
| `Xenova/wav2vec2-base-superb-ks` | `audio-classification` | speech-command keyword spotting |
| `Xenova/clap-htsat-unfused` | `zero-shot-audio-classification` | score against free-text prompts |

**These are the `Xenova/*` repos, not `onnx-community/*`.** The latter do not exist
for AST or wav2vec2-KS — the Hub returns 401 and the load fails with "Unauthorized
access to file". That mistake shipped once; `just fe-e2e-models` is what catches it
now, in seconds.

Adding another discriminative audio task means extending the `PipelineTask` union in
`audio/pipelineTypes.ts`. It does not mean a new worker.

### 3.3 Text to Speech — shipped

Route [`/text-to-speech`](../../frontend/src/routes/text-to-speech.tsx) ·
catalogue [`audio/tts.ts`](../../frontend/src/audio/tts.ts) · hook `useTts`.

| Model | Params | Path | Notes |
|---|---|---|---|
| `onnx-community/Kokoro-82M-v1.0-ONNX` | 82M | `kokoro-js` | best small-model quality, six named voices, WebGPU |
| `Xenova/mms-tts-eng` | 36M | `text-to-speech` pipeline | tiny, end-to-end; swap the id for other languages |
| `Xenova/speecht5_tts` | 144M | `text-to-speech` pipeline | needs the x-vector speaker embedding |

Kokoro is the default and the only model with a voice list — `TtsModel.voices` is
optional in the catalogue for exactly that reason, and the pipeline models ignore
the `voice` option.

Bark (1B codec LM) is too heavy for a tab. It stays server-side.

### 3.4 Text to Audio (music / SFX) — shipped, and deliberately gated

Route [`/text-to-audio`](../../frontend/src/routes/text-to-audio.tsx) · catalogue
[`audio/textToAudio.ts`](../../frontend/src/audio/textToAudio.ts) · shares
`tts.worker.ts`.

`Xenova/musicgen-small` is 300M params and a **571 MB q8 / ~1 GB fp16** download,
generating autoregressively at ~50 tokens per second of audio. The route states that
cost and downloads nothing until the user opts in; an E2E spec asserts zero Hub
requests before the click. **Use this pattern for anything this heavy** — the size
guardrail in `ModelPicker` is the mechanism, not a suggestion.

Two implementation notes worth keeping:

- MusicGen needs `MusicgenForConditionalGeneration` **directly**. The `text-to-audio`
  *pipeline* throws "Missing the following inputs: input_ids" on 4.2.0.
- Generation length is capped (`MAX_SECONDS = 15`, default 5) because the cost is
  linear in the audio duration and a user cannot tell a slow model from a hung tab.

AudioLDM and Stable Audio are `diffusers` latent-diffusion models with no
Transformers.js path. They stay on a server.

### 3.5 Audio to Audio (speech enhancement) — shipped on bare ONNX

Route [`/audio-to-audio`](../../frontend/src/routes/audio-to-audio.tsx) · engine
[`audio/enhance/`](../../frontend/src/audio/enhance/) · hook `useEnhance` (the
reference implementation of the §3 hook contract).

There is no `audio-to-audio` pipeline in Transformers.js, so this route talks to
`onnxruntime-web` directly. DeepFilterNet3 (`soniqo/DeepFilterNet3-ONNX`, 2.3M
params, 8.3 MB) publishes the neural graph only — normalised ERB/spectral features
in, an ERB mask plus complex deep-filter coefficients out — so **the STFT, ERB
filterbank, feature normalisation, deep filtering and overlap-add are ours.**

Its four standing rules, each of which cost something to learn:

- **48 kHz, not 16 kHz.** The only audio route that isn't. Pass `SAMPLE_RATE` to
  `decodeToMono` / `recordMic`.
- **Import ORT as `onnxruntime-web/webgpu`** — the same subpath Transformers.js uses,
  so Vite emits one shared WASM asset instead of a second 26 MB build. It is in
  `optimizeDeps.include` too: discovered inside a Worker mid-session, Vite
  re-optimises and reloads the page mid-load.
- **Validate `deepfilter-auxiliary.bin` on load and refuse to run on a mismatch**
  (`parseAux`). It is 124 KB of untyped float32 whose forward matrix is `[481,32]`
  and inverse `[32,481]` — *different* orders, where a transposed read still looks
  like a valid matrix.
- **Wrong DSP fails silently**, so it is pinned against the official libDF
  implementation through a captured fixture rather than against our own
  expectations. `SPEC_SCALE = 2*hop/fft²` is load-bearing: the unit-norm feature
  divides by `sqrt(state)` and is *not* level-invariant, so dropping it makes the
  network mask clean speech away as noise.

**Unit tests mock the network and ORT, so they cannot catch a broken model.** Both
DSP bugs above shipped past a green suite. `just fe-e2e-enhance` measures a real SDR
improvement on real weights; that is the guard.

Demucs stem separation is large and non-causal — it wants the whole track. Prefer
server-side.

### 3.6 Voice Activity Detection — shipped

Route [`/vad`](../../frontend/src/routes/vad.tsx) · module
[`audio/vad/`](../../frontend/src/audio/vad/) · hook `useVad`. Built in #9.

| Model | Params | Download | Notes |
|---|---|---|---|
| `onnx-community/silero-vad` | 0.56M | 2.2 MB fp32 (`onnx/model.onnx`) | the standard. Frame-level speech probability, robust to noise and music |
| Energy VAD | — | none | short-time level, no model at all — the baseline to beat, and the page works before any download |

Silero is loaded as the **fp32** graph rather than a quantized variant on purpose:
2.2 MB is already noise next to every other model here, an LSTM graph is exactly
what q8 export bugs bite (see `asrLoadOpts`, §1), and 0.3 ms per frame leaves
nothing worth trading accuracy for. The energy baseline is the catalogue's only
entry with `repo: null` — a model row that downloads nothing, which is what lets
the page be useful before the first byte.

The output is a *timeline*, not a label: the probability is drawn under the
waveform on the same time axis, and the threshold is the user's to drag.
Dragging re-derives segments from the scores already in hand
([`segments.ts`](../../frontend/src/audio/vad/segments.ts)) — the model never re-runs.
Segmentation bridges short gaps (`minSilenceMs = 100`) and drops short runs
(`minSpeechMs = 250`) rather than thresholding frame-by-frame: speech dips below
any threshold at a stop consonant, so a raw mask cuts one phrase into four.

**Two corrections to what this section used to say**, both found by reading the
graph rather than the model card:

- **The window is 576 samples, not 512.** Silero v5 wants 64 samples of preceding
  context prepended to each 512-sample frame. The input dims are dynamic, so a
  bare 512 runs happily and returns scores that never fire — measured on jfk.wav,
  speech reads **0.05** without the context and **0.83–0.99** with it. No mocked
  test can see this; the `@slow` spec is the only guard.
- **v5 has one `state` tensor, not `h`/`c`.** Inputs are `input` f32`[B,N]`,
  `state` f32`[2,B,128]`, `sr` int64 scalar; outputs are `output` f32`[B,1]` and
  `stateN`. State and context reset per clip, never per session.

**It runs on WASM by design, not as a fallback.** 0.30 ms per 32 ms frame on CPU
is about 100x real time, so a per-frame GPU dispatch and readback would cost more
than the work — and Silero's LSTM/`If` ops aren't covered by ORT's WebGPU
provider anyway. The `@slow` spec asserts the backend so this can't be loosened
by accident. `just fe-e2e-vad` runs that spec in seconds against real weights —
it is cheap enough to run on any change to `audio/vad/`.

Its second life as a gate in front of ASR — `useLiveAsr` still transcribes
silence as eagerly as speech — was deliberately left out of #9 and is unclaimed.

### 3.7 Audio-Text-to-Text — see the Multimodal guide

Audio LLMs (Qwen2-Audio and friends) are multi-billion-parameter and have no browser
runtime. The buildable version is a **cascade** — ASR then a small text model, both
already in the catalogue — and it is specified in
[`Multimodal_Models_in_React_WebGPU_and_CPU.md`](https://github.com/bthek1/model_playground/issues/4)
§3.5.

---

## 4. Feasibility summary

| Task | In-browser? | Model | Best backend | If not |
|---|---|---|---|---|
| **ASR** | Yes — shipped | Whisper-base / Moonshine-tiny | WebGPU (WASM ok) | — |
| **Audio classification** | Yes — shipped | AST / wav2vec2-KS / CLAP | WASM or WebGPU | — |
| **TTS** | Yes — shipped | Kokoro-82M, MMS-VITS, SpeechT5 | WebGPU (Kokoro) / WASM | Bark → server |
| **Text-to-Audio** | Partial — shipped, gated | MusicGen-small, short clips only | WebGPU | AudioLDM / Stable Audio → server |
| **Audio-to-Audio** | Yes — shipped | DeepFilterNet3 (bare ONNX) | WebGPU / WASM | Demucs, voice conversion → server |
| **Voice Activity Detection** | Yes — shipped | `onnx-community/silero-vad` + an energy baseline | WASM (by design) | — |
| **Audio-Text-to-Text** | As a cascade | Whisper + a small LLM | WebGPU | native audio LLMs → server |

Rule of thumb: **discriminative and small** (ASR, classification, VAD, small TTS)
runs well client-side; **generative and large** (music, separation, audio LLMs)
belongs on a server with the browser as the UI.

---

## 5. Memory and performance notes

A tab is a tighter budget than a 12 GB card and far less forgiving.

- **One model live at a time.** Null the reference, *then* `dispose()` — a failed
  teardown must not leave a stale model live. There is no
  `torch.cuda.empty_cache()`; disposing is the entire mechanism.
- **Quantize for CPU.** `q8` cuts the download and RAM 2–4x on WASM; `fp16` on
  WebGPU. The exception is ASR's fp32 decoder (§1).
- **Size before you load.** `model/size.ts` quotes both backends and warns past
  `LARGE_MODEL_BYTES` (200 MB). Supply measured `bytes` whenever the params estimate
  would mislead. The estimate is quoted **once**, by `ModelPicker`.
- **Feature-detect and degrade.** Always `pickBackend()`; never assume WebGPU. A GPU
  failure *after* load surfaces as a run error, not a silent fallback — switching
  quietly would make the timings the user is reading meaningless.
- **Warm up on load.** The first inference compiles WebGPU shaders and JITs the WASM.
  Charge it to the load, not to the user's first real request.
- **Report the aggregate, never a raw per-file event.** Transformers.js reports
  progress per file (4–8 of them); `model/progress.ts` folds them into one monotonic
  percentage. Rendering the raw payload makes the bar restart at zero eight times.
- **Cache is the refresh story.** Weights persist in Cache Storage, so a resume is
  free — which is precisely why `useModelSelection` auto-loads only when a stored
  intent meets a cache hit.

---

## 6. Reference

- **Page construction**: [`docs/guides/adding-a-task-page.md`](../guides/adding-a-task-page.md).
- **The page contract**: [`docs/standards/model-page-pattern.md`](../standards/model-page-pattern.md).
- **Recipes**: [`docs/guides/adding-a-model.md`](../guides/adding-a-model.md) §8
  (Transformers.js) and §9 (a bare ONNX graph).
- **What shipped**: the routes themselves, plus closed plan issue
  [#9](https://github.com/bthek1/model_playground/issues/9) for VAD.
- **Guards on real weights**: `just fe-e2e-slow` (all `@slow` audio specs),
  `just fe-e2e-enhance` (SDR on DeepFilterNet3), `just fe-e2e-vad` (Silero on
  jfk.wav), `just fe-e2e-models` (every catalogue id resolves on the Hub).
- **Transformers.js** — `@huggingface/transformers`: the
  `automatic-speech-recognition`, `audio-classification`,
  `zero-shot-audio-classification`, `text-to-speech` and `text-to-audio` pipelines.
- **kokoro-js** — browser-native Kokoro-82M TTS with WebGPU.
- **onnxruntime-web** — direct ONNX inference for models with no pipeline; always
  the `onnxruntime-web/webgpu` entry point.
- **ONNX model hubs** — the `onnx-community/*` and `Xenova/*` orgs. Neither mirrors
  everything; verify with `just fe-e2e-models`.



