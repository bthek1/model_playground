# Audio Models in the Browser (Transformers.js / ONNX Runtime Web)

**Status:** In Progress (Phases 1–4 and 6 complete; Phase 5 partially complete — Text-to-Audio
shipped, Audio-to-Audio deliberately left server-side. All in-browser tasks **verified in a real
browser**.)

Bring the audio tasks from the `DL_tasks/nbs/Audio/` notebooks into the React frontend,
running the models **client-side** on the user's GPU (WebGPU) or CPU (WASM) — no Python
server in the inference path. The vertical slice ships **Automatic Speech Recognition
(ASR)** end-to-end first as the reference implementation; later phases add classification,
TTS, and the partial/server-side tasks.

> Source notebooks (in the separate `DL_tasks` notebook set, not this repo):
> `00_Text_to_Speech`, `01_Text_to_Audio`, `02_Automatic_Speech_Recognition`,
> `03_Audio_to_Audio`, `04_Audio_Classification`.

---

## Context

The Python notebooks load every model through Hugging Face `transformers`. The browser
equivalent is **Transformers.js** (`@huggingface/transformers`), which runs the *same* HF
checkpoints (exported to ONNX) on **ONNX Runtime Web**, with two execution backends:

- **WebGPU** — runs the model on the GPU. 5–30× faster than CPU; needs a Chromium-based
  browser (or Safari 18+ / Firefox with the flag) with WebGPU enabled.
- **WASM (CPU)** — WebAssembly + SIMD + threads. Works everywhere, slower, universal fallback.

Tasks with no ONNX / Transformers.js path (source separation, diffusion audio) drop down to
**ONNX Runtime Web directly** with a hand-exported model, or stay server-side. The
[feasibility table](#feasibility-summary) says which is which.

### Architectural note — how this coexists with the raw-WebGPU rule

The repo's raw-WebGPU-only rule is **scoped to the `src/webgpu/` custom-kernel runtime**
(hand-written WGSL for matmul, linear-model training, tensor ops). Running *pretrained*
models in the UI is a different concern and **may use Transformers.js / ONNX Runtime Web**.
[`CLAUDE.md`](../../../CLAUDE.md) and
[`.github/copilot-instructions.md`](../../../.github/copilot-instructions.md) were updated to
document this carve-out (done alongside this plan; re-confirm in Phase 1). The new audio code
lives in its **own domain folder** (`src/audio/`) so it never entangles with `src/webgpu/`.

---

## Design decisions

- **New `src/audio/` domain folder.** Transformers.js helpers, workers, and model wrappers
  live here — kept apart from the raw-WebGPU runtime in `src/webgpu/`. (Note `src/webgpu/`
  already has a `device.ts`; the audio backend picker is named `backend.ts` to avoid confusion.)
- **Always run inference in a Web Worker.** Inference blocks its thread; the model loads and
  runs in a worker and posts messages back, mirroring the existing `webgpu/worker.ts` +
  `webgpu/workerClient.ts` split. The React UI never blocks.
- **Feature-detect and fall back.** `pickBackend()` requests a WebGPU adapter and degrades to
  WASM; the app never assumes WebGPU exists. Reuse the spirit of
  `webgpu/capabilities.ts::detectWebGPU()` (never throws, returns a status).
- **Precision by backend.** WebGPU → `dtype: "fp16"` (matches the notebooks' `float16`); WASM →
  quantized `q8`/`q4` to keep download + RAM small.
- **One model live at a time.** Dispose (`await task.dispose()`), null the reference, let GC +
  WebGPU teardown reclaim memory — the browser analog of the notebooks' `free_memory()`. A tab's
  budget is tighter than the notebooks' 12 GB box.
- **Taxonomy-driven routing.** ASR is already the placeholder task *Automatic Speech Recognition*
  under the **Audio** category in
  [`taskTaxonomy.ts`](../../../frontend/src/components/layout/taskTaxonomy.ts). Wire it to a real
  route via `REAL_ROUTES` (`automatic-speech-recognition → /asr`); later audio tasks follow the
  same one-line pattern.
- **Follow the model-visualization standard.** The ASR page presents a waveform + live transcript
  (with timestamps) per [`docs/standards/model-visualization.md`](../../standards/model-visualization.md);
  reuse `components/viz/` primitives where they fit rather than inventing parallel ones.
- **Weights cache for free.** Transformers.js caches ONNX weights in Cache Storage / IndexedDB, so
  the second load is instant and works offline — the analog of the notebooks' `hf_cache`.

---

## Phase 1 — Dependencies, docs, and the shared audio stack ✅

*Done: `@huggingface/transformers@4.2.0` installed; docs carve-out landed in `CLAUDE.md` +
`copilot-instructions.md`; `src/audio/backend.ts` (`pickBackend`/`loadOpts`) and `src/audio/io.ts`
(`decodeToMono`/`recordMic`/`play`/`toWavBlob`) added with unit tests. Vite bundles the ASR worker +
ORT WASM asset cleanly.*

- Add deps: `npm i @huggingface/transformers` (pulls `onnxruntime-web` transitively). Defer a
  direct `onnxruntime-web` install to Phase 5 (custom models only).
- **Confirm the docs carve-out** in [`CLAUDE.md`](../../../CLAUDE.md) and
  [`.github/copilot-instructions.md`](../../../.github/copilot-instructions.md) (scoping the
  raw-WebGPU rule to `src/webgpu/`, permitting Transformers.js/ONNX-Web for pretrained models).
- `src/audio/backend.ts` — `pickBackend(): Promise<"webgpu" | "wasm">` (request an adapter to
  confirm a usable GPU) and `loadOpts(backend)` → `{ device, dtype }` (`fp16` on WebGPU, `q8` on WASM).
- `src/audio/io.ts` — Web Audio I/O helpers (the browser's `soundfile` + `librosa`):
  - `decodeToMono(data, targetRate = 16000)` — decode any `File`/`Blob`/`ArrayBuffer` to mono
    `Float32` at a target rate via `OfflineAudioContext` (resamples for free). Whisper wants 16 kHz mono.
  - `recordMic(seconds, targetRate)` — capture N seconds from the mic as mono `Float32`.
  - `play(samples, sampleRate)` and `toWavBlob(samples, sampleRate)` — playback + a downloadable
    16-bit PCM WAV (for TTS/music output in later phases).
- Vite worker config: confirm `new Worker(new URL(...), { type: "module" })` bundling works for the
  Transformers.js worker (it ships WASM/ONNX assets — verify they're served, not tree-shaken).

## Phase 2 — ASR worker + hook + route (ship this slice) ✅

*Done: `src/audio/types.ts` (protocol + model catalogue), `asrEngine.ts` (testable message handler,
one-model-live dispose), `asr.worker.ts` (thin wrapper), `asrClient.ts` (worker factory),
`hooks/useAsr.ts` (id-correlated `transcribe`), and `routes/asr.tsx` (model picker, mic/upload,
load-progress, transcript). `REAL_ROUTES["automatic-speech-recognition"] = "/asr"` wired.
**Timestamps — restored 2026-09-02.** An earlier revision of this plan claimed a timestamped
transcript; the streaming rewrite to `useLiveAsr` had dropped it, surfacing only `text`. Now fixed:
`useLiveAsr` exposes `chunks`, and `routes/asr.tsx` renders each segment as `m:ss` + text, falling
back to plain text when a model returns no segmentation.

The subtlety is that the live loop only re-transcribes the tail 30 s, so the model's timestamps
restart at 0 once a take is longer than that — on screen they would rewind mid-recording.
`shiftChunks()` offsets them by the window start so they stay **relative to the whole take**.
Unit-tested (offset, open-ended `end: null`, cleared on a new capture) plus `formatTimestamp`
(`65 → "1:05"`, negatives clamped), and verified in a real browser: the JFK sample renders
`0:00 · "And so my fellow Americans, ask not what your country can do for you…"`.
Unit-tested: engine protocol, hook lifecycle, `/asr` route rendering, audio I/O helpers
(`decodeToMono`/`play`/`recordMic`/`toWavBlob`), and taxonomy mapping — all green (0 lint errors, clean
build). **Manual browser check still pending** (needs HTTPS + a real model download; see Testing).*

**Record + visualize + re-apply — done.** The ASR page now fulfils the
waveform half of the model-visualization design decision: `audio/waveform.ts`
(`computePeaks`/`formatDuration`, unit-tested) feeds two theme-aware canvases in
`components/audio/Waveform.tsx` — `<LiveWaveform>` (AnalyserNode-driven scrolling mic
signal while recording) and `<Waveform>` (static min/max-peak view of a clip; bars ride
`currentColor` so both themes work). `useLiveAsr` retains the take: it exposes the live
`stream` for visualization and, after `stop()` (or an upload via `transcribeClip`), the
decoded 16 kHz `clip` — copies are sent to the worker since `transcribe` transfers its
buffer. The route's Audio card offers **Play** (`io.play`), **Download WAV** (`toWavBlob`),
and **Transcribe clip** (re-apply the currently selected model to the retained take).
The final pass on stop is no longer skipped when a live tick is in flight.

**Real-time captioning (streaming) — done.** The page now does live voice-to-text rather than a
one-shot "record 5 s → transcribe". `hooks/useLiveAsr.ts` captures the mic continuously with
`MediaRecorder` (1.5 s timeslice) and, on each chunk, decodes the take-so-far and re-transcribes its last
30 s (Whisper's native chunk size) through the worker — so the transcript refines live as the user speaks.
Overlapping ticks are skipped (one transcription in flight at a time). `routes/asr.tsx` is now a
Start/Stop listening toggle with a live "Listening…" indicator and a growing transcript; file upload
remains as a one-shot path (`transcribeClip`). Unit-tested in `useLiveAsr.test.ts` (start→chunk→text,
busy-skip, stop→final pass + mic release, one-shot clip).*

Reference model: `onnx-community/whisper-base` (timestamps, 99 langs, translate). Offer
`onnx-community/moonshine-tiny-ONNX` as the low-latency English option (the notebook's edge row).
The notebook's Qwen3-ASR / Parakeet are **not** ONNX-exported — use Whisper/Moonshine in-browser.

- `src/audio/asr.worker.ts` — loads the `automatic-speech-recognition` pipeline with
  `loadOpts(await pickBackend())` and a `progress_callback`; handles `load` / `run` messages and
  posts `progress` / `ready` / `result` back. Run args mirror the notebook: `return_timestamps: true`,
  `chunk_length_s: 30` (long-form chunking), `language`, `task` (`transcribe` | `translate`).
- `src/hooks/useAsr.ts` — spawns the worker, exposes `{ ready, progress, text, chunks, transcribe }`,
  and `terminate()`s the worker on unmount (frees the model + backend context).
- `src/routes/asr.tsx` — file route `/asr`: file upload + `recordMic` button, model picker
  (Whisper-base / Moonshine-tiny), a load-progress bar, and the transcript with per-chunk timestamps.
  Degrade gracefully when neither backend is `ready` (reuse the WebGPU-unavailable notice pattern from
  [`routes/training.tsx`](../../../frontend/src/routes/training.tsx)).
- Wire `REAL_ROUTES["automatic-speech-recognition"] = "/asr"` in
  [`taskTaxonomy.ts`](../../../frontend/src/components/layout/taskTaxonomy.ts) so the Audio-category
  task links to the real page instead of the `/tasks/$slug` placeholder.

## Phase 3 — Audio classification (fully in-browser) ✅

*Done: a **task-agnostic pipeline worker** (`pipelineTypes.ts` protocol, `pipelineEngine.ts` handler that
spreads run args positionally, `pipeline.worker.ts` thin wrapper, `pipelineClient.ts` factory) + the
generic `hooks/usePipeline.ts`. `audio/classification.ts` holds the model catalogue; `hooks/useAudioClassifier.ts`
shapes the args per task (`{ top_k }` for fixed-label, candidate-label list for zero-shot). `routes/audio-classification.tsx`
has the model picker, a zero-shot prompt textarea, mic/upload, and ranked score bars, reusing the extracted
`components/audio/ModelStatus.tsx` (now shared with the ASR route). `REAL_ROUTES["audio-classification"] = "/audio-classification"`
wired. Unit-tested: engine (arg-spreading, dispose, errors), `usePipeline` (load/ready/correlated-run/terminate),
`useAudioClassifier` (per-task arg shaping, empty-label guard), route rendering, taxonomy mapping. ASR keeps
its own worker (it drives the real-time capture loop); everything discriminative routes through the generic one.
**Manual browser check pending** (real model downloads).*

Direct port of `04_Audio_Classification`; all families export to ONNX:

| Notebook model | Browser model | Pipeline |
|----------------|---------------|----------|
| AST AudioSet tagging | `Xenova/ast-finetuned-audioset-10-10-0.4593` | `audio-classification` |
| wav2vec2 keyword spotting | `Xenova/wav2vec2-base-superb-ks` | `audio-classification` |
| CLAP zero-shot | `Xenova/clap-htsat-unfused` | `zero-shot-audio-classification` |

- ✅ Generalized the Phase-2 worker into a small task-agnostic worker factory (pipeline type + model id +
  opts in the `load` message) rather than one worker file per task.
- ✅ `src/routes/audio-classification.tsx` (+ `useAudioClassifier` hook): fixed-label top-k for AST/KWS,
  and a free-text prompt list for zero-shot CLAP (the notebook's open-set path).
- ✅ Map `audio-classification → /audio-classification` in `REAL_ROUTES`.

## Phase 4 — Text-to-Speech (in-browser; pick the right model) ✅

*Done: `kokoro-js@1.2.1` installed. `audio/tts.ts` (catalogue: Kokoro-82M default with six named voices ·
MMS-VITS · SpeechT5 with the x-vector speaker embedding — plus the worker protocol), `ttsEngine.ts`
(testable handler; result audio buffer **transferred** back to the main thread), `tts.worker.ts` (two
engines behind one `TtsSynthesizer` interface — `KokoroTTS.from_pretrained` and the `text-to-speech`
pipeline), `ttsClient.ts`, `hooks/useTts.ts`, and `routes/text-to-speech.tsx` (model + voice picker, text
input, Speak → immediate playback, replay + download-WAV via the Phase-1 `play`/`toWavBlob`).
`REAL_ROUTES["text-to-speech"] = "/text-to-speech"` wired. TTS gets its **own worker** (not the generic
pipeline worker) deliberately: the modality differs (text in → audio out) and kokoro-js is heavy — it
bundles as its own ~2.7 MB chunk so classification users never download it. Unit-tested: engine
(load/synthesise/transfer/dispose/errors), `useTts` lifecycle, route rendering (voice-picker visibility,
synthesise-and-play flow), taxonomy mapping. **Manual browser check pending** (real model downloads).*

Port of `00_Text_to_Speech`. Uses the Phase-1 `play`/`toWavBlob` output helpers.

| Notebook model | Browser path | Backend | Notes |
|----------------|--------------|---------|-------|
| Kokoro-82M | **`kokoro-js`** package | WebGPU / WASM | best small-model quality; purpose-built for the browser |
| MMS-VITS | `Xenova/mms-tts-eng` (`text-to-speech`) | WASM | tiny, end-to-end, multilingual by model-id swap |
| SpeechT5 | `Xenova/speecht5_tts` (`text-to-speech`) | WASM (WebGPU partial) | needs the x-vector speaker embedding, as in the notebook |
| Bark (1B codec LM) | not recommended in-browser | — | too heavy; keep server-side |

- Add `kokoro-js` (its own WebGPU package) as the default TTS; SpeechT5/MMS via the standard pipeline.
- `src/routes/text-to-speech.tsx` (+ hook): text input, voice/model picker, play + download-WAV.
- Map `text-to-speech → /text-to-speech`.

## Phase 5 — Partial / server-boundary tasks (Text-to-Audio, Audio-to-Audio) 🟡

*Text-to-Audio **shipped**; Audio-to-Audio **researched and deliberately not built** — the
in-browser path is a DSP project, not a wiring job. Details below.*

### Text-to-Audio (`01`) — done ✅

`routes/text-to-audio.tsx` + `audio/textToAudio.ts`, reusing the **TTS worker** (same modality:
text in → audio out, so `TtsSynthesizer` already described it) rather than adding a fourth worker.
`REAL_ROUTES["text-to-audio"] = "/text-to-audio"` wired. Verified in a real browser: downloads,
warms up, reports ready on WASM, and generates a 0.9 s clip at 32 kHz in ~6 s.

Three things the build turned up:

- **The `text-to-audio` pipeline is broken for MusicGen** on `@huggingface/transformers` 4.2.0 —
  `pipeline("text-to-audio", …)("prompt")` throws *"Missing the following inputs: input_ids"*. The
  worker therefore drives `MusicgenForConditionalGeneration` + `AutoTokenizer` directly. Re-test on
  upgrade. (The ONNX sessions themselves open fine — this is not the ASR/QDQ bug.)
- **The route is gated, not lazy-loaded.** MusicGen is **571 MB** quantized (**~1.05 GB** at fp16):
  a text encoder, a decoder and an EnCodec vocoder. Auto-downloading that on navigation would be
  hostile, so `useTts` is only mounted after an explicit opt-in that states the size and the speed
  caveat up front. An E2E spec asserts **zero** Hugging Face requests before the click.
- **Warm-up needed a token budget.** The generic warm-up called `synth(text)` with no options, so
  MusicGen fell through to its 256-token default and spent **37 s** generating five seconds of music
  nobody hears. `WARMUP_TOKENS = 16` cut it to ~3 s. The speech engines ignore the field.

Speed measured on WASM (no GPU): ~50 audio tokens per second of compute, i.e. roughly 6× slower
than real time. Fine for the 1–15 s toy the route offers; the slider caps at 15 s for that reason.

### Audio-to-Audio (`03`) — not built, and here is why ❌

The plan proposed **DeepFilterNet** speech enhancement via `onnxruntime-web` on framed audio. The
research says that is a much bigger job than "add a dependency":

- The available exports (`soniqo/DeepFilterNet3-ONNX` and friends) are, in the author's own words,
  **"the neural graph only"**. The model takes `feat_erb [1,1,T,32]` and `feat_spec [1,2,T,96]` and
  returns an ERB mask plus complex deep-filter coefficients — **not** audio in, audio out.
- Making it useful means writing, in TypeScript, the whole surrounding DSP contract: a 960-point
  real FFT with a 480-sample hop and a Vorbis window at 48 kHz, a 32-band ERB filterbank, streaming
  exponential feature normalisation with specified state initialisation (−60 dB → −90 dB), an
  order-5 complex deep filter with 2 frames of lookahead applied to an immutable copy of the noisy
  spectrum, and ISTFT overlap-add synthesis. The browser ships no real-FFT primitive, so that comes
  too.
- None of it is verifiable from inside this repo without a reference implementation to diff against,
  and "subtly wrong DSP" doesn't fail loudly — it produces plausible audio with artefacts.

That is a self-contained project with its own plan and its own testing story, not a phase of this
one. **Recommendation:** keep Audio-to-Audio server-side (where `torchaudio`/`speechbrain` already
do this correctly) and leave the taxonomy entry pointing at the `/tasks/$slug` placeholder — which a
test now asserts, so nobody wires it up by accident. Demucs stem separation was already server-side
in the original plan for the same class of reason.

### Audio-Text-to-Text (multimodal) — out of scope, unchanged

Audio LLMs (Qwen2-Audio, …) are multi-billion-parameter with no browser runtime. Server API only.

## Phase 6 — Memory, performance, and docs polish ✅

*Done: warm-up inference, a hardened one-model-live path, the size-before-load guardrail, and the docs
note — all three engines and all three routes. Unit-tested and manually verified in a real browser
(see Testing).*

- ✅ **Warm-up inference** on model load. Each engine (`asrEngine` / `pipelineEngine` / `ttsEngine`) runs
  one throwaway inference before posting `ready` — 0.25 s of silence for the audio tasks, `"Hi."` for TTS
  — so the first real request doesn't pay to compile the WebGPU shaders / JIT the WASM module. The
  generic pipeline worker picks warm-up args per task (`{ top_k: 1 }`, or a single candidate label for
  zero-shot, which can't be called without one). A `{ status: "warmup" }` progress drives a
  "Warming up the model…" line in `ModelStatus`; a warm-up that **throws is swallowed** — the model is
  still usable, and the first real run just pays the compile cost instead. Handlers take
  `{ warmup }` so tests can opt out.
- ✅ **One-model-live, hardened.** The engines already disposed the previous model; they now null the
  reference **first** and dispose through `disposeQuietly`, so a backend whose teardown throws (a lost
  device, say) can't abort the load or leave the stale model live. Covered by a
  "loads the next model even when disposing the previous one fails" test in all three engine suites.
- ✅ **Size-before-load guardrail** (`audio/size.ts`). Every catalogue entry now carries a `params` count
  in millions; `sizeEstimate()` derives the download for both backends (fp16 = 2 bytes/param,
  q8 = 1) and flags anything past `LARGE_MODEL_BYTES` (300 MB). The new shared
  `components/audio/ModelPicker.tsx` — which replaces the button row triplicated across the three audio
  routes — shows the selected model's params + both estimates, and an amber warning on a large model.
  CLAP (153M ⇒ ≈292 MB fp16) is the closest to the line today.
- ✅ **Docs.** [`adding-a-model.md`](../../guides/adding-a-model.md) gained a
  **§7 "Adding a pretrained (Transformers.js) task"** — catalogue entry (incl. `params`), when to reuse
  the generic pipeline worker vs. write a dedicated one, the three behaviours every engine owes
  (one-model-live, warm-up, never block the main thread), hook + route, the `REAL_ROUTES` entry, and how
  to verify. The intro now splits the guide into the custom-kernel path (§1–6) and the pretrained path (§7).
- n/a **`api-contracts.md`** — unchanged, as planned: Phases 1–4 and 6 add no server endpoint. Revisit in Phase 5.
- ⏳ Move this plan to `docs/plans/completed/` when `Status` reaches `Complete` (after Phase 5).

---

## Feasibility summary

| Task (notebook) | In-browser? | Recommended model | Best backend | If not | Phase |
|-----------------|-------------|-------------------|--------------|--------|-------|
| **ASR** (`02`) | Yes — excellent | Whisper-base / Moonshine-tiny (ONNX) | WebGPU (WASM ok) | — | 2 |
| **Audio classification** (`04`) | Yes — full | AST / wav2vec2 / CLAP (ONNX) | WASM or WebGPU | — | 3 |
| **TTS** (`00`) | Yes | Kokoro-82M (`kokoro-js`), MMS-VITS, SpeechT5 | WebGPU (Kokoro) / WASM | Bark → server | 4 |
| **Text-to-Audio** (`01`) | ✅ Yes, gated | MusicGen-small (1-15 s clips) | WebGPU / WASM | AudioLDM / Stable Audio → server | 5 |
| **Audio-to-Audio** (`03`) | ❌ Not viable in scope | — (DFN export is graph-only; needs a full STFT/ERB stack) | — | DeepFilterNet / Demucs / VC → server | 5 |
| **Audio-Text-to-Text** (multimodal) | No | — | — | server API | out |

Rule of thumb: **discriminative + small** (ASR, classification, small TTS) runs great client-side;
**generative + large / diffusion** (music, separation, audio LLMs) belongs on a server, browser as UI.

---

## Testing

Neither WebGPU nor the Web Audio API exists in the test env (happy-dom), so unit tests **mock** them;
real inference is verified manually. Automated coverage for Phases 1–2 is **done and green** (checkmarks
below); later phases extend the same patterns.

- ✅ **`backend.test.ts`** — `pickBackend()` returns `"webgpu"` when a mocked `navigator.gpu.requestAdapter`
  resolves an adapter, `"wasm"` when it's absent, returns null, or throws; `loadOpts` returns the right
  `device`/`dtype` pair.
- ✅ **`io.test.ts`** — `toWavBlob` writes a valid 44-byte header + clamps/quantizes to full-scale int16;
  `decodeToMono` resamples through a mocked `OfflineAudioContext` at the target rate (and guards ≥1 frame);
  `play` copies samples into a mono buffer and starts; `recordMic` captures a chunk via a fake
  `MediaRecorder`, decodes it, and releases the mic tracks.
- ✅ **`asrEngine.test.ts`** — the message handler with a fake pipeline factory (no model download):
  load → progress → ready, transcribe (default timestamp/chunk args), array-result unwrap, "no model
  loaded" run error, id-less load error, and dispose-of-previous on reload.
- ✅ **`useAsr.test.ts`** — a fake `Worker`: posts `load` on mount, flips `ready` + records backend,
  id-correlated `transcribe` resolves/rejects and transfers the audio buffer, load error sets `status:
  "error"`, and the worker is `terminate()`d on unmount.
- ✅ **Route + taxonomy** — `asr.test.tsx`: `/asr` renders the heading + both model buttons, disables
  controls until `ready`, shows the backend badge, renders a timestamped transcript, and surfaces a load
  error. `taskTaxonomy.test.ts` asserts `automatic-speech-recognition → /asr` and
  `categoryForPath("/asr") === "Audio"`.
- ✅ **Lint / build** — `eslint` 0 errors; `vite build` clean, emitting `asr.worker-*.js` +
  `ort-wasm-*.wasm` (confirms the worker + ONNX Runtime WASM bundle correctly).
- ✅ **Manual browser verification — done** (2026-09-01, headless Chromium against the HTTPS dev
  server). This machine has no GPU (`requestAdapter()` returns null), so it exercised the **WASM
  fallback end to end** — which is the path that turned out to be broken. Results:
  - `/asr` — Whisper-base downloads, warms up, reports **"Model ready · running on WASM"**, and the
    JFK sample transcribes **exactly** to the reference: *"And so my fellow Americans, ask not what
    your country can do for you, ask what you can do for your country."*
  - `/audio-classification` — AST loads and reaches ready; CLAP zero-shot scores prompts sensibly
    (`speech` 0.68 vs `music` 0.32 on silence-padded input).
  - `/text-to-speech` — Kokoro-82M loads and synthesises (5.2 s @ 24 kHz), played back in-page.
  - Graceful degradation confirmed: no GPU adapter → `pickBackend()` picks WASM, the UI says so, and
    nothing crashes.
  - Phase 6 features confirmed live: the **"Warming up the model…"** state appears between download
    and ready, and the size note + amber **large-model warning** render on Whisper-base and CLAP.
  - ⏳ **Still unverified:** the **WebGPU** path (no GPU on this machine) and the **mic** capture loop
    (no audio input in headless). Both need a human on real hardware.
- ✅ **Regression specs (2026-09-02).** The manual checks are now permanent Playwright specs:
  - `e2e/specs/audio.spec.ts` — 10 specs in the **default** run (~5 s). Every Hugging Face request is
    aborted, so they assert the route shell, the size-guardrail warning firing on the heavy model and
    staying quiet on the light one, and that a blocked download **fails visibly** instead of spinning.
  - `e2e/specs/audio-models.spec.ts` — 5 **`@slow`** specs (`just fe-e2e-slow`, ~2.5 min) that load
    every catalogue model for real and assert Whisper transcribes JFK to the right words. These are
    the only tests that open a real ONNX Runtime session — the layer both bugs lived in.
  - `just fe-e2e-models` runs the id check alone in ~3 s.
  - A `@slow` spec asserts the transcript renders as **timestamped `<li>` segments** matching
    `m:ss`, scoped to the Transcript card. The plan once claimed timestamps while the route rendered
    plain text; this is the guard against that happening again silently.

---

## What manual verification found

Two bugs that **every unit test missed**, because both live in the layer the tests mock away
(a real ONNX Runtime session, and real Hugging Face repos). Both are fixed.

1. **No ASR model could load on the WASM fallback — the universal path.** ONNX Runtime failed at
   session creation with
   `qdq_actions.cc:137 TransposeDQWeightsForMatMulNBits Missing required scale:
   model.decoder.embed_tokens.weight_merged_0_scale`.
   Isolated by probing the Hub directly from the page: it reproduces on **every** ASR repo tried
   (`onnx-community/whisper-base`, `onnx-community/whisper-tiny.en`, `Xenova/whisper-tiny.en`,
   `moonshine-tiny`) and at every dtype whose **decoder** is quantized — so it is a bug in the ORT
   build bundled with `@huggingface/transformers` 4.2.0 (already the latest release), not a bad
   export. The **encoder** quantizes fine.
   **Fix:** `asrLoadOpts()` in `audio/backend.ts` — WASM now loads
   `{ encoder_model: "q8", decoder_model_merged: "fp32" }`; WebGPU is unchanged at fp16. Cost is
   size, not correctness: Whisper-base on WASM is ~221 MB instead of ~71 MB. Worth it against a
   fallback that could not load a model at all. Revisit when the bundled ORT updates.

2. **Two of the three classification models did not exist.**
   `onnx-community/ast-finetuned-audioset-10-10-0.4593` and
   `onnx-community/wav2vec2-base-superb-ks` both return **401** from the Hub — the load failed with
   "Unauthorized access to file". The real repos are under **`Xenova/`**. Verified all five audio
   repos against the Hub API before changing. **Fix:** corrected ids in `audio/classification.ts`.

A third finding shaped Phase 6 itself: the params-based size estimate was **badly wrong** for ASR
once the decoder went fp32 (it claimed 71 MB for a 221 MB download). Catalogue entries now carry
optional **measured** `bytes`, `sizeEstimate` prefers them, and `large` keys off the **bigger** of
the two backends rather than fp16 — otherwise the ASR warning would never fire. The threshold moved
from 300 MB to **200 MB**: at 300 MB nothing we ship would have tripped it, making the guardrail
dead code. It now fires on exactly the two heavy models (Whisper-base, CLAP) and stays quiet on the
rest — asserted by a test, so a future catalogue change can't silently kill it.

**Method note:** the checks were driven with Playwright against the real HTTPS dev server, since
`navigator.gpu` and the Web Audio API need a real browser. They are now **permanent specs** rather
than throwaway scripts — see the Testing section and
[`e2e-testing.md`](../../guides/e2e-testing.md). The model-id check reproduces bug 2 in 2.6 s and
names the offending repo, verified by temporarily reverting the id.

---

## Out of scope / follow-ups

- ~~Streaming / live-caption ASR~~ — **done** (see Phase 2): `useLiveAsr` re-transcribes a rolling 30 s
  window per `MediaRecorder` chunk. Remaining refinements: true incremental decoding (avoid re-decoding the
  whole take each tick — bounded today by the 30 s transcribe window, but decode cost still grows with take
  length), word-level streaming, and an `AudioWorklet` capture path to skip the decode round-trip entirely.
- Server-side inference endpoints for the generative/large tasks (MusicGen long-form, Demucs, audio LLMs) and
  the accompanying `api-contracts.md` changes.
- Aligning backend `ModelCard.pipeline_tag` values with these audio tasks so runs are recorded in the registry.
- A shared `<AudioModelRunner>` shell (upload / mic / progress / dispose) factored out once 2–3 audio routes exist.
  *(Partly done in Phase 6: `components/audio/ModelPicker.tsx` now factors out the picker + size
  guardrail that was triplicated across the three routes; the transport controls are still per-route.)*
- ~~Fold the audio browser checks into the Playwright suite~~ — **done** (2026-09-02):
  `e2e/specs/audio.spec.ts` (10 fast specs, downloads blocked) runs in the default suite, and
  `e2e/specs/audio-models.spec.ts` (5 `@slow` specs) loads every catalogue model for real via
  `just fe-e2e-slow`. `just fe-e2e-models` checks the ids alone in ~3 s.
- **Re-test the ASR quantized decoder** when `@huggingface/transformers` ships a newer ONNX Runtime;
  if fixed, drop the fp32 decoder override in `asrLoadOpts` and reclaim ~150 MB on the WASM path.

---

## Appendix — reference snippets

Distilled from the original research doc; wrap every `pipeline` call in the Phase-1 worker for production.

```ts
// src/audio/backend.ts
export type Backend = "webgpu" | "wasm";

export async function pickBackend(): Promise<Backend> {
  if ("gpu" in navigator) {
    try {
      const adapter = await (navigator as { gpu: GPU }).gpu.requestAdapter();
      if (adapter) return "webgpu";
    } catch {
      /* fall through to wasm */
    }
  }
  return "wasm";
}

export function loadOpts(backend: Backend) {
  return backend === "webgpu"
    ? { device: "webgpu" as const, dtype: "fp16" as const }
    : { device: "wasm" as const, dtype: "q8" as const };
}
```

```ts
// src/audio/io.ts — decode any audio to mono Float32 @ targetRate (OfflineAudioContext resamples for free)
export async function decodeToMono(data: ArrayBuffer, targetRate = 16000): Promise<Float32Array> {
  const tmp = new AudioContext();
  const decoded = await tmp.decodeAudioData(data.slice(0));
  await tmp.close();
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * targetRate), targetRate);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0);
}
```

```ts
// src/audio/asr.worker.ts
import { pipeline } from "@huggingface/transformers";

let task: Awaited<ReturnType<typeof pipeline>> | null = null;
self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;
  if (type === "load") {
    task = await pipeline("automatic-speech-recognition", payload.model, {
      ...payload.opts,
      progress_callback: (p: unknown) => self.postMessage({ type: "progress", p }),
    });
    self.postMessage({ type: "ready" });
  } else if (type === "run" && task) {
    const out = await task(payload.audio, payload.args);
    self.postMessage({ type: "result", out });
  }
};
```

```tsx
// src/hooks/useAsr.ts
import { useEffect, useRef, useState } from "react";

export function useAsr(model = "onnx-community/whisper-base") {
  const workerRef = useRef<Worker | null>(null);
  const [ready, setReady] = useState(false);
  const [text, setText] = useState("");

  useEffect(() => {
    const w = new Worker(new URL("../audio/asr.worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (e) => {
      if (e.data.type === "ready") setReady(true);
      if (e.data.type === "result") setText(e.data.out.text);
    };
    w.postMessage({ type: "load", payload: { model, opts: {} } }); // backend/dtype picked in-worker
    workerRef.current = w;
    return () => w.terminate();
  }, [model]);

  const transcribe = (audio: Float32Array) =>
    workerRef.current?.postMessage({ type: "run", payload: { audio, args: { chunk_length_s: 30 } } });

  return { ready, text, transcribe };
}
```
