# Audio Models in the Browser (Transformers.js / ONNX Runtime Web)

**Status:** In Progress (Phases 1–4 complete — real-time ASR, audio classification, and TTS ship end-to-end; Phases 5–6 pending)

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
load-progress, timestamped transcript). `REAL_ROUTES["automatic-speech-recognition"] = "/asr"` wired.
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
| AST AudioSet tagging | `onnx-community/ast-finetuned-audioset-10-10-0.4593` | `audio-classification` |
| wav2vec2 keyword spotting | `onnx-community/wav2vec2-base-superb-ks` | `audio-classification` |
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

## Phase 5 — Partial / server-boundary tasks (Text-to-Audio, Audio-to-Audio)

These are where in-browser stops being the right call — plan them as **degraded/optional** or as a
thin client over a server API. Adds a **direct** `onnxruntime-web` dependency for the custom-model path.

- **Text-to-Audio** (`01`): `Xenova/musicgen-small` via `text-to-audio` *can* run in-browser but is
  autoregressive (~50 tok/s of audio) — fine for a 5–10 s toy, painful longer. AudioLDM / Stable Audio
  are `diffusers` latent-diffusion with **no** Transformers.js path → server-side only. Ship the MusicGen
  toy behind an explicit "experimental / slow" gate; document the server route as the production path.
- **Audio-to-Audio** (`03`): no `audio-to-audio` pipeline exists in `transformers` *or* Transformers.js.
  Realistic in-browser option is **DeepFilterNet** speech enhancement via `onnxruntime-web/webgpu` on
  framed audio (you own the STFT framing / overlap-add that `torchaudio`/`speechbrain` do in Python).
  Demucs stem separation is large + non-causal → prefer server-side. Custom-ONNX session skeleton:
  `ort.InferenceSession.create(url, { executionProviders: ["webgpu", "wasm"] })`.
- **Audio-Text-to-Text** (multimodal): audio LLMs (Qwen2-Audio, …) are multi-billion-param with no
  browser runtime → **server API only**. Out of scope for in-browser work.

## Phase 6 — Memory, performance, and docs polish

- **Warm-up inference** on model load (compiles WebGPU shaders / JITs WASM) so the first real request is fast.
- Enforce **one-model-live**: dispose + null before loading another; verify no leak across model switches.
- **Size-before-load** guardrail: warn on models past a few hundred M params (fp16 ≈ params × 2 bytes;
  Whisper-base ≈ 150 MB).
- Add a short `docs/guides/` note (or extend [`adding-a-model.md`](../../guides/adding-a-model.md)) on
  adding a Transformers.js audio task: worker factory + hook + route + `REAL_ROUTES` entry.
- Update [`docs/standards/api-contracts.md`](../../standards/api-contracts.md) **only if** Phase 5 adds a
  server inference endpoint (none for Phases 1–4 — all client-side).
- Move this plan to `docs/plans/completed/` when `Status` reaches `Complete`.

---

## Feasibility summary

| Task (notebook) | In-browser? | Recommended model | Best backend | If not | Phase |
|-----------------|-------------|-------------------|--------------|--------|-------|
| **ASR** (`02`) | Yes — excellent | Whisper-base / Moonshine-tiny (ONNX) | WebGPU (WASM ok) | — | 2 |
| **Audio classification** (`04`) | Yes — full | AST / wav2vec2 / CLAP (ONNX) | WASM or WebGPU | — | 3 |
| **TTS** (`00`) | Yes | Kokoro-82M (`kokoro-js`), MMS-VITS, SpeechT5 | WebGPU (Kokoro) / WASM | Bark → server | 4 |
| **Text-to-Audio** (`01`) | Partial | MusicGen-small (short demos only) | WebGPU | AudioLDM / Stable Audio → server | 5 |
| **Audio-to-Audio** (`03`) | Partial | DeepFilterNet (custom ONNX) | WebGPU / WASM | Demucs / VC → server | 5 |
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
- ⏳ **Manual (still pending — the real check):** `just fe-dev` over HTTPS — load Whisper-base on WebGPU
  and on WASM (disable WebGPU to force fallback), transcribe an uploaded clip and a 5 s mic recording,
  confirm timestamps, model switch frees memory (no growth across reloads), and the page degrades
  gracefully with WebGPU off.

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
