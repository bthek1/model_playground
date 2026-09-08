# How to Add a Task Page

> Taking one task out of the sidebar taxonomy and turning it into a working
> **Model Playground** route that runs the model in the browser, on the user's
> own GPU or CPU. This file is the procedure; the
> `*_Models_in_React_WebGPU_and_CPU.md` files beside it are the per-category
> model lookup tables it sends you to.

Every task in [`taskTaxonomy.ts`](../../../frontend/src/components/layout/taskTaxonomy.ts)
answers "I have this input and want that output, what should I use". Most of
them currently render the `/tasks/$slug` placeholder. Turning one into a real
page is mechanical enough to write down, which is what this file does.

The model research behind these guides comes from a companion collection of
Python notebooks, which is a separate project — the checkpoints, the input
contracts and the head-to-head comparisons are quoted here so you never have to
open it. **Nothing in this repo depends on it.** Where a section says "the
reference notebook", read it as "the research this recommendation came from",
not as a file you can open.

---

## 0. First decide whether it belongs in the browser at all

This is step zero because it is the step that gets skipped, and skipping it
costs a week.

Model Playground downloads weights into the tab and runs every matrix multiply
locally. Django is a **registry** — it stores catalogue metadata and run
records, and it has no GPU role at all. So a task that will not fit in a tab
does not become a page. Three questions, in order:

1. **Does a browser-runnable checkpoint exist?** In practice that means an ONNX
   export under `onnx-community/*`, `Xenova/*`, or the model's own repo. If the
   only weights are PyTorch, somebody has to export them, and that is a project
   of its own rather than a page.
2. **Is it under roughly 1 GB in the quantization you will ship?** A tab is a
   far tighter budget than the 12 GB card this research was done on. Anything
   past about 2B parameters at `q4` is a slow first load and a real risk of
   killing the tab. `LARGE_MODEL_BYTES` in `audio/size.ts` is 200 MB — past that
   the warning is mandatory, not optional.
3. **Is it discriminative, or at worst a small generator?** Classification,
   detection, segmentation, embedding, ASR and small TTS all run well. Latent
   diffusion, video generation and 7B multimodal LLMs do not, and no amount of
   care makes them.

If the answer to any of these is no, the task still gets a row in its
category guide's feasibility table saying so and why, and it keeps its
`/tasks/$slug` placeholder. **A documented "server-side, and here is the
reason" is a finished piece of work.** It stops the next person spending three
days rediscovering it.

The one exception is the raw WebGPU path
([`frontend/src/webgpu/`](../../../frontend/src/webgpu/)). A task with no ONNX
export can still become a page if the maths is small enough to write as WGSL by
hand, which is how `/training` and `/tensor` exist. Tabular models, graph
convolutions and RL policies all fall in that bucket. See the
[`Tabular`](Tabular_Models_in_React_WebGPU_and_CPU.md),
[`Other`](Graph_Models_in_React_WebGPU_and_CPU.md) and
[`Reinforcement_Learning`](Reinforcement_Learning_in_React_WebGPU_and_CPU.md)
guides. **Keep the two runtimes apart**: `src/webgpu/` is hand-written WGSL with
no ML framework in it, and Transformers.js / ONNX Runtime Web live under
`src/audio/` and its future siblings. They never mix in one directory.

---

## 1. Pin down four facts

Everything downstream is derived from these four, and getting them wrong is the
only way this process actually fails.

| Fact | Where to find it | Why the page needs it |
|---|---|---|
| **Task string** | the category guide's table, then the model card on the Hub | it is the Transformers.js task verbatim, and it travels in the worker's `load` message |
| **Checkpoint** | the category guide's table (browser id, not the PyTorch one) | becomes the catalogue entry's `id`, and `ModelCard.slug` on the backend |
| **Input contract** | `preprocessor_config.json` in the model's own repo | sample rate, image size, colour order, normalisation. This is where silent wrongness lives |
| **Output shape** | the pipeline's documented return type | decides what `OutputPanel` renders, and it is the only part of the page a user actually looks at |

Two of these deserve suspicion.

**The input contract is where bugs hide.** Audio gets resampled to 16 kHz (48 kHz
for enhancement), images get letterboxed to a fixed size, and everything is
normalised with constants you never see. Transformers.js ships the model's own
processor, so use it rather than reimplementing: `AutoProcessor.from_pretrained(id)`
reads `preprocessor_config.json` from the same repo. Hand-rolling preprocessing is
exactly how the two DSP bugs in `audio/enhance/` shipped past a green test suite —
and that route only hand-rolls it because DeepFilterNet3 publishes no processor at
all.

**The output shape decides the page's whole layout.** A label list is a bar
chart. A set of boxes is a canvas overlay on the input image. A depth map is a
colourised canvas. A caption is a paragraph. Sketch this before writing any
worker code, because a page whose OUTPUT slot is an afterthought reads as
unfinished no matter how good the inference is. The drawing primitives in
[`components/viz/`](../../../frontend/src/components/viz/) and the standard in
[`../../standards/model-visualization.md`](../../standards/model-visualization.md)
are where to start, not a fresh canvas helper.

---

## 2. Resolve a browser checkpoint, and verify it

The upstream checkpoint is almost never the one you ship. `microsoft/resnet-50`
is PyTorch; the browser wants `Xenova/resnet-50`. The category guide beside this
file already lists the mapping for every task in it.

Whatever you pick, **verify it exists before writing a line of UI**. Two
catalogue entries in Model Playground once pointed at repos that do not exist,
and the failure surfaced as a load error in the browser rather than as anything
useful. The check is one HTTP call:

```bash
# does the repo exist, and does it carry ONNX weights?
curl -s "https://huggingface.co/api/models/onnx-community/depth-anything-v2-small" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); \
      print([s['rfilename'] for s in d['siblings'] if s['rfilename'].endswith('.onnx')])"
```

The same check is a recipe here, and it covers every id in the catalogue at
once:

```bash
just fe-e2e-models   # seconds, no downloads, fails loudly on a dead id
```

It is the `@slow model catalogue` block in
[`e2e/specs/audio-models.spec.ts`](../../../frontend/e2e/specs/audio-models.spec.ts),
which imports the catalogue modules directly and HEADs every `id` against the Hub
API. **Add your new catalogue module to that import list in the same commit as
the page.** A catalogue nobody imports there is a catalogue nobody is checking —
which is how two `onnx-community/*` repos that return 401 reached the app.

### Choosing the precision

Two backends, two answers, and `torch.float16` maps onto the first one. The
defaults are in [`audio/backend.ts`](../../../frontend/src/audio/backend.ts);
`loadOpts(backend)` already returns the first and third rows:

| Backend | `dtype` | Notes |
|---|---|---|
| WebGPU | `fp16` | the direct analogue of PyTorch half precision |
| WebGPU, large generative models | `q4f16` | 4-bit weights, fp16 compute. The only way a 0.6B LLM is pleasant in a tab |
| WASM (CPU) | `q8` | 2 to 4 times smaller download and RAM. `q4` when even that is too big |

Quote the **download size in the chosen dtype**, not the parameter count, in the
model card. A user waiting on 300 MB does not care how many parameters that is.
`sizeEstimate()` in [`audio/size.ts`](../../../frontend/src/audio/size.ts) does
the arithmetic and quotes both backends, because the picker runs before a backend
is resolved.

There is one standing exception to the table, and it is worth knowing before you
copy it: **ASR keeps its decoder at `fp32` on WASM** (`asrLoadOpts`), because the
quantized Whisper/Moonshine decoders cannot open a session on the bundled ONNX
Runtime. If a q8 export refuses to load with a `Missing required scale` error,
that is the same bug, not yours.

---

## 3. Pick the worker: one per modality, never one per task

The rule that keeps the frontend from sprouting a worker per route. Today there
are five: [`audio/asr.worker.ts`](../../../frontend/src/audio/asr.worker.ts),
[`audio/pipeline.worker.ts`](../../../frontend/src/audio/pipeline.worker.ts),
[`audio/tts.worker.ts`](../../../frontend/src/audio/tts.worker.ts),
[`audio/enhance/enhance.worker.ts`](../../../frontend/src/audio/enhance/enhance.worker.ts)
and the raw-WGSL [`webgpu/worker.ts`](../../../frontend/src/webgpu/worker.ts). A
*sixth* is only justified when a modality genuinely needs different machinery.

Decide with this test: **does the new task load through the same runtime, with
the same input decode path, as an existing worker?** If yes, the task string
travels in the `load` message and no new file is created.

| Your task | Worker | New file? |
|---|---|---|
| A generic Transformers.js pipeline over the same modality as an existing worker | that worker | no — extend its task union (`audio/pipelineTypes.ts` is the model) |
| A generic pipeline over a *new* modality (text, images) | a new `src/<modality>/pipeline.worker.ts` | yes, once per modality |
| Anything that owns a capture loop (mic, camera) | its own modality worker | yes, capture loops are stateful |
| Anything on bare `onnxruntime-web` with hand-written pre/post | its own worker | yes, `audio/enhance/` is the precedent |
| Hand-written WGSL, no ML framework | `webgpu/worker.ts` | no, add a shader and a pipeline |

The reason is not tidiness. Every worker duplicates the load/progress/cancel
protocol, and every duplicate is a place for it to drift. Five protocols are
maintainable; fifteen are not.

---

## 4. Write the engine, not the worker

The worker is a `postMessage` shell. The logic goes in a plain module that
imports nothing from `self`, because that module is the only part a unit test
can reach.

The precedent to copy is
[`audio/pipelineEngine.ts`](../../../frontend/src/audio/pipelineEngine.ts): it
exports a `createPipelineHandler(post, factory)` that closes over the live model
and returns the async message handler, so a unit test drives it with a fake
factory and no download at all. Mirror that shape.

```ts
// src/vision/visionEngine.ts
import { loadOpts, pickBackend } from "@/audio/backend";   // the probe is shared

export function createVisionHandler(post: Post, factory: PipelineFactory) {
  let model: CallablePipeline | null = null;

  return async function handle(msg: VisionRequest) {
    if (msg.type === "load") {
      // 1. one model live at a time: null the reference BEFORE disposing, so a
      //    failed teardown can never leave a stale model live.
      const previous = model;
      model = null;
      await disposeQuietly(previous);

      // 2. the backend is resolved here, once, and reported with `ready`.
      const backend = msg.backend ?? (await pickBackend());
      const opts = loadOpts(backend);
      model = await factory(msg.task, msg.model, {
        ...opts,
        progress_callback: (p) => post({ type: "progress", progress: p }),
      });

      // 3. warm up: the first inference compiles WebGPU shaders and JITs the
      //    WASM. Never let this fail the load — a warm-up error is a slow page,
      //    not a broken one.
      post({ type: "progress", progress: { status: "warmup" } });
      try {
        await model(blankImage(64, 64));
      } catch { /* warm-up is best effort */ }

      post({ type: "ready", model: msg.model, backend: opts.device });
    }
    // "run" branch: id-correlated, see §5.
  };
}
```

**Every engine owes three behaviours**, and a code review that does not check
all three is not a review:

1. **One model live at a time.** Null the reference first, then dispose. There
   is no `torch.cuda.empty_cache()` in a browser, so a leaked session is a
   leaked GPU context and the next load OOMs the tab.
2. **Warm up on load, and never fail the load because of it.** A cold first
   inference is several seconds of shader compilation, which the user reads as
   a broken button. Post `{ status: "warmup" }` so the LOAD slot can label the
   phase — `model/progress.ts` renders it as its own indeterminate step rather
   than as the tail of the download.
3. **Return structured-cloneable data.** Post `ArrayBuffer`s and transfer them;
   never try to post a live session, a canvas, or a class instance.

---

## 5. Wrap it in the worker

Thin, and identical in shape to the other workers: it wires the engine to
`self` and supplies the real pipeline factory. Nothing else.

```ts
// src/vision/vision.worker.ts
import { pipeline } from "@huggingface/transformers";

import { createVisionHandler } from "./visionEngine";
import type { VisionRequest, VisionResponse } from "./visionTypes";

// Avoid `/// <reference lib="webworker" />` — it collides with the app's DOM
// lib. Narrow `self` to what is actually used, as `pipeline.worker.ts` does.
const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<VisionRequest>) => void) | null;
  postMessage: (m: VisionResponse, transfer?: Transferable[]) => void;
};

const handle = createVisionHandler(
  (m, transfer) => ctx.postMessage(m, transfer),
  (task, model, opts) => pipeline(task, model, opts),
);

ctx.onmessage = (e) => void handle(e.data);
```

**The message envelope is not yours to invent.** It is
[`ModelRequest`/`ModelResponse` in `model/types.ts`](../../../frontend/src/model/types.ts),
and only the load/run *payloads* are task-specific:

```ts
export type VisionRequest = ModelRequest<
  { task: VisionTask; model: string; opts?: LoadOpts },   // load
  { input: RawImageData; args?: unknown[] }               // run
>;
export type VisionResponse = ModelResponse<VisionResult>;
//  → { type: "progress"; progress }
//  | { type: "ready"; model; backend }
//  | { type: "result"; id; result }
//  | { type: "error"; id?; error }
```

`id` on an error is the discriminator between the two state machines: `id != null`
is a request failure and `status` stays `ready`; `id == null` is a load failure.
Getting that backwards means a single bad input drops the page back to an error
screen with the model still loaded.

Note what the worker does *not* do: it does not aggregate progress. It forwards
the raw per-file callback and lets `model/progress.ts` on the main thread turn
four to eight file callbacks into one monotonic percentage. A per-file bar that
restarts at zero eight times is worse than no bar.

---

## 6. The hook is a wrapper, not a place for logic

The load and run state machines already exist, exactly once, in
`src/model/useModelWorker.ts`. A task hook picks the worker and names the
messages, and that is all it does.

```ts
// src/hooks/useDepth.ts
import { createVisionWorker } from "@/vision/visionClient";
import { useModelWorker } from "@/model/useModelWorker";

export function useDepth(model: string, autoLoad = false) {
  const worker = useModelWorker<DepthResult>({
    createWorker: createVisionWorker,       // isolated so tests can mock it
    key: `depth-estimation:${model}`,       // changing it tears the worker down
    loadMessage: { task: "depth-estimation", model },
    autoLoad,                               // false for anything that downloads
    notReadyMessage: "Vision worker not ready",
  });

  const run = useCallback(
    (input: RawImageData) => worker.run({ input }, [input.data.buffer]),
    [worker.run],
  );
  return { ...worker, run };
}
```

`createWorker` lives in its own tiny module (`visionClient.ts`, mirroring
[`audio/pipelineClient.ts`](../../../frontend/src/audio/pipelineClient.ts))
purely so the hook's tests can mock worker creation — `import.meta.url` and
`new Worker` do not resolve under happy-dom.

Every task hook returns the same contract, so every page can be written the same
way and every test can assert the same things:

- **Load**: `idle -> loading -> ready | error`, with `progress` as a self-loop,
  `retry(overrides?)` out of `error`, and `cancel()` back to `idle`.
- **Run**: id-correlated requests, where `running` is derived from an in-flight
  **count**, never a boolean. A boolean lies the moment two requests overlap,
  and it lies in the direction of a stuck spinner.

The hook returns the [§3 contract](../../standards/model-page-pattern.md#3-the-hook-contract)
**verbatim**. [`useEnhance`](../../../frontend/src/hooks/useEnhance.ts) is the
reference implementation; the older audio hooks still expose a task-named alias
for `run` (`transcribe`, `synthesize`, `classify`) and are being migrated route
by route. **Do not add a new alias** — a reviewer who has read one hook should
have read them all, and a renamed `run` is exactly what breaks that.

If you find yourself adding state to a task hook, the state almost certainly
belongs in `useModelWorker` where the other five tasks can share it.

---

## 7. Build the route out of the four slots

The page is `ModelPage` plus the four slot components. Full specification in
`docs/standards/model-page-pattern.md`; the short version is that
**SELECT -> LOAD -> RUN -> OUTPUT** is the only shape a task page takes.

The four slots are **named props**, not children. A route therefore cannot
reorder them, cannot drop OUTPUT when it has nothing to show, and cannot quietly
grow a fifth stage.

```tsx
// src/routes/depth-estimation.tsx
function DepthPage() {
  const session = useModelSelection({
    routeKey: "depth-estimation",           // namespaces the persisted selection
    models: DEPTH_MODELS,
    fallback: DEPTH_MODELS[0],
  });
  const task = useDepth(session.model.id, session.autoLoad);
  useCacheRefresh(session, task.ready);

  // A load error belongs in LOAD, a run error in OUTPUT. `status` splits them.
  const loadError = task.status === "error" ? task.error : null;
  const runError  = task.status === "error" ? null : task.error;

  return (
    <ModelPage
      icon={Mountain}
      title="Depth Estimation"
      description="Relative depth from a single image, entirely in your browser."
      select={
        <ModelPicker
          models={DEPTH_MODELS}
          value={session.model.id}
          onChange={session.setModel}
          disabled={task.loading || task.running}
          cached={session.cached}
          onEvict={(m) => void session.evict(m.id)}
        />
      }
      load={<ModelStatus {...task} error={loadError} />}
      run={<InputPanel … />}
      output={<OutputPanel … error={runError} />}
    />
  );
}
```

`useModelSelection` supplies `session.autoLoad`, which is the entire refresh
story: it is true **only** when a stored load intent meets a cache hit. Do not
pass a literal `true`.

Three rules fall out of the pattern, and all three are asserted by tests:

1. **Nothing downloads until the user asks.** `idle` is the default. The size
   estimate and the large-model warning are shown first, and quoted exactly
   once, by `ModelPicker` — `ModelStatus` must not repeat the number.
2. **A refresh restores decisions, not sessions.** A Worker cannot outlive a
   page load. [`store/models.ts`](../../../frontend/src/store/models.ts) persists
   the selected model and the *intent* to load it;
   [`model/useModelSelection.ts`](../../../frontend/src/model/useModelSelection.ts)
   resumes on mount only when that intent meets a cache hit from `model/cache.ts`.
   When the cache probe is uncertain it answers "not cached", because the safe
   direction is one extra click, never bandwidth spent unasked.
3. **Layout is horizontal, and `ModelPage` already does it.** SELECT and LOAD
   collapse into a setup rail of about 20rem; RUN and OUTPUT sit side by side as
   the workbench, so a result never lands below the fold. Placement is by
   CSS-grid *area*, which is what keeps **DOM order 1, 2, 3, 4 at every
   breakpoint** while the visual arrangement changes. Do not add a grid of your
   own inside a slot's column, and give any nested scroller `min-h-0 min-w-0` or
   a wide result will push the whole page sideways.

---

## 8. Register the route in the taxonomy and the registry

Three registrations, and all three are easy to forget.

**The frontend catalogue** is the one the page actually reads. One module per
task, beside the worker, shaped like
[`audio/classification.ts`](../../../frontend/src/audio/classification.ts):

```ts
export interface DepthModel {
  id: string;      // the Hub id, verified — this is what `just fe-e2e-models` checks
  label: string;   // what the picker shows
  hint: string;    // one line: what makes this model different from its neighbour
  params: number;  // millions. Drives the size estimate in `audio/size.ts`
  bytes?: MeasuredBytes;  // per-backend override, when the estimate would mislead
}
export const DEPTH_MODELS: DepthModel[] = [ /* … */ ];
export const DEFAULT_DEPTH_MODEL = DEPTH_MODELS[0].id;
```

Supply measured `bytes` whenever the params estimate is wrong by more than a
little. ASR does, because its fp32 WASM decoder makes the real download roughly
three times the estimate, and a size guardrail that under-quotes is worse than
none.

**The frontend taxonomy.** The sidebar is data. Categories and tasks live in
[`components/layout/taskTaxonomy.ts`](../../../frontend/src/components/layout/taskTaxonomy.ts)
and are mapped to real routes through `REAL_ROUTES`. An unmapped task falls
through to the generic `/tasks/$slug` placeholder, which is how the full Hugging
Face taxonomy is displayed without every task existing. Adding a page means
adding one `REAL_ROUTES` entry keyed by the slugified task label.

**The backend registry**, optionally. One `ModelCard` per checkpoint the page
offers — but note that `task` is a **fixed choice set**
(`llm` · `vision` · `embedding` · `audio` · `custom`, see
[`backend/apps/registry/models.py`](../../../backend/apps/registry/models.py)),
*not* the Transformers.js task string. The precise task belongs in `config`:

```jsonc
{
  "slug": "depth-anything-v2-small",
  "name": "Depth Anything V2 Small",
  "task": "vision",                        // ModelTask choice, not the pipeline task
  "description": "Relative depth from a single image.",
  "weights_url": "https://huggingface.co/onnx-community/depth-anything-v2-small",
  "config": {                              // free-form; the browser runtime reads it
    "pipeline": "depth-estimation",
    "dtype": { "webgpu": "fp16", "wasm": "q8" }
  },
  "size_bytes": 99000000,
  "license": "apache-2.0"
}
```

Adding an endpoint or changing a payload shape means updating
[`../../standards/api-contracts.md`](../../standards/api-contracts.md) in the
same commit.

The backend is a registry, not an inference server. It never sees the weights
and it never sees the input. `weights_url` points the browser at the model host
directly, which is the whole reason Django has no GPU role.

`InferenceRun` is the other half: the client reports latency, throughput and
the GPU adapter it used after a run. That is what turns a head-to-head benchmark
into a number measured on real user hardware instead of on one developer's
card.

---

## 9. Tests: the same contract on every page

Every task page asserts the identical list. Copy it from the nearest existing
spec rather than writing it fresh.

**Vitest, mocked network and mocked ONNX Runtime** (`src/__tests__/routes/`,
with [`asr.test.tsx`](../../../frontend/src/__tests__/routes/asr.test.tsx) as the
model):

- nothing downloads on mount: `autoLoad: false` **and** `load` was not called
- `load` and `retry` fire from the LOAD slot, `cancel` from the progress row
- the refresh pair: a stored selection is restored; a stored intent **plus** a
  cache hit resumes the load, and a stored intent **without** one does not
- all four slots render, and `output-empty` is present before any run
- run controls are disabled until `ready`
- an error lands in the slot that produced it, not in a global banner
- the engine's own maths, tested directly with no worker in the way — the engine
  modules exist precisely so this test needs no `new Worker`

One query trap: a band is a labelled `region`, so `getByLabelText(/text/i)`
matches both a band named "Text" and a field inside it. Query by role, or by the
field's own control.

**Playwright, mocked API (the default suite):** routing, the app shell, real
browser auth, the real WebGPU probe, and page layout.

**Playwright `@slow`, real downloads:** real weights, real ONNX sessions. This
is the only layer that catches a broken model, and it exists because unit tests
mock the network and therefore cannot.

Shared `data-testid`s across both runners: `slot-1` through `slot-4`,
`output-panel`, `output-empty`, `output-running`, `model-ready`,
`load-progress`, `load-cancel`, `model-size-note`, `model-size-warning`,
`error-note`. They are a contract — renaming one breaks tests in two suites at
once, so add to the table in the page-pattern standard rather than inventing an
ad-hoc id.

Three traps worth knowing before you hit them:

- **Layout is never a Vitest assertion.** happy-dom has no geometry.
  Arrangement is asserted only in
  [`e2e/specs/model-page.spec.ts`](../../../frontend/e2e/specs/model-page.spec.ts).
  Import `test`/`expect` from `e2e/fixtures/base`, never from `@playwright/test`.
- **Never route-match every `/api/` URL** in Playwright with a bare
  double-wildcard pattern. It also matches the `/src/api/*` module URLs and the
  app never boots.
- **Keep the `test.include` / `test.exclude` block in `vite.config.ts` pinned to
  `src/`**, or Vitest swallows the E2E specs.

---

## 10. Write the docs in the same commit

Docs travel with code. A new page touches these, and none of them is optional:

1. the category guide beside this one, so its status table stops saying "not
   built" and starts pointing at the route;
2. [`../../standards/model-page-pattern.md`](../../standards/model-page-pattern.md),
   if the page needed a genuinely new slot behaviour or a new testid;
3. [`../../standards/api-contracts.md`](../../standards/api-contracts.md), if
   you touched an endpoint;
4. [`README.md`](README.md) in this folder, whose category status table is where
   a newcomer looks first — move the task out of the "left" column.

Anything touching more than one file also gets a **plan first**, phased, with a
Testing section, in [`../in-progress/`](../in-progress/). Update its `Status` as
you go (`Draft → In Progress → Complete`) and `git mv` it to
[`../completed/`](../completed/) when it lands. Completed plans are kept as a
record, not deleted. The files in *this* folder are not plans — they are the
research a plan gets written from.

---

## Appendix A: the Python-to-browser phrasebook

Most model research is published as PyTorch. This is the translation table.

| Python | Browser (TypeScript) |
|---|---|
| `pipeline("task", id, torch_dtype=torch.float16, device_map="auto")` | `pipeline("task", id, { device: "webgpu", dtype: "fp16" })` |
| `AutoProcessor.from_pretrained(id)` | `AutoProcessor.from_pretrained(id)`, same JSON, same constants |
| `Image.open(path)` | `RawImage.fromURL(url)` or `RawImage.fromCanvas(canvas)` |
| `librosa.load(path, sr=16000)` | `decodeToMono(buf)` in [`audio/io.ts`](../../../frontend/src/audio/io.ts) |
| `cv2.VideoCapture(0)` | `navigator.mediaDevices.getUserMedia({ video: true })` into a `<video>` |
| a local `hf_cache` directory | Cache Storage, automatic, survives reloads, works offline; probed by `model/cache.ts` |
| `del model; free_memory()` | `await model.dispose()` after nulling the reference |
| `torch.cuda.empty_cache()` | no equivalent. Disposing is the whole mechanism |
| `malloc_trim(0)` | no equivalent, and no need: the tab's heap is not glibc's |
| `%%time` around a cell | `loadedInMs` from the hook, or `performance.now()` posted as an `InferenceRun` |
| "run the cells above first" guard | the `idle` state plus disabled run controls |

The last row is the deepest similarity between the two. A notebook guards its
live-capture cells because people scroll to the bottom and press shift-enter
without running Setup. A task page has the same problem and solves it
structurally: the run controls simply do not work until the load machine says
`ready`.

---

## Appendix B: the checklist

```
[ ] Triaged: browser-runnable checkpoint exists, fits in a tab, not a diffusion model
[ ] Four facts extracted: task string, checkpoint, input contract, output shape
[ ] Browser id verified against the Hub; catalogue imported by `just fe-e2e-models`
[ ] dtype chosen per backend, download size (not param count) quoted
[ ] Worker chosen by modality; no new worker without a reason
[ ] Envelope reused from `model/types.ts`; error `id` discriminates load vs run
[ ] Engine is a pure module: one model live, warm-up on load, cloneable output
[ ] Hook wraps useModelWorker, returns the §3 contract verbatim, adds no alias
[ ] Selection persisted through useModelSelection with the route's own routeKey
[ ] Route renders all four slots; autoLoad from the session; DOM order 1-4
[ ] REAL_ROUTES entry added; placeholder retired
[ ] ModelCard rows created if used; task is a ModelTask choice, not a pipeline
[ ] Vitest contract asserted; @slow spec added for the real download
[ ] Category guide status table updated; plan moved to docs/plans/completed/
```

---

## Reference

- Category guides, listing the verified browser checkpoint for every task:
  [`Audio`](Audio_Models_in_React_WebGPU_and_CPU.md) — the one with shipped
  routes, and the reference for the shared plumbing,
  [`Computer_Vision`](Computer_Vision_Models_in_React_WebGPU_and_CPU.md),
  [`Natural_Language_Processing`](NLP_Models_in_React_WebGPU_and_CPU.md),
  [`Multimodal`](Multimodal_Models_in_React_WebGPU_and_CPU.md),
  [`Tabular`](Tabular_Models_in_React_WebGPU_and_CPU.md),
  [`Reinforcement_Learning`](Reinforcement_Learning_in_React_WebGPU_and_CPU.md),
  [`Other`](Graph_Models_in_React_WebGPU_and_CPU.md).
- **Transformers.js** (`@huggingface/transformers`): pipelines, WebGPU and WASM
  backends, `AutoProcessor`, `RawImage`.
- **onnxruntime-web**: direct ONNX inference for models with no pipeline;
  `onnxruntime-web/webgpu` is the entry point that enables the GPU provider.
- **ONNX weight mirrors**: the `onnx-community/*` and `Xenova/*` orgs mirror
  most published checkpoints. Neither mirrors everything, and neither is
  authoritative — verify.
- **In-repo standards**:
  [`model-page-pattern.md`](../../standards/model-page-pattern.md) (the page
  contract), [`model-visualization.md`](../../standards/model-visualization.md)
  (how a model is drawn), [`../../guides/adding-a-model.md`](../../guides/adding-a-model.md)
  (§8 Transformers.js, §9 a bare ONNX graph),
  [`../../guides/e2e-testing.md`](../../guides/e2e-testing.md).
