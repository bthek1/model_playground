# Guide: Adding a Model

There are two kinds of model here, and they take different routes:

- **A custom kernel** — you write the maths yourself as a **WGSL compute shader**
  in `src/webgpu/`, plus a **registry entry** for its metadata. Sections 1–6 below.
- **A pretrained model** — an off-the-shelf Hugging Face checkpoint run through
  **Transformers.js / ONNX Runtime Web**. No kernel, no WGSL. [Section 7](#7-adding-a-pretrained-transformersjs-task).

Background: [`../explanations/webgpu-inference.md`](../explanations/webgpu-inference.md).

## 1. Write the kernel (frontend)

Add a WGSL compute shader under `frontend/src/webgpu/shaders/`. Follow the
pattern in `matmul.wgsl`:

- Declare inputs/outputs as `@group(0) @binding(n)` storage buffers and small
  params as a `uniform`.
- Pick a `@workgroup_size(...)` and **guard against out-of-range invocations**
  with an early `return`.

```wgsl
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&input)) { return; }
  output[i] = input[i] * 2.0; // your op here
}
```

## 2. Wire it into the runtime (frontend)

Import the shader as a string and drive it with the existing helpers. A new
kernel typically adds a function to `runtime.ts`:

```ts
import myShader from "./shaders/my_kernel.wgsl?raw";
import { createStorageBuffer, createOutputBuffer, readBackFloat32 } from "./buffers";
import { createComputePipeline } from "./pipeline";
import { getGPUDevice } from "./device";

export async function runMyKernel(input: Float32Array): Promise<Float32Array> {
  const device = await getGPUDevice();
  const pipeline = createComputePipeline(device, myShader);

  const inBuf = createStorageBuffer(device, input);
  const outBuf = createOutputBuffer(device, input.byteLength);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inBuf } },
      { binding: 1, resource: { buffer: outBuf } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(input.length / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);

  const result = await readBackFloat32(device, outBuf, input.byteLength);
  [inBuf, outBuf].forEach((b) => b.destroy());
  return result;
}
```

To run it **off the main thread**, add a message type to `worker.ts` and a
matching helper in `workerClient.ts` (mirror `matmul` / `runMatmulInWorker`).

> Tip: cross-check a few outputs against a CPU reference during development, like
> `useGpuBenchmark` does — a wrong kernel often still runs and returns garbage.

## 3. Register the model (backend)

Add a catalog entry so the model appears in the playground. Either via the Django
admin (`/admin/` → Model cards) or the API:

```bash
curl -X POST http://localhost:8000/api/registry/models/ \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "slug": "my-net",
    "name": "My Net",
    "task": "vision",
    "description": "What it does.",
    "weights_url": "https://cdn.example.com/my-net.safetensors",
    "config": { "entry_point": "main", "input_shape": [1, 3, 224, 224] },
    "license": "MIT",
    "is_public": true
  }'
```

`config` is free-form JSON — put whatever the browser runtime needs to build the
pipeline and preprocess inputs (shapes, quantization, WGSL entry point, tokenizer
refs). Weights are fetched by the **browser** from `weights_url`; they are not
served by Django.

Field reference: [`../standards/api-contracts.md`](../standards/api-contracts.md).

## 4. (Optional) Report run metrics

After a run, `POST /api/registry/runs/` with `params` and `metrics`
(`latency_ms`, `tokens_per_sec`, `gflops`, adapter info) so the playground can
show history and benchmarks.

## 5. Build the task page

Every task route is the same pipeline — **Select → Load → Run → Output**. Do not design a
new page shape; fill in the four slots. The contract (the two orthogonal state machines,
the hook shape every task hook returns, the slot rules, and the deferred-load guardrail) is
in [`../standards/model-page-pattern.md`](../standards/model-page-pattern.md), which ends
with a checklist for exactly this step.

The short version: wrap the shared worker plumbing rather than re-deriving it, default to
`idle` so nothing downloads until the LOAD button is pressed, hold the input rather than
running on it as it arrives, gate the GENERATE trigger on `ready`, always render the
OUTPUT slot, and put each error in the slot that produced it. Selection comes from
`useModelSelection` — don't hold the model in a route-local `useState`, or the user's
choice dies on every reload.

**Two buttons, and only two.** LOAD spends bandwidth and memory; GENERATE spends GPU and
time. Everything else — picking a model, picking an input, editing a parameter — is a
choice that costs nothing and commits to nothing. A page where clicking a sample image
runs the model, or where arriving at the page downloads it, has collapsed a choice into a
commitment; that is the failure this pipeline exists to prevent, and it is spelled out in
[`model-page-pattern.md` §1.2 and §1.6](../standards/model-page-pattern.md).

**The hook.** Wrap [`useModelWorker`](../../frontend/src/model/useModelWorker.ts) — it owns
worker creation and teardown, the id-correlated pending table, the response switch, and both
state machines. A task hook is a thin typed wrapper plus whatever is genuinely
task-specific:

```ts
export function useMyTask(model: string, autoLoad = false) {
  const worker = useModelWorker<MyResult>({
    createWorker: createMyWorker,
    key: model,                       // changing this tears down and resets
    loadMessage: { model },
    autoLoad,                         // leave it false for anything that downloads
    notReadyMessage: "My worker not ready",
  });
  const { run } = worker;
  const doThing = useCallback((input: Input) => run({ input }), [run]);
  return { ...worker, doThing };
}
```

**The selection.** One hook owns which model is picked and whether its weights are
already downloaded ([`model-page-pattern.md` §5b](../standards/model-page-pattern.md)):

```ts
const session = useModelSelection({
  routeKey: "my-task",              // stable per route; keys the stored preference
  models: MY_MODELS,
  fallback: MY_MODELS[0],
});
const { status, ready, loadProgress, loadedInMs, load, retry, cancel, … } =
  useMyTask(session.model.id);      // no second argument — the default is `idle`
useCacheRefresh(session, ready);    // re-probe the cache once the download lands
```

The hook has **no `autoLoad` to hand you**, deliberately: it cannot start a load, so
there is no path by which arriving at a page, switching models, or refreshing begins a
download. What the cache probe gives you is `session.isCached`, which changes the LOAD
button's words ("Load model (cached)") and the copy beneath it — an informed click, not
an absent one.

**The page.** Four named slots — the shell will not let you reorder them, drop OUTPUT, or
grow a fifth stage:

```tsx
<ModelPage
  icon={Waves}
  title="My Task"
  description="What it does and where it runs."
  select={<ModelPicker models={MY_MODELS} value={session.model.id}
                       onChange={session.setModel} disabled={loading || running}
                       cached={session.cached}
                       onEvict={(m) => void session.evict(m.id)} />}
  load={<ModelStatus status={status} backend={backend}
                     loadProgress={loadProgress} loadedInMs={loadedInMs}
                     cached={session.isCached}
                     error={status === "error" ? error : null}
                     onLoad={session.onLoad(load)}
                     onCancel={session.onCancel(cancel)}
                     onRetry={retry} />}
  run={<InputPanel ready={ready} error={ioError} controls={…}>{fields}</InputPanel>}
  output={<OutputPanel title="Result" running={running}
                       error={status === "error" ? null : error}
                       empty="What the user will get.">{result && …}</OutputPanel>}
/>
```

`ModelStatus` takes `loadProgress` — the aggregate from `model/progress.ts` — never the
raw `progress` event. `session.onLoad` / `session.onCancel` wrap the actions so every
route's LOAD slot is wired identically.

**The RUN slot holds its input.** Whatever the modality, the decoded thing is state:
[`useImagePick`](../../frontend/src/hooks/useImagePick.ts) for images,
[`useAudioPick`](../../frontend/src/hooks/useAudioPick.ts) for audio, or your own
`useState` for a task neither fits. Neither pick hook takes an "on picked" callback, and
that is the point — the RUN slot's single trigger reads the held input and is the only
thing that calls `run`:

```tsx
const input = useAudioPick();                 // or useImagePick()

const generate = () => {
  const audio = input.take();                 // a *copy* — the worker detaches it
  if (!audio) return;
  input.clearError();
  void run(audio).catch(() => { /* surfaced in OUTPUT */ });
};
```

Capture whatever OUTPUT needs to render the result *inside* the run, not from the input
currently held — otherwise choosing a new input restyles the previous result.

The `status === "error" ? … : …` split on both slots is the §2 discriminator in practice:
a load failure belongs in LOAD, anything else came from a run and belongs in OUTPUT.

**If your task has no weights** — a WGSL kernel compiles in milliseconds — use
[`DeviceStatus`](../../frontend/src/components/model/DeviceStatus.tsx) in the LOAD band
instead, and that is the one case for passing `autoLoad: true` explicitly. The `/tensor`
route is the reference. The band still
renders, so the page keeps the same rhythm as one that downloads 200 MB.

**Band labels stay generic** (Model / Load / Input / Output). `ModelPage` accepts a `labels`
override, but naming a band after the task duplicates the field label or card title directly
beneath it — and makes the accessible name ambiguous, since the band is a labelled region.
Reach for it only when the task has genuinely better words, as `/tensor` does with
Operation / Device / Operands.

## 6. Visualize the model

Give the model a view that shows the user its identity, structure, parameters, and
performance. Follow the UI standard in
[`../standards/model-visualization.md`](../standards/model-visualization.md): reuse the
stage/arrow schematic, canvas weight/activation heatmaps, param chips, theme-token colors,
and lazy charts (reference impl: `components/training/ModelArchitecture.tsx` +
`ModelWeights.tsx`). Drive it from the real `ModelCard.config` and weight stream — not a stock
diagram — and make sure it still renders its structure with no GPU. Work through the checklist
at the end of that document.

## 7. Verify

- `just fe-lint && just fe-build` — kernel imports and types compile.
- `just be-test` — registry endpoints still pass.
- Open `/playground`, confirm the model appears in the catalog and your kernel
  runs on the GPU.
- Toggle light/dark and confirm the model's visualization reads correctly in both.

---

## 8. Adding a pretrained (Transformers.js) task

Running a *pretrained* checkpoint is a different job from writing a kernel: the
model already exists as ONNX on the Hub, so there is no WGSL to write. These live
in their own domain folder — `frontend/src/audio/` for the audio tasks,
`frontend/src/vision/` for the vision ones — and are explicitly carved out of the
raw-WebGPU-only rule, which scopes to `src/webgpu/`. Reference implementations:
the ASR, audio-classification and text-to-speech routes
([`docs/roadmaps/audio.md`](../roadmaps/audio.md)) and `/image-classification`
([`docs/roadmaps/vision.md`](../roadmaps/vision.md)).

The two modalities are deliberately the same shape: a generic worker per
modality, a pure engine, a thin task hook. What differs is only the payload —
`Float32Array` samples one side, a flattened image the other, because a
`RawImage` is a class instance and does not survive `postMessage` (see
`vision/image.ts` → *Worker transport*). The **backend probe and the size
guardrail are shared by both** and live in `src/model/` (`backend.ts`,
`size.ts`); a third modality imports them rather than copying them.

**On pinning `dtypes`:** the rule is "leave it unset unless a real measurement
says otherwise", and when you do pin one, say in the comment **whether it is a
measurement or a precaution** — they are different claims and only one of them is
evidence. `/image-classification` pins fp32 because a q8 MobileNetV4 called a
tiger a rattlesnake, which is a measurement. `/super-resolution` pins its WASM
precision as a precaution (dense regression puts int8 error straight into the
picture) and names the spec that would relax it. A pinned entry then owes
**measured** `bytes`, because the params estimate no longer describes it.

The shape is always the same four pieces:

**a. Catalogue entry.** Add the model to the task's catalogue (`audio/types.ts`
for ASR, `audio/classification.ts`, `audio/tts.ts`). Every entry carries a
`params` count in millions — that drives the **size-before-load guardrail** in
`ModelPicker`, which quotes the download for both backends and warns past
`LARGE_MODEL_BYTES` (300 MB). Weights are fp16 on WebGPU, quantized q8 on WASM.

```ts
{
  id: "onnx-community/whisper-base",   // must be ONNX-exported on the Hub
  label: "Whisper base",
  hint: "Timestamps, 99 languages, translate.",
  params: 74,                          // millions — drives the size estimate
}
```

**b. Worker.** For a discriminative task, reuse the **generic pipeline worker** —
add the pipeline string to `PipelineTask` in `audio/pipelineTypes.ts` and give it
warm-up args in `pipelineEngine.ts`; no new worker file. Write a dedicated worker
only when the modality differs (TTS is text → audio) or the dependency is heavy
enough that it should bundle separately (kokoro-js). A dedicated engine mirrors
`asrEngine.ts`: a testable `create…Handler(post, factory, { warmup })` plus a thin
`*.worker.ts` wrapper that wires it to `self`.

Every engine owes three behaviours:

- **One model live at a time.** Null the reference *first*, then dispose, so a
  failed teardown can't leave a stale model held (`disposeQuietly`).
- **Warm up on load.** Run one throwaway inference (silence, or a two-word
  phrase) before posting `ready`, so the first real request doesn't pay to
  compile the WebGPU shaders / JIT the WASM module. Post a
  `{ status: "warmup" }` progress; never fail the load if it throws.
- **Never block the main thread.** Inference runs in the worker; buffers are
  transferred, not copied.

**c. Hook + route.** `usePipeline(task, model)` already handles load, progress,
id-correlated runs, and worker teardown — wrap it in a task hook that shapes the
run args (see `useAudioClassifier.ts`). The route composes `ModelPicker` +
`ModelStatus` with the task's own controls, and must degrade gracefully when the
backend is WASM-only.

**d. Taxonomy.** Map the task slug to the real route in `REAL_ROUTES`
(`components/layout/taskTaxonomy.ts`), or it falls through to the
`/tasks/$slug` placeholder.

### Three traps these cost us

- **Verify the repo id against the Hub before shipping it.** Two catalogue entries
  pointed at `onnx-community/*` repos that don't exist; the Hub answers **401** and
  the model fails at load with "Unauthorized access to file". A quick
  `curl -s -o /dev/null -w "%{http_code}" https://huggingface.co/api/models/<id>`
  would have caught both. Unit tests can't — they never touch the network.
- **A model that downloads is not a model that runs.** The quantized Whisper and
  Moonshine *decoders* cannot open an ONNX Runtime session on the WASM execution
  provider bundled with `@huggingface/transformers` 4.2.0 (it throws
  `qdq_actions.cc:137 … Missing required scale`). `asrLoadOpts` works around it by
  keeping the decoder at `fp32` on WASM. When a new task fails only on the WASM
  fallback, suspect the quantized weights before your own code, and check whether
  a per-module dtype (`{ encoder_model: "q8", decoder_model_merged: "fp32" }`)
  clears it.
- **A model that runs is not a model that is right.** `onnx-community/mobilenetv4_conv_small`
  at q8 loads, runs at full speed, and labels a photo of a tiger "sidewinder,
  horned rattlesnake" at 44%; the same weights at fp32 say "tiger 62%".
  Depthwise-separable convolutions are the classic casualty of per-tensor int8
  quantization, so treat the whole MobileNet family as suspect. Nothing in the
  load path reports this — the only thing that catches it is asserting a **known
  label on a known image** in the `@slow` E2E spec. A vision catalogue entry pins
  the precision per backend with `dtypes: { wasm: "fp32" }`, and then owes
  measured `bytes`, because the params estimate is now off by 4x.

### Verify

Neither WebGPU nor the Web Audio API exists in the test env, so unit tests mock
both: cover the engine against a fake pipeline factory (no download), the hook
against a fake `Worker`, and the route's rendering. Then check it for real —
`just fe-dev` over **HTTPS** (`navigator.gpu` needs a secure context), load the
model on WebGPU *and* with WebGPU disabled to force the WASM fallback, and switch
models a few times watching that memory doesn't grow.

---

## 9. Adding a custom-ONNX task (no Transformers.js)

Some models have no Transformers.js task at all: they publish a bare ONNX graph
that takes hand-built feature tensors and returns hand-interpreted outputs.
Two routes do this today: `/audio-to-audio` (DeepFilterNet3, `src/audio/enhance/`)
is the reference example, and `/vad` (Silero VAD, `src/audio/vad/`) is the
smaller one to read first — ~200 lines against enhance's ~600.

The route, worker, hook and page pattern are **unchanged** — reuse
`useModelWorker`, the engine's three duties, `ModelPicker`, `ModelStatus`. Only
the layer below the engine differs:

**a. Import ONNX Runtime from the same entry Transformers.js uses.**

```ts
import * as ort from "onnxruntime-web/webgpu";
```

`onnxruntime-web` is pinned in `package.json` to the exact version
`@huggingface/transformers` depends on, so npm dedupes to one copy. The
**subpath matters as much as the version**: the bare `onnxruntime-web` entry
resolves to a different build and makes Vite emit a *second* 26 MB WASM asset.
Import `/webgpu` and both share one. Add the subpath to `optimizeDeps.include`
in `vite.config.ts` too — discovered mid-session inside a Web Worker, Vite
re-optimises and triggers a full page reload that resets a route mid-load.

Check what actually shipped after any change here:

```bash
just fe-build
grep -o "ort-wasm[^\"']*\.wasm" frontend/dist/assets/*.worker-*.js | sort -u
```

**b. Own the pre/post-processing, and treat it as the risky part.** The graph is
the easy half. `src/audio/enhance/` is ~600 lines of DSP against ~80 lines of
session code, and none of the DSP fails loudly — a wrong constant yields
plausible audio with artefacts, not an exception.

So: **validate against the reference implementation, not against your
expectations.** For DeepFilterNet3 that meant running the official
`DeepFilterNet` package (its `libDF` wheel needs Python ≤3.11) over the same
input and comparing arrays at each stage. That caught what reading the model
card could not — the analysis spectrum has to carry libDF's `wnorm` scaling
(`2 * hop / fft²`), because the unit-norm feature divides by `sqrt(state)` and
is therefore *not* level-invariant. Without it the network sees a signal ~31x
too loud and masks clean speech away as noise, silently. Capture the reference
arrays as a JSON fixture (`src/audio/enhance/__fixtures__/`) so the unit tests
keep checking against them.

**c. Validate the constants file on load, and refuse to run on a mismatch.**
DeepFilterNet3 ships `deepfilter-auxiliary.bin`: 124 KB of untyped float32 whose
layout is documented in prose. `parseAux` asserts every invariant it can (band
sums, contiguity, window symmetry and power-complementarity) and throws rather
than emit noise. Note the model card describes the forward matrix as `[481,32]`
and the inverse as `[32,481]` — they are stored in *different* orders, and
reading either as the other still yields a plausible-looking matrix.

**d. A recurrent graph owns two extra invariants.** Silero VAD scores one 32 ms
frame at a time and returns a state tensor to feed into the next call, which adds
two ways to be silently wrong (`src/audio/vad/vad.ts`):

- **The window is not the frame.** Silero v5 expects 64 samples of preceding
  context prepended to each 512-sample frame — 576 in, not 512. The graph's input
  dimensions are dynamic, so a bare 512 runs happily and returns numbers that
  never cross any threshold. Measured on the same clip: with the context, silence
  reads 0.005 and speech 0.83–0.99; without it, speech reads 0.05.
- **State is per clip, not per session.** `state` starts at zeros and the context
  at silence for every take. Carrying either across takes leaks the previous
  clip's tail into the next one's first frames.

Read the contract off the graph rather than off the model card — the card for
`onnx-community/silero-vad` is four lines of YAML, and §3.6 of the Audio roadmap
described v4's separate `h`/`c` inputs, which v5 merged into one `state`.

**e. Pick the execution provider deliberately.** `["webgpu", "wasm"]` is the
default provider list, not a law. Silero is pinned to **WASM**: a 576-sample
window costs 0.30 ms on CPU (about 100x real time), so a per-frame GPU dispatch
and readback would cost more than the work, and its LSTM/`If` ops are not covered
by ORT's WebGPU provider anyway. Say which you chose and why in the session
module, and have the `@slow` spec assert it — otherwise a later "optimisation"
quietly loosens it.

**f. Add a `@slow` E2E that measures the output.** "A waveform appeared" cannot
distinguish good audio from metallic, and "the page rendered" cannot distinguish
a working detector from one whose every score is 0.04. `e2e/utils/enhance.ts`
runs the real graph on a synthetic noisy clip and asserts the scale-invariant SDR
improves by ≥6 dB; `e2e/specs/webgpu/enhance.spec.ts` repeats it on the GPU. The
VAD spec asserts a real speech *fraction* on a known clip for the same reason —
it is the only test in the suite that would have caught a 512-sample window.

---

## 10. When the pipeline is the wrong abstraction

§8 assumes `pipeline(task, model)` fits. Four vision routes found it did not, and
they all failed the same test: **is this a plain `pipeline()` call?** If the
answer is no, the task owns an engine — `engine.ts` (pure, testable with fakes),
`*.worker.ts` (the only file that imports the runtime), `client.ts` (so the hook
can mock worker creation). The plumbing above it does not change: `useModelWorker`,
the three engine duties, `ModelPicker`, `ModelStatus`, the four slots.

The four reasons, each a different shape:

**a. The pipeline throws away work you want to keep.**
`/zero-shot-image-classification` — the pipeline re-encodes the labels on every
call, and the label embeddings do not depend on the image. `src/vision/zeroshot/`
drives the two towers separately and caches the text side. The cost of splitting
is owning the graph's tail (normalise → scale → softmax) in `scoring.ts`, where a
wrong `logit_scale` fails *silently*: scores stay in [0, 1] and the ranking is
unchanged. `just fe-e2e-zeroshot` compares against the full graph, which is the
only thing that can catch it.

**b. The architecture is a split, and the split is the point.**
`/mask-generation` — SAM ships a vision encoder and a prompt-encoder/mask-decoder.
`src/vision/sam/` runs the encoder once per image and decodes a mask per click.
The trap: `SamModel.forward` computes the embeddings itself when they are
missing, so omitting them still returns **correct masks** at full encoder cost on
every click. Nothing in the output says so. This is also the route that showed
encode-once/decode-many needs *two states* in the UI — LOAD is the download,
encoding is the per-image pass, and collapsing them means the user clicks, waits
a second, and is told nothing.

**c. The pipeline cannot load the model at all.**
`/image-to-text` — `ImageToTextPipeline` resolves through
`AutoModelForVision2Seq`, whose registry maps `vision-encoder-decoder`,
`idefics3` and `smolvlm`. Florence-2's model type is `florence2`, registered
under *image-text-to-text*. It also does exactly two things — run the processor
for `pixel_values`, call `generate({ inputs })` — so there is nowhere to put a
task token even for the models it can load. Same shape as MusicGen needing
`MusicgenForConditionalGeneration` rather than the `text-to-audio` pipeline.
**Check the registry before writing the hook**, not after the first load fails:

```bash
grep -n "MODEL_FOR_.*MAPPING_NAMES" -A 20 \
  frontend/node_modules/@huggingface/transformers/src/models/registry.js
```

`vision/caption/` then puts *both* families behind one `Captioner` interface,
which is the same trade `tts.worker.ts` makes for Kokoro / MMS / MusicGen. One
engine per **modality**, not per model.

**d. The task is two models.**
`/pose` — a detector finds people, a pose model runs on each person's crop.
`src/vision/pose/` holds both live, which is the single documented exception to
"one model live at a time", and pays for it in three places: the catalogue entry
quotes the **combined** download, both models are loaded together so their
progress events interleave into one bar, and both are disposed with
`Promise.allSettled` so a teardown that throws does not skip the larger one.

### The rule these share

Every one of the four introduced a failure that **produces plausible output**
rather than an error — a cached embedding that is silently recomputed, a scale
that leaves the ranking intact, a task token a model has never seen answered with
a fluent unrelated sentence, a skeleton offset by a crop's origin. So each one
owes a `@slow` spec that measures a **property**, never a count:

| Route | What the `@slow` spec asserts |
|---|---|
| `/zero-shot-image-classification` | the split path's numbers against the full graph |
| `/mask-generation` | the mask's **coverage band** — ~0% and ~100% are what a broken coordinate space produces |
| `/image-to-text` | a substring of the text actually printed in the picture |
| `/pose` | the nose is **above** the ankles |
| `/zero-shot-object-detection` | a present phrase finds boxes and an absent one finds none |
| `/image-features` | an animal's nearest neighbour is an animal |
| `/background-removal` | the matte's **coverage band** — 0% and 100% both render beautifully |
| `/super-resolution` | **PSNR against a ground truth**, beating a bicubic resize |
| `/image-to-3d` | the point count tracks the stride, and sliders re-derive without a run |

"Five rows appeared" passes for all of them.

Two of the Wave 3 rows are worth a sentence each, because they are the cases
where a *plain* `pipeline()` call is still the right answer and the risk sits
somewhere else entirely:

- **`/background-removal` is a plain pipeline call whose risk was a licence.**
  `briaai/RMBG-1.4` is Creative Commons **non-commercial**, in an MIT repo, so the
  default is Apache-2.0 MODNet and the restriction is rendered beside the choice.
  Read the model card before the catalogue entry — it is the one thing that can
  invalidate a route after it is built. The second risk was the *sample set*:
  MODNet is a **portrait** matting model, and on a photo with no person in it it
  returns a near-empty matte rather than failing. The `@slow` spec measured 0.2%
  coverage before `PORTRAIT_SAMPLES` existed.
- **`/super-resolution` is a plain pipeline call whose risk was arithmetic.** The
  model is one pass, but a *run* is many of them, and the tiling and seam
  blending around it are ours. That code (`vision/tile.ts`) is pure and
  exhaustively unit-tested — including an identity round-trip that must reproduce
  the input pixel for pixel — because a mis-assembled upscale is perfectly sharp
  and looks fine. Its `@slow` spec **builds its own ground truth**: there was no
  reference image to score against, so it crops a bundled sample, halves it in
  the page, uploads that as the input (`setInputFiles({ buffer })`, no file on
  disk) and scores the 2x output against the crop it started from. When "what is
  the right answer?" has no bundled answer, making one from a known input is
  usually cheaper than lowering the assertion.
