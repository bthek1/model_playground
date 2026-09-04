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
`idle` so nothing downloads until the user asks, gate every run control on `ready`, always
render the OUTPUT slot, and put each error in the slot that produced it.

**The hook.** Wrap [`useModelWorker`](../../frontend/src/model/useModelWorker.ts) — it owns
worker creation and teardown, the id-correlated pending table, the response switch, and both
state machines. A task hook is a thin typed wrapper plus whatever is genuinely
task-specific:

```ts
export function useMyTask(model: string, autoLoad = true) {
  const worker = useModelWorker<MyResult>({
    createWorker: createMyWorker,
    key: model,                       // changing this tears down and resets
    loadMessage: { model },
    autoLoad,                         // pass `false` for weight downloads
    notReadyMessage: "My worker not ready",
  });
  const { run } = worker;
  const doThing = useCallback((input: Input) => run({ input }), [run]);
  return { ...worker, doThing };
}
```

**The page.** Four named slots — the shell will not let you reorder them, drop OUTPUT, or
grow a fifth stage:

```tsx
<ModelPage
  icon={Waves}
  title="My Task"
  description="What it does and where it runs."
  select={<ModelPicker models={MY_MODELS} value={model} onChange={…}
                       disabled={loading || running} />}
  load={<ModelStatus status={status} backend={backend} progress={progress}
                     error={status === "error" ? error : null}
                     size={sizeEstimate(meta.params, meta.bytes)}
                     onLoad={load} onRetry={retry} />}
  run={<InputPanel ready={ready} error={ioError} controls={…}>{fields}</InputPanel>}
  output={<OutputPanel title="Result" running={running}
                       error={status === "error" ? null : error}
                       empty="What the user will get.">{result && …}</OutputPanel>}
/>
```

The `status === "error" ? … : …` split on both slots is the §2 discriminator in practice:
a load failure belongs in LOAD, anything else came from a run and belongs in OUTPUT.

**If your task has no weights** — a WGSL kernel compiles in milliseconds — use
[`DeviceStatus`](../../frontend/src/components/model/DeviceStatus.tsx) in the LOAD band
instead and leave `autoLoad` at `true`. The `/tensor` route is the reference. The band still
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
in their own domain folder (`frontend/src/audio/` for the audio tasks) and are
explicitly carved out of the raw-WebGPU-only rule, which scopes to `src/webgpu/`.
Reference implementations: the ASR, audio-classification, and text-to-speech
routes ([`../plans/completed/audio-models-in-browser.md`](../plans/completed/audio-models-in-browser.md)).

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

### Two traps this cost us

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
`/audio-to-audio` (DeepFilterNet3) is the reference example — see
[`docs/plans/completed/audio-to-audio-deepfilternet.md`](../plans/completed/audio-to-audio-deepfilternet.md)
and `src/audio/enhance/`.

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

**d. Add a `@slow` E2E that measures the output.** "A waveform appeared" cannot
distinguish good audio from metallic. `e2e/utils/enhance.ts` runs the real graph
in the page on a synthetic noisy clip and asserts the scale-invariant SDR
improves by ≥6 dB; `e2e/specs/webgpu/enhance.spec.ts` repeats it on the GPU.
