# Computer Vision Models in the Browser (WebGPU or CPU)

> The **Computer Vision** category of `components/layout/taskTaxonomy.ts`, task by
> task: what runs client-side on the user's GPU (WebGPU) or CPU (WebAssembly),
> which checkpoint to use, and which tasks stay on a server. No Python server in
> the inference path.

**Fourteen of twenty are built.** §3.1 to §3.11 and §3.13 to §3.15 all ship —
[`/depth`](../../frontend/src/routes/depth.tsx),
[`/image-classification`](../../frontend/src/routes/image-classification.tsx),
[`/object-detection`](../../frontend/src/routes/object-detection.tsx),
[`/segmentation`](../../frontend/src/routes/segmentation.tsx),
[`/zero-shot-image-classification`](../../frontend/src/routes/zero-shot-image-classification.tsx),
[`/zero-shot-object-detection`](../../frontend/src/routes/zero-shot-object-detection.tsx),
[`/mask-generation`](../../frontend/src/routes/mask-generation.tsx),
[`/image-features`](../../frontend/src/routes/image-features.tsx),
[`/image-to-text`](../../frontend/src/routes/image-to-text.tsx),
[`/pose`](../../frontend/src/routes/pose.tsx),
[`/video-classification`](../../frontend/src/routes/video-classification.tsx),
[`/background-removal`](../../frontend/src/routes/background-removal.tsx),
[`/super-resolution`](../../frontend/src/routes/super-resolution.tsx) and
[`/image-to-3d`](../../frontend/src/routes/image-to-3d.tsx).
The other six tasks in the sidebar still render the `/tasks/$slug`
placeholder, and §3.12 says why each of them stays on a server. This file is
therefore two things at once: a description of the shipped `src/vision/` module
(§1–§2), and the record of what each route settled (§3) — the verified
checkpoint, the shape of the page, and the specific failure each decision guards.

The category is **twenty** rows rather than nineteen because
[`/background-removal`](../../frontend/src/routes/background-removal.tsx) added
one: the Hub has no `background-removal` task, so listing it is a deliberate
departure from the taxonomy's mirror of the Hub's pipeline tags. §3.12 and
[#24](https://github.com/bthek1/model_playground/issues/24) record that call.

Two of the last three are **partial** by design, and their pages say so: Image to
Image ships super-resolution and not editing, Image to 3D ships depth-to-cloud
and not reconstruction. The procedure for a new page is
[`docs/guides/adding-a-task-page.md`](../guides/adding-a-task-page.md).

Vision splits more cleanly than any other category. The
**discriminative half** (depth, classification, detection, segmentation,
embeddings, keypoints, open-vocabulary anything) runs beautifully in a tab: the
models are 20 MB to 200 MB, they are single-pass, and almost all of them have an
official ONNX export. The **generative half** (text-to-image, image-to-video,
text-to-video, text-to-3D, unconditional diffusion) does not run in a tab at
all, and the reason is structural rather than a missing export: latent diffusion
needs multi-gigabyte weights and tens of denoising steps.

So this file is honest about a hard split. Sections 3.1 to 3.11 and 3.13 to 3.15
are the pages that were built; section 3.12 is the list of tasks that stay on a
server, with the reason for each — and the three carve-outs hiding inside it that
turned out not to need one. **§3.12 keeps its number**: it is cited from a dozen
source comments and from `CLAUDE.md`, so the Wave 3 pages were appended after it
rather than renumbering everything.

Every model id below was checked against the Hugging Face API. Re-check before
shipping with `just fe-e2e-models`.

---

## 1. The core stack

`@huggingface/transformers` and `onnxruntime-web` are already dependencies —
nothing needs installing. Vision adds no new runtime.

Backend selection and dtype are **shared, not re-derived**. Import them:

```ts
import { pickBackend, loadOpts } from "@/model/backend";
```

They live in `src/model/` rather than `src/audio/`, where they started: nothing
in `pickBackend()` or `loadOpts()` was ever audio-specific, so they moved when
the first vision page landed rather than being copied. The size-before-load
guardrail moved with them, to
[`model/size.ts`](../../frontend/src/model/size.ts).

**A catalogue entry can override the precision per backend**
(`VisionModel.dtypes`), and one already does. `onnx-community/mobilenetv4_conv_small`'s
q8 export loads, runs, and calls a photo of a tiger "sidewinder, horned
rattlesnake" at 44%; at fp32 the same weights say "tiger 62%". Depthwise-separable
convolutions are the classic casualty of per-tensor int8 quantization, so treat
the whole MobileNet family as suspect and measure before shipping. This is the
same shape of exception as `asrLoadOpts()` on the audio side — and the same
lesson: **a quantized export can be wrong rather than merely worse, and nothing
in the load path will tell you.**

The rule that carries hardest into a tab: **hold one large model live at a time
and free it before loading the next.** A WebGPU context can OOM the tab exactly
the way a careless notebook cell OOMs a 12 GB container.

---

## 2. The shipped module (`src/vision/`)

The vision counterpart of `src/audio/`, and what every vision route imports. It
was established by `/image-classification`; the four Wave 1 routes added a
catalogue, a hook and a route each, plus the three shared pieces at the bottom of
the table — the parts all five pages turned out to need identically. A sixth page
should add a catalogue, a hook and a route, and nothing else.

| File | What it is |
|---|---|
| [`image.ts`](../../frontend/src/vision/image.ts) | Decoding and capture: `fromFile` / `fromUrl` / `fromVideo` / `openCamera`, `downscale`, and the worker transport (`toPayload` / `fromPayload`) |
| [`draw.ts`](../../frontend/src/vision/draw.ts) | Canvas overlays: `drawBoxes`, `drawHeatmap`, `drawMasks`, `drawPixels`, `scaleDetections`, `colorForLabel`, `RAMP_CSS` |
| [`types.ts`](../../frontend/src/vision/types.ts) | `VisionTask`, the `VisionModel` catalogue shape, the worker envelope |
| [`engine.ts`](../../frontend/src/vision/engine.ts) | The pure message handler — one model live, warm-up before `ready`, unit-tested with a fake factory |
| [`serialize.ts`](../../frontend/src/vision/serialize.ts) | `toCloneable` — flattens a result so `postMessage` will take it |
| [`vision.worker.ts`](../../frontend/src/vision/vision.worker.ts) | Thin wrapper: wires the engine to `self` and supplies the real Transformers.js factory |
| [`samples.ts`](../../frontend/src/vision/samples.ts) | Bundled demo images, so a route works before the user picks a file |
| [`useVisionPipeline`](../../frontend/src/hooks/useVisionPipeline.ts) | The hook every task hook wraps |
| [`useLiveFrames`](../../frontend/src/hooks/useLiveFrames.ts) | The camera frame pump, plus `useCamera` |
| [`useCameraFrames`](../../frontend/src/hooks/useCameraFrames.ts) | The three of them wired together: open → grab → `downscale` → hand over one frame at a time |
| [`useImagePick`](../../frontend/src/hooks/useImagePick.ts) | File / drop / sample decoding, and the object-URL lifecycle |
| [`components/vision/`](../../frontend/src/components/vision/) | `ImageSourcePanel` (the RUN slot's input surface) and `OverlayCanvas` (a canvas at source resolution, painted by a callback) |

**One worker for every discriminative vision task.** The pipeline `task` travels
in the `load` message, exactly as `audio/pipeline.worker.ts` does — classification,
detection, segmentation, depth, zero-shot and features all share it. A task that
is not a plain `pipeline()` call (SAM's encode-once/decode-many, for one) gets its
own engine instead, the way `audio/enhance/` and `audio/vad/` do.

### Five things this module already settled

**A `RawImage` does not survive `postMessage`.** It is a class instance; the
structured clone arrives as a plain object with no methods and the pipeline
rejects it. `toPayload` sends the pixels plus `{width, height, channels}` and
`fromPayload` rebuilds it inside the worker. `toPayload` **copies by default** —
the page is usually still displaying the image it just sent, and transferring the
buffer detaches it into a blank preview. Pass `{ copy: false }` only for a webcam
frame the main thread has finished with.

**A `Tensor` does not survive `postMessage` either — and it fails louder.** A
Transformers.js `Tensor` exposes `data` and `dims` as **prototype getters** over
an internal ONNX Runtime tensor, and structured clone copies own properties and
refuses the rest, so posting one throws outright:

```
Failed to execute 'postMessage' on 'DedicatedWorkerGlobalScope':
#<_Tensor> could not be cloned.
```

Depth estimation is the task that hits it (`{ predicted_depth: Tensor, … }`), and
`toCloneable` in [`serialize.ts`](../../frontend/src/vision/serialize.ts) flattens
every result before the engine posts it — duck-typed, so `engine.ts` still needs
no runtime import. Note where this bug could and could not be seen: the unit
suite mocks the worker away, and a mocked E2E run never loads a model, so **only
a real load surfaced it.** That is what the `@slow` specs are for, and it is the
same lesson as the quantized MobileNetV4 calling a tiger a rattlesnake.

**Do not resize or normalise for the model.** Every pipeline calls
`AutoProcessor`, which reads `preprocessor_config.json` from the model's own repo
and applies that model's resize, crop, rescale and normalise. That file *is* the
published input contract. Reimplementing it by hand is the most common way to end
up with a model that runs, produces plausible-looking output, and is quietly
wrong — which is exactly how the two DSP bugs in `audio/enhance/` shipped past a
green test suite. `downscale()` is the one exception and is not preprocessing: it
caps the *source* resolution before the processor ever sees it, because a
detector at 1280x720 costs roughly 4x the same detector at 640x480.

**Normalise a single-channel map before painting it.** A relative-depth model
emits inverse depth on an arbitrary scale; without rescaling to the values
actually present the canvas is uniformly black or uniformly white, and the page
looks broken rather than wrong. `drawHeatmap` does this per map and returns the
bounds it used, for the legend. Before writing a new drawing helper, check
[`components/viz/`](../../frontend/src/components/viz/) and the standard in
[`docs/standards/model-visualization.md`](../standards/model-visualization.md) —
`HeatmapTile` and `DivergingLegend` are already there and already theme-aware.

**Never queue frames.** `useLiveFrames` grabs a frame only once the previous
result is back. A naive `requestAnimationFrame` loop that posts every frame builds
an unbounded backlog, the overlay drifts seconds behind the picture, and the page
looks like the model is slow when it is the queue. `useCamera` owns the teardown
for the same reason there is one owner: a leaked `MediaStream` leaves the webcam
light on after the user has navigated away. Routes do not assemble these
themselves — `useCameraFrames` is the wiring, and it also drops the first frames,
before the stream has dimensions, which otherwise yield a 0x0 canvas and one
pipeline error per frame.

### And one the four Wave 1 routes settled

**The input surface and the object-URL lifecycle are shared, not copied.** Five
routes need the same file / drop / sample picker, and the same rule underneath
it: exactly one object URL alive at a time, and none after unmount — a preview
URL that outlives its `<img>` pins the decoded bitmap for the tab's lifetime.
`useImagePick` owns that, `ImageSourcePanel` renders it, and
`/image-classification` was migrated onto both rather than left as a fifth copy.

`ImageSourcePanel` renders the *input* only. The overlay — boxes, a depth map,
class masks — is a **result**, and results live in OUTPUT
([model-page-pattern.md](../standards/model-page-pattern.md) §4). Painting the
answer on top of the input would collapse the two slots into one, which is
exactly the drift the four-slot shell exists to prevent.

**And picking is not running.** `useImagePick` originally took an `onPicked`
callback so a route could classify the picture the instant it decoded, which read
as a convenience and was not one: thirteen routes used it, so a user comparing the
sample row spent one GPU inference per click and the Classify button beside them
did nothing they had not already been charged for. The callback is **gone** — not
deprecated, not merely unused — so it cannot come back one route at a time, and
[§1.6](../standards/model-page-pattern.md) is now the rule for both modalities.
The corollary shows up in the panel too: its sources are **not** gated on `ready`,
because choosing what to run before choosing what to run it with is a sensible
order to work in.

---

## 3. Task by task

### 3.1 Depth Estimation — **shipped** at [`/depth`](../../frontend/src/routes/depth.tsx)

Taxonomy task **Depth Estimation** · built. Upstream research: Depth Anything V2,
Depth Pro, ZoeDepth.

Depth Anything V2 Small is 25M parameters, single pass, and has an official ONNX
export. It runs at interactive rates on WebGPU and produces the most immediately
impressive output of any task here.

| Model | Download (WebGPU · WASM) | Role |
|---|---|---|
| `onnx-community/depth-anything-v2-small` | 47 MB fp16 · 26 MB q8 | relative depth, the default |
| `Xenova/depth-anything-small-hf` | 48 MB fp16 · 26 MB q8 | the older mirror, equivalent |
| `onnx-community/DepthPro-ONNX` | **962 MB q8**, WebGPU only | metric depth plus focal length — gated |
| `Intel/zoedepth-nyu-kitti` | none | no export. Use Depth Pro for metric |

Sizes are read off the Hub's blob listing. Two findings that only a real look at
the repos could produce:

* **Depth Pro is pinned to q8 on both backends**, not merely warned about. Its
  fp16 export is **1.8 GB**, which is past what a tab reliably holds alongside a
  WebGPU context; q8 is still ~1 GB, so the route gates it behind an explicit
  opt-in notice — the same gate `/text-to-audio` puts in front of MusicGen —
  rather than relying on the picker's size line alone.
* **The legend direction is read from the catalogue, not hard-coded.** Depth
  Anything emits *inverse* depth (a big number is near); Depth Pro emits metres
  (a big number is far). One ramp, two labellings, and `DepthModel.metric` picks.

The page's two obligations, both about honesty rather than code:

1. **The output is relative depth, not metres**, on a scale re-fitted to each
   image — so two frames cannot be compared without aligning them first. That is
   UI copy next to the colour bar, not a code comment.
2. **Normalise per frame before drawing.** `drawHeatmap` does it; bypassing it
   renders an arbitrary-scale map as uniformly black or white, and the page then
   looks broken rather than wrong — a much harder bug to notice.

```ts
const depth = await pipeline("depth-estimation", "onnx-community/depth-anything-v2-small",
                             loadOpts(backend));
const { predicted_depth, depth: asImage } = await depth(image);
// predicted_depth is the raw tensor -> drawHeatmap()
// asImage is an already-normalised RawImage if you just want a picture
```

`useDepth` returns both. `depthDims()` reads `[1, h, w]` and `[h, w]` alike —
getting that backwards transposes the map into diagonal streaks rather than
failing.

### 3.2 Image Classification — **shipped** at [`/image-classification`](../../frontend/src/routes/image-classification.tsx)

Taxonomy task **Image Classification** · built. Upstream research: ResNet-50,
ViT-B/16, ConvNeXt V2, DINOv2.

The simplest possible instance of the four-slot pattern, which is why it went
first: it is the page that established `src/vision/`, and every vision page after
it is a fraction of the cost.

| Model | Download (WebGPU · WASM) | Role |
|---|---|---|
| `Xenova/vit-base-patch16-224` | 173 MB fp16 · 88 MB q8 | the transformer baseline, and the default |
| `Xenova/resnet-50` | 51 MB fp16 · 26 MB q8 | the CNN baseline — a third of the download, close behind on accuracy |
| `onnx-community/mobilenetv4_conv_small.e2400_r224_in1k` | 7.6 MB fp16 · **15 MB fp32** | near-instant load, and the honest accuracy floor |

Sizes are read off the Hub's blob listing, not estimated. Two findings that only
a real load could produce, both now pinned by tests:

* **MobileNetV4 runs at fp32 on WASM, not q8** — see §1. `dtypes: { wasm: "fp32" }`
  in the catalogue, with measured `bytes`, because the params estimate would
  quote 3.8 MB for a 15 MB download.
* **`Xenova/mobilevitv2-1.0-imagenet1k-256` is not in the catalogue.** It
  publishes a single fp32 `model.onnx` and no fp16 or quantized export, so
  `loadOpts()` cannot resolve a file for either backend. `just fe-e2e-models`
  now checks the *files*, not just that the repo exists.

The page's one real design decision: **show the top five, never the argmax.** A
0.31 / 0.29 near-tie is the case worth seeing, and a single confident-looking
label is exactly what hides it. The margin between first and second is printed,
and a margin under 0.10 gets called out in words.

```ts
const clf = await pipeline("image-classification", "Xenova/vit-base-patch16-224",
                           { ...loadOpts(backend), topk: 5 });
const preds = await clf(image);      // [{ label, score }, ...]
```

Run it: `just fe-e2e-vision` loads MobileNetV4 for real and asserts the tiger
sample comes back as a tiger — the assertion that caught the quantization bug.

### 3.3 Object Detection — **shipped** at [`/object-detection`](../../frontend/src/routes/object-detection.tsx)

Taxonomy task **Object Detection** · built. Upstream research: DETR, YOLOS,
RT-DETRv2, D-FINE, RF-DETR.

The flagship live demo: this is the page where the app is visibly doing
real-time inference on the user's own GPU.

| Model | Download (WebGPU · WASM) | Role |
|---|---|---|
| `onnx-community/dfine_n_coco-ONNX` | 7.5 MB fp16 · 4.3 MB q8 | nano — the default, and the one for a webcam |
| `onnx-community/dfine_s_coco-ONNX` | 20 MB fp16 · 11 MB q8 | the localisation specialist, still live-capable |
| `Xenova/yolos-small` | 59 MB fp16 · 54 MB q8 | a plain ViT that detects |
| `onnx-community/rtdetr_r50vd` | 84 MB fp16, WebGPU only | the r18 variant has no export; r50 does |
| `Xenova/detr-resnet-50` | 80 MB fp16 · 41 MB q8 | the set-prediction original, slow and canonical |
| `Roboflow/rf-detr-*` | none | no ONNX export. Table only |

```ts
const det = await pipeline("object-detection", "onnx-community/dfine_n_coco-ONNX",
                           loadOpts(backend));
const out = await det(image, { threshold: 0.4, percentage: false });
drawBoxes(ctx, out);
```

Three things the route pins, each guarding a specific failure:

* **`percentage: false` lives in `useObjectDetector`, not in a caller.** The
  default returns 0–1 fractions, `drawBoxes` wants absolute pixels, and the wrong
  choice piles every box into the top-left corner. A unit test asserts the flag
  reaches the pipeline — a regression test for a named bug.
* **The threshold slider re-filters; it never re-runs.** The model is asked once
  at a deliberately low floor (`MODEL_THRESHOLD = 0.05`) and the slider derives
  the visible set from that list — the pure-derivation trick `/vad` uses. A
  slider that re-runs the model is a slider nobody drags.
* **Boxes are scaled back to the source** (`scaleDetections`). The frame is
  downscaled before inference, so coordinates return in the smaller frame's
  pixels while the canvas shows the original. Skipping the scale draws every box
  a constant fraction too small and too far top-left, which reads as a mediocre
  detector rather than as a bug in our arithmetic.

### 3.4 Image Segmentation — **shipped** at [`/segmentation`](../../frontend/src/routes/segmentation.tsx)

Taxonomy task **Image Segmentation** · built (semantic; DETR panoptic offered as
the one exception). Upstream research: SegFormer, Mask2Former, OneFormer,
EoMT-DINOv3.

Semantic segmentation ports cleanly. Instance and panoptic are thinner: only
the DETR panoptic head has an export; Mask2Former, OneFormer and EoMT have no
browser path, so the page states which of the three it is doing.

| Model | Class space | Download (WebGPU · WASM) |
|---|---|---|
| `Xenova/segformer-b0-finetuned-ade-512-512` | 150 ADE20K scene classes | 7.6 MB fp16 · 4.2 MB q8 — the default |
| `Xenova/face-parsing` | 19 face parts | 164 MB fp16 · 85 MB q8 |
| `mattmdjaga/segformer_b2_clothes` | 18 garment/body classes | **105 MB fp32, both backends** |
| `Xenova/detr-resnet-50-panoptic` | COCO things + stuff | 83 MB fp16 · 42 MB q8 |
| `shi-labs/oneformer_*`, `tue-mps/eomt-*`, Mask2Former | - | no export. Table only |

Sizes measured, and one of them changes a decision:

* **`mattmdjaga/segformer_b2_clothes` publishes only `onnx/model.onnx`** — no
  fp16, no quantized export — so `loadOpts()` cannot resolve a file and the load
  404s. It is pinned to `dtypes: { webgpu: "fp32", wasm: "fp32" }` with measured
  bytes, because the params estimate would quote 55 MB for a 105 MB download.
  This is the same situation that keeps `Xenova/mobilevitv2-1.0-imagenet1k-256`
  out of the classification catalogue; the difference is that pinning fp32 makes
  this one work. `just fe-e2e-models` now checks the files for *every* vision
  catalogue, per backend the entry claims.
* **`Xenova/face-parsing` is not small.** The research note called it "a strong
  small demo"; it is 340 MB fp32 / 164 MB fp16. Measured beats assumed.

```ts
const seg = await pipeline("image-segmentation", "Xenova/segformer-b0-finetuned-ade-512-512",
                           loadOpts(backend));
const masks = await seg(image);   // [{ label, score, mask: RawImage }, ...]
```

Each mask comes back as a single-channel `RawImage` — **one per class present**,
not an indexed label map. `drawMasks` composites them into one canvas with a
per-label colour; rendering 150 separate images is the failure mode it exists to
prevent. The class toggles and the opacity slider are pure derivations over the
masks already in hand, so neither re-runs the model, and `visibleMasks()` paints
the *smallest* class last so a 2%-coverage class is not buried under the sky.

### 3.5 Zero-Shot Image Classification — **shipped** at [`/zero-shot-image-classification`](../../frontend/src/routes/zero-shot-image-classification.tsx)

Taxonomy task **Zero Shot Image Classification** · built. Upstream research:
CLIP, OpenCLIP, SigLIP 2, MetaCLIP 2.

The most satisfying page in the category, because the user types the labels.

| Model | Scoring | Download (WebGPU · WASM) |
|---|---|---|
| `Xenova/clip-vit-base-patch32` | softmax over your labels | 289 MB fp16 · 147 MB q8 — the default |
| `Xenova/siglip-base-patch16-224` | independent sigmoid | 388 MB fp16 · 201 MB q8 |
| `onnx-community/siglip2-base-patch16-224-ONNX` | independent sigmoid | 716 MB fp16 · 360 MB q8 |
| `laion/CLIP-ViT-B-32-laion2B-*` | - | no usable export |

Sizes measured. Note that CLIP-B/32 is **289 MB fp16**, not the ~150 MB the
earlier research note quoted — that figure was the q8 download.

```ts
const zs = await pipeline("zero-shot-image-classification", "Xenova/clip-vit-base-patch32",
                          loadOpts(backend));
const out = await zs(image, ["a photo of a cat", "a photo of a dog", "an empty room"]);
```

**The prompt-template experiment shipped, and it is why this page exists** rather
than a fourth classifier. The route scores *both* wordings and puts them in
adjacent columns — a before/after the user has to hold in their head is not a
demonstration. Each template is its own call and its own softmax; concatenating
the two prompt sets into one call would make the wordings compete and mean
nothing.

**A trap worth knowing about, because it silently inverts the result.** The
`zero-shot-image-classification` pipeline applies its *own* default template,
`"This is a photo of {}"`, on top of whatever you hand it. A page that templates
its own prompts and does not pass `hypothesis_template: "{}"` therefore compares
"This is a photo of cat" against "This is a photo of a photo of a cat" — two
templates, neither of them the one on screen, with the double-templated column
losing. The first real run of this route measured exactly that (bare 0.860,
"templated" 0.842) and it read as a finding about prompt templates. It was a bug.

Labels are stored as **bare nouns** (`cat`, not `a cat`) for the same reason: the
template supplies the article, and phrases compose into "a photo of a a cat".

**Encode once, decode many is implemented** — `src/vision/zeroshot/`, and the one
place in the app that holds an inference cache.

The pipeline re-encodes the labels on every call. Label embeddings do not depend
on the image, so on a live feed that is the text tower — roughly 40% of CLIP's
work — re-run per frame to produce identical numbers. This task therefore owns
its engine rather than riding the generic vision worker, which is the criterion
§2 already stated: a task that is not a plain `pipeline()` call gets its own
engine, as `audio/enhance/` and `audio/vad/` do. `CLIPTextModelWithProjection`
and `CLIPVisionModelWithProjection` (`SiglipTextModel` / `SiglipVisionModel` for
the sigmoid pair) are driven separately, and the text side is kept.

The cache holds **several prompt sets, not one** (`TEXT_CACHE_LIMIT`). The page
scores two templates side by side, so a single-entry cache would be evicted on
every alternation and buy nothing on precisely the screen the feature exists for.
It is cleared whenever a different checkpoint loads — embeddings from another
model share no space with these.

| Family | Text output | Tokenizer padding | Final step |
|---|---|---|---|
| CLIP | `text_embeds` | to the longest prompt | `softmax(scale · cos)` |
| SigLIP | `pooler_output` | **`max_length`** | `sigmoid(scale · cos + bias)` |

SigLIP's padding is not a style choice: its position embeddings assume the full
width, and padding to the longest prompt returns different embeddings.

**What splitting the towers costs, and how that cost is controlled.** The full
`model.onnx` ends with normalise → matmul → scale → softmax. Running the towers
alone means owning those steps, so they live in
[`zeroshot/scoring.ts`](../../frontend/src/vision/zeroshot/scoring.ts) as pure
arithmetic. `scale` is `exp(logit_scale)` and `bias` is `logit_bias`, **read out
of each checkpoint's published weights** rather than guessed:

| Model | `exp(logit_scale)` | `logit_bias` | Source |
|---|---|---|---|
| CLIP ViT-B/32 | 100.000006 | - | `openai/clip-vit-base-patch32`, `pytorch_model.bin` |
| SigLIP base/16 | 117.330795 | -12.932437 | `google/siglip-base-patch16-224`, `model.safetensors` |
| SigLIP 2 base/16 | 112.668907 | -16.771725 | `google/siglip2-base-patch16-224`, `model.safetensors` |

CLIP's raw parameter is 4.605170, and `exp(4.605170) = 100.000006` — the ln(100)
clamp OpenAI trains against, confirmed from the weights rather than assumed.

**This arithmetic is the kind that fails silently**, which is why it is pinned by
a spec of its own. A wrong scale leaves every score in [0, 1], leaves the ranking
exactly as it was, and is simply not what the model said — no ordering assertion
can catch it, and the repo has shipped this shape of bug twice before in
`audio/enhance/`. `just fe-e2e-zeroshot` runs the real pipeline and the
split-tower path over the same checkpoint, image and prompts in one browser and
compares every label's score, via a harness that imports the **shipped** scoring
module rather than a copy of it.

It does one thing more, and it is the part that actually pins the constant. A
softmax is shift-invariant, so `ln(p_i) − ln(p_j) = scale · (cos_i − cos_j)`.
Feeding the *pipeline's* probabilities and *our* cosines into that identity
recovers the `logit_scale` the full graph is using, independently of what the
catalogue claims. Measured: **100.00017 and 100.00045** across two label pairs,
against a catalogue value of 100.000006. A plain agreement check would let a
scale of 90 through as a small offset; this would not.

**The comparison has to run at fp32, and finding out why was instructive.** At q8
the two paths disagree badly — 0.812 against 0.886 on the same label. The cause
is not the arithmetic: `model.onnx` and `text_model.onnx`/`vision_model.onnx` are
**separately quantized exports**, so their embeddings differ slightly, and a
softmax at scale 100 turns a ~0.001 cosine difference into a ~0.07 probability
difference. The tell was the implied scale coming out at 93.8 and 142.3 for the
two label pairs — a wrong constant would have produced one consistent wrong
number, so inconsistency pointed at the embeddings instead. At fp32 the paths
agree to six decimal places.

Two things follow. The spec runs at fp32 (~1.2 GB, hence opt-in). And the route,
which runs q8 on WASM, does **not** reproduce the numbers the pipeline would have
shown at q8 — both are valid quantizations of the same model, neither is the
reference, and the label ranking is unaffected on well-posed prompts. Worth
knowing before comparing a screenshot against a notebook.

The saving is reported in the UI (`encode-cost`): image milliseconds every run,
label milliseconds only on a miss. An optimisation nobody can see is an
optimisation nobody can check.

### 3.6 Zero-Shot Object Detection — **shipped** at [`/zero-shot-object-detection`](../../frontend/src/routes/zero-shot-object-detection.tsx)

Taxonomy task **Zero Shot Object Detection** · built. Upstream research: OWL-ViT,
OWLv2, Grounding DINO, LLMDet.

§3.5 without the fixed class list, §3.3 with localisation — the page where the
two halves of the category meet.

| Model | Queries read as | Download (WebGPU · WASM) |
|---|---|---|
| `Xenova/owlv2-base-patch16-ensemble` | candidate labels | 308 MB fp16 · 155 MB q8 — the default |
| `Xenova/owlvit-base-patch32` | candidate labels | 307 MB fp16 · 155 MB q8 |
| `onnx-community/grounding-dino-tiny-ONNX` | free-text phrases | 360 MB fp16 · 204 MB q8 |
| `iSEE-Laboratory/llmdet_tiny` | - | no export; server-side |

Sizes are ONNX blob totals read off the Hub, not a params estimate — OWLv2 carries
both towers in one graph and the estimate would be badly wrong. All three pass
`LARGE_MODEL_BYTES` on WebGPU, and the picker says so before anything downloads.

```ts
const owl = await pipeline("zero-shot-object-detection", "Xenova/owlv2-base-patch16-ensemble",
                           loadOpts(backend));
const out = await owl(image, ["a person", "a car"], { threshold: 0.02, percentage: false });
```

**The prompt is the model** — and here more literally than on §3.5. That page
templates a bare noun into `"a photo of a {}"`; this pipeline tokenizes
`candidate_labels` **verbatim**, so the text on screen *is* what the model is
asked, and the shipped defaults keep their articles (`"a person"`). Quietly
templating here would show one prompt and score another.

Three things this route settled:

- **The threshold starts at 0.1, not 0.4.** An open-vocabulary model spreads its
  probability over an unbounded label space, so OWLv2's confident hits land where
  a COCO detector's uncertain ones do. A closed-detector default renders an empty
  canvas over a picture full of correctly-found objects — which reads as a broken
  page rather than a badly-chosen number.
- **The slider re-filters; only editing a query re-runs.** Same pure derivation as
  §3.3 and `/vad`, and it matters more here: a re-run costs a 300 MB model a
  second of work.
- **Queries that found nothing keep their row.** "Which of my phrases matched
  nothing" is the question the page exists to answer, and a list that silently
  omits the misses answers it wrong. Grounding DINO replies with a *fragment* of
  the query rather than the query itself, so an unasked-for label gets its own
  heading instead of being dropped (`groupByQuery`).

The `@slow` spec asserts the asymmetry, not a count: `"a cat"` finds boxes on the
cats sample and `"a purple giraffe"` finds none. A detector that boxes everything
is as broken as one that boxes nothing, and only asking for something absent
tells them apart.

### 3.7 Mask Generation — **shipped** at [`/mask-generation`](../../frontend/src/routes/mask-generation.tsx)

Taxonomy task **Mask Generation** · built. Upstream research: SAM, SAM-HQ,
SAM 2.1, Grounded SAM.

The one task whose architecture was designed for exactly the interaction a web
page provides: **encode once, decode many.** The vision encoder runs once when
the picture is picked; every click after that decodes a mask in milliseconds.

| Model | Graphs | Download (WebGPU · WASM) |
|---|---|---|
| `Xenova/slimsam-77-uniform` | `vision_encoder` + `prompt_encoder_mask_decoder` | 21 MB fp16 · 14 MB q8 — the default |
| `onnx-community/sam2.1-hiera-tiny-ONNX` | same split, external weights | 78 MB fp16 · 62 MB q8 |
| `syscv-community/sam-hq-vit-base`, `facebook/sam3` | - | no export; server-side |

Sizes are the sum of **both** graphs' blobs, including the `.onnx_data`
external-weights sidecars SAM 2.1 uses. Quoting only the decoder — the small
half — would understate SlimSAM by 60% and SAM 2.1 by 85%.

```ts
// once per image: `get_image_embeddings` runs vision_encoder alone
const inputs = await processor(image);
const embeddings = await model.get_image_embeddings(inputs);

// once per click: passing the embeddings in makes `forward` skip the encoder
const input_points = processor.reshape_input_points([[[x, y]]],
                       inputs.original_sizes, inputs.reshaped_input_sizes);
const input_labels = new Tensor("int64", BigInt64Array.from([1n]), [1, 1, 1]);
const { pred_masks, iou_scores } = await model({ ...embeddings, input_points, input_labels });
const masks = await processor.post_process_masks(pred_masks,
                 inputs.original_sizes, inputs.reshaped_input_sizes);
```

This is the third module in the app to own an engine rather than ride a generic
pipeline worker (`audio/enhance/`, `audio/vad/`, `vision/zeroshot/` are the
others), and the criterion is the documented one: it is not a plain `pipeline()`
call. Five things it settled:

- **Two states, not one.** LOAD is the download; a separate *encoding* state is
  the per-image encoder pass — the slow half of the first click, and something
  the user has no way to guess is happening. The page shows both, and shows the
  decode time beside every mask, because sub-100 ms is a claim this page can
  simply prove.
- **The encode is its own id-correlated `run` request, not a progress event.**
  #17 sketched `{ status: "encoding" }` on the shared envelope. A progress event
  has **no failure path**: an encode that fails would either be swallowed or
  push Machine A to `error` and force a reload of weights that are perfectly
  fine. As a request it lands in Machine B, where the model stays loaded and the
  next picture still works — and the id keeps a decode from resolving against a
  superseded encode.
- **Passing `image_embeddings` in is what makes the split real.** `SamModel`'s
  `forward` computes them itself when they are missing, so omitting them still
  returns correct masks — at full encoder cost, on every click, silently. That
  is the one bug on this page that no output inspection can catch.
- **Clicks are converted to source pixels by the canvas** (`OverlayCanvas`'s
  `onPick`), not by the route. The canvas is sized to the source and scaled down
  by CSS, so a raw offset is wrong by the scale factor — and SAM answers a
  mis-mapped point with a perfectly plausible mask of whatever happens to be
  there. The `@slow` spec therefore asserts a **coverage band**: a mask covering
  ~0% or ~100% of the frame is exactly what a broken coordinate space produces,
  and both look fine until measured.
- **`input_labels` is int64.** `BigInt64Array`, matching the `ones()` default the
  model falls back to; a `Float32Array` fails inside ONNX Runtime with a shape
  error naming neither the tensor nor the cause.

All three candidate masks are offered, not just the winner: a single point is
ambiguous by construction — the wheel, the door, or the whole car — and SAM
returns one mask per reading rather than guessing. Point markers are drawn over
the *input* (`ImageSourcePanel`'s `preview` override exists for this one case),
because they are input; the mask is the result and lives in OUTPUT.

### 3.8 Image Feature Extraction — **shipped** at [`/image-features`](../../frontend/src/routes/image-features.tsx)

Taxonomy task **Image Feature Extraction** · built. Upstream research: DINOv2,
CLIP, SigLIP 2, DINOv3.

No visible output of its own, which makes the page a similarity search: embed a
twelve-picture gallery in the tab, embed the query, show the nearest neighbours.
The whole index lives in memory, so it is genuinely private.

| Model | Vector | Download (WebGPU · WASM) |
|---|---|---|
| `Xenova/dinov2-small` | CLS or mean of patches, 384-d | 44 MB fp16 · 24 MB q8 — the default |
| `onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX` | CLS or mean, 384-d | 22 MB q8 on **both** backends |
| `Xenova/dinov2-base` | CLS or mean, 768-d | 173 MB fp16 · 91 MB q8 |
| `Xenova/clip-vit-base-patch32` | one projected 512-d vector | 176 MB fp16 · 89 MB q8 (vision tower only) |

```ts
const feat = await pipeline("image-feature-extraction", "Xenova/dinov2-small", loadOpts(backend));
const tokens = await feat(image);            // Tensor [1, 1 + patches, dim]
```

Four things this route settled, and two of them contradict the plan in #18 —
in both cases because a measurement said so.

- **The pipeline's option is `pool`, a boolean, not `{ pooling, normalize }`.**
  That signature belongs to the *text* feature-extraction pipeline. The image
  one returns `last_hidden_state` when `pool` is falsy and `pooler_output` when
  it is true, and it never normalises. So the normalising is ours
  (`vision/similarity.ts`), and the page displays the pre-normalisation L2 norm
  next to the unit-length result — "the vectors are normalised" is a claim, and
  this is the page that can simply show it.
- **The pooling toggle re-ranks; it does not re-embed.** One forward pass returns
  every token row, so `poolEmbedding` derives CLS *and* the mean of the patches
  from the same pass. #18 assumed the toggle would re-run the model; over a
  twelve-image gallery that is the difference between instant and a quarter of a
  minute, and the shipped behaviour follows the same pure-derivation rule as the
  detection and VAD thresholds. Only the **checkpoint** invalidates the index.
- **CLIP is in the catalogue, and it offers no pooling choice.**
  `AutoModelForImageFeatureExtraction` resolves CLIP to
  `CLIPVisionModelWithProjection`, which loads `vision_model.onnx` (not
  `model.onnx`) and — verified by reading the export's output names — emits
  `image_embeds` alone. So the page gets one projected, language-aligned 512-d
  vector and says there is nothing to choose, rather than showing a dead toggle.
  `VisionModel.graphs` is what tells `just fe-e2e-models` to check the right file.
- **DINOv3 publishes no fp16 export** — fp32, q4 and q8 only — so its precision is
  pinned to q8 on both backends. Left to `loadOpts()`, WebGPU would ask for a
  file that does not exist. Same class of failure as the missing `mobilevitv2`
  export, and `just fe-e2e-models` is what catches it.

The gallery is twelve pictures chosen as **clusters** (three animals, three city
scenes, two portraits, two landscapes, two objects), because a similarity search
over a set with no near neighbours has nothing to show. The `@slow` spec queries
with the tiger — which is deliberately *not* in the gallery — and asserts the top
neighbour is an animal. A mis-pooled or unnormalised vector destroys exactly that
ordering while still producing a full, plausible-looking list.

### 3.9 Image to Text — **shipped** at [`/image-to-text`](../../frontend/src/routes/image-to-text.tsx)

Taxonomy task **Image to Text** · built. Upstream research: BLIP, Florence-2,
SmolVLM2, GOT-OCR 2.0.

The first vision route with a **generative decoder**: runs take seconds, not
milliseconds, and there is deliberately no live camera mode.

| Model | Modes | Download (WebGPU · WASM) |
|---|---|---|
| `onnx-community/Florence-2-base-ft` | caption · detailed caption · OCR · grounding | 544 MB fp16 · 275 MB q8 — the default, **WebGPU only** |
| `Xenova/vit-gpt2-image-captioning` | caption | 482 MB fp16 · 246 MB q8 |
| `HuggingFaceTB/SmolVLM-256M-Instruct` | - | needs a chat template; that is Image-Text-to-Text, not this row |
| `Salesforce/blip-*` | - | the only mirror ships `split_0/1.onnx`, not the transformers.js layout |
| `stepfun-ai/GOT-OCR-2.0-hf` | - | no export; Florence-2 covers OCR instead |

**The plan in #18's sibling — #19 — assumed `pipeline("image-to-text")` with a
task token in the run payload. It cannot work, and the reason is worth
recording.** `ImageToTextPipeline._call` does exactly two things: run the
processor for `pixel_values`, and call `model.generate({ inputs })`. There is
nowhere to put a task token. Worse, it resolves its model through
`AutoModelForVision2Seq`, whose registry maps `vision-encoder-decoder`,
`idefics3` and `smolvlm` — Florence-2's model type is `florence2`, which lives in
the *image-text-to-text* mapping, so the pipeline cannot load it at all. Same
shape of finding as MusicGen needing `MusicgenForConditionalGeneration` rather
than the `text-to-audio` pipeline.

So `vision/caption/` owns an engine, with two implementations behind one
`Captioner` interface — the precedent being `tts.worker.ts`, which owns Kokoro,
MMS and MusicGen behind one `TtsSynthesizer` for the same reason.

```ts
const prompts = processor.construct_prompts("<OCR>");   // → "What is the text in the image?"
const inputs = await processor(image, prompts);
const ids = await model.generate({ ...inputs, max_new_tokens: 512 });
const text = processor.batch_decode(ids, { skip_special_tokens: false })[0];
const answer = processor.post_process_generation(text, "<OCR>", [image.width, image.height]);
```

Four details in those five lines are silent when wrong:

- **`skip_special_tokens: false`.** Florence-2's box answers *are* special tokens
  — `<loc_412>` per coordinate — so stripping them returns a caption with the
  boxes quietly deleted rather than an error.
- **`image_size` is `[width, height]`.** `post_process_generation` maps
  coordinates with `image_size[i % 2]` over an `x, y, x, y` sequence, so it wants
  width first. Its own JSDoc says "height x width", and `inputs.original_sizes`
  *is* `[height, width]` — passing that transposes every box on a non-square
  picture. Use `RawImage.size`.
- **The prompt is text, not the token.** `construct_prompts` resolves `<OCR>`
  against the model's own `preprocessor_config.json`. Sending the raw token
  tokenizes it as literal characters and the model answers something fluent and
  unrelated.
- **A task token a model has never seen does not error.** It produces a
  confident, fluent, wrong sentence. So the catalogue carries per-model
  capability flags, the UI offers exactly those, and switching model resets an
  unsupported mode rather than sending it.

**Florence-2 is gated to WebGPU by the picker, not left to fail at load.** That
required actually consuming `VisionModel.backends`, which every catalogue had
been free to declare since Wave 0 and which nothing read — `model/useBackendProbe.ts`
answers "what would this load on" before anything downloads, and `ModelPicker`
disables a model the machine cannot run, with the reason on the row. A `null`
probe (not yet answered) gates nothing: treating the undecided state as WASM
greys out every WebGPU model for a frame on each page load.

The `@slow` spec runs in the `webgpu` project and asserts three things — a
plausible caption on the tiger, **`coca|cola` read off the bundled
advertisement**, and grounding rendering as a canvas rather than as prose. The
OCR substring is the one that catches a broken processor path: a model handed
mis-normalised pixels still produces fluent text.

### 3.10 Keypoint Detection — **shipped** at [`/pose`](../../frontend/src/routes/pose.tsx)

Taxonomy task **Keypoint Detection** · built. Upstream research: ViTPose,
Sapiens2, SuperPoint, LightGlue.

Top-down pose is **two models** — detect people, then run the pose model on each
person's crop — which makes this the one page that deliberately holds two models
live, and the documented exception to §5.

| Pair | Combined download (WebGPU · WASM) |
|---|---|
| `onnx-community/dfine_n_coco-ONNX` + `onnx-community/vitpose-base-simple` | 180 MB fp16 · 92 MB q8 — the default, and the live one |
| `onnx-community/rtdetr_r50vd` + `onnx-community/vitpose-base-simple` | 260 MB fp16 · 133 MB q8, still images only |
| `facebook/sapiens2-pose-0.4b` | 308 keypoints, no export |
| `magic-leap-community/superpoint`, `ETH-CVG/lightglue_superpoint` | interest points / matching stay server-side |

**The catalogue entry is the pair, not either half**, and it quotes the
**combined** download — a guardrail that quotes half the bytes is worse than
none. It is also the case that makes "fires on the sum" a real distinction: the
RT-DETR pair is 88 MB of detector plus 172 MB of pose model, each comfortably
under `LARGE_MODEL_BYTES` and 260 MB together, over it.

```ts
const detections = await detector(image, { threshold: 0.05, percentage: false });
const boxes = personBoxes(detections, { threshold, maxPeople })
  .map((d) => cropBox(d.box, image.width, image.height));   // COCO [x, y, w, h]
const crops = await Promise.all(boxes.map(([x, y, w, h]) => image.crop([x, y, x + w, y + h])));
const { heatmaps } = await poseModel(await processor(crops));
const poses = processor.post_process_pose_estimation(heatmaps, boxes.map((b) => [b]));
```

**The coordinate round-trip is the whole correctness surface of this page, and
the trap is inside `post_process_pose_estimation`:**

```js
const xScale = bbox.at(-2) / width;                    // box *width* / heatmap width
const keypoint = [(xScale * xWeightedSum) / sum, …];   // …and no origin
```

It scales the heatmap peak by the box's **size** and never adds the box's
**origin**. Pass the whole image as the box — as the single-person example
upstream ships does — and the origin is (0, 0), so the omission is invisible.
Pass a crop and every joint is offset by the crop's top-left corner: the skeleton
floats beside the person, drawn confidently, and every count-based test passes.
`pose.ts` owns that addition, it is pure, and it is asserted against
hand-computed values; the `@slow` spec then makes the coarse anatomical check —
**the nose must be above the ankles** — which is the only kind of assertion that
catches this.

Four other things this route settled:

- **`model/progress.ts` had to be keyed on repo + file.** Both checkpoints
  publish an `onnx/model_fp16.onnx`, so keyed on the file name the second
  model's bytes overwrote the first's: the denominator became one model's size,
  the bar reached 100% halfway through, and the second download read as a stall.
  Both models are also *loaded together* (`Promise.all`) so their progress
  events interleave and the aggregate has both denominators from the start.
- **The threshold and the people cap change the work, not the view.** The
  opposite of §3.3's slider, and deliberately: filtering afterwards would mean
  running the pose model on people the user has already excluded, and that pass
  is the expensive half. The page says so rather than looking inconsistent.
- **Both models are disposed, with `Promise.allSettled`.** A detector whose
  teardown throws must not skip the pose model's, which is the larger of the
  two — this is the one place a half-finished teardown leaks 170 MB.
- **A low-confidence joint is dimmed, never hidden and never drawn boldly.** A
  heatmap always has a maximum somewhere, so an occluded ankle comes back as a
  confident-looking guess rather than as an absence. A limb is drawn at the
  *lower* of its two joints' confidences: averaging would let one solid joint
  carry a guessed one into looking certain.

### 3.11 Video Classification — **shipped** at [`/video-classification`](../../frontend/src/routes/video-classification.tsx), **as a frame-level baseline**

Taxonomy task **Video Classification** · built, *and labelled for what it is*.
Upstream research: VideoMAE, TimeSformer, X-CLIP, V-JEPA 2.

**None of the four video transformers has an ONNX export.** They attend across
time as well as space, and nobody has shipped a browser-runnable one. What ships
here is the control experiment: sample frames, score each with CLIP zero-shot,
pool over a sliding window. The page's thesis is the question *does motion
actually matter*, answered from the other direction — this is what you get when
the model can only see single frames.

| Model | Role |
|---|---|
| `Xenova/clip-vit-base-patch32` and the rest of §3.5's catalogue | scores each sampled frame; reused wholesale |
| VideoMAE · TimeSformer · X-CLIP · V-JEPA 2 | no export; a real temporal model stays server-side |

**The labelling is the correctness requirement, not decoration.** The limitation
is in the page title's description, next to the result, and asserted by an E2E
spec — which is unusual and deliberate. Shipping this as "Video Classification"
without the framing teaches something false.

Three things it settled:

- **"Encode once, decode many" applies twice on this page.** The label
  embeddings are constant across every frame, so §3.5's text cache computes them
  once per label edit rather than once per frame — on a 60-frame clip, the
  difference between seconds and a minute. And the **pooling window re-derives**:
  smoothing is pure (`pool.ts`) over scores already in hand, so moving the
  slider redraws the chart without re-scoring the clip, which would be N CLIP
  passes.
- **Sampling seeks, it does not play.** Setting `currentTime` and awaiting
  `seeked` decodes only the frames asked for; a 30-second clip must not take 30
  seconds to sample. Two caps, both stated in the UI: the *rate* is the cost dial
  and the *count* is the hard stop, because this is the page most likely to be
  handed a ten-minute video and without the second cap that is not slow, it is a
  hung tab.
- **Frames are sampled at half-step offsets, not from zero.** The first frame of
  a clip is very often black or a fade, and a baseline whose first data point is
  "an image of darkness" reads as a model failure rather than an editing
  convention.

The sliding mean is also the *honest* pooling for this page: the route has no way
to know a frame follows the one before it, and averaging a neighbourhood is the
most a model that cannot see motion is entitled to do with time. The clip-level
verdict is computed from the **unpooled** series — averaging an already-averaged
series would weight the middle of the clip more heavily than its ends for no
defensible reason.

### 3.12 The tasks that stay on a server

Nine tasks, and each has a structural reason rather than a missing export. Six
keep their `/tasks/$slug` placeholders outright; **Image to Image and Image to 3D
now have routes covering the single-pass part of each**, and the generative part
of both stays here.

| Taxonomy task | Why not in a tab |
|---|---|
| Text to Image | SD 1.5 is 1.7 GB at fp16 before the text encoder, and needs 20 to 50 UNet passes. SDXL-Turbo cuts the steps but not the weights |
| Image to Image | same weights as above. **Super-resolution is the exception and now ships** — §3.14 |
| Image to Video | SVD and LTX are multi-gigabyte, and decode a 3-D VAE per clip |
| Unconditional Image Generation | DDPM at 1000 steps is a server job even at 32x32. Consistency models are the only part that comes close |
| Text to Video | at the edge of a 12 GB card, let alone a tab |
| Text to 3D | Shap-E is a diffusion model plus a NeRF decoder |
| Image to 3D | Zero123++ is SD-derived. **The depth-to-point-cloud half is the exception and now ships** — §3.15 |
| Video to Video | per-frame diffusion, so the cost of Image to Image multiplied by the frame count |
| Grounded SAM, OneFormer, RF-DETR | model-specific: no export exists yet, so they are missing *variants* rather than missing tasks |

Three carve-outs hid inside that list, and **all three now ship**. Each covers a
*part* of its slug, and each page says which part — a route that quietly answered
a smaller question than its name promises would be the worse outcome.

- **Super-resolution runs in the browser** — §3.14,
  [`/super-resolution`](../../frontend/src/routes/super-resolution.tsx),
  [#22](https://github.com/bthek1/model_playground/issues/22). One forward pass
  per tile, no diffusion. Mapped onto the **Image to Image** slug; editing and
  img2img stay in the table above.
- **The depth-to-point-cloud half of Image to 3D runs in the browser** — §3.15,
  [`/image-to-3d`](../../frontend/src/routes/image-to-3d.tsx),
  [#23](https://github.com/bthek1/model_playground/issues/23). It adds no model
  at all: §3.1's checkpoint plus arithmetic, rendered through hand-written WGSL.
  Zero123++ and full reconstruction stay in the table.
- **Background removal** — §3.13,
  [`/background-removal`](../../frontend/src/routes/background-removal.tsx),
  [#24](https://github.com/bthek1/model_playground/issues/24). The one page here
  with **a taxonomy row of its own that the Hub does not have**, and the one with
  a licence constraint worth stating in the sidebar-level record. Both calls are
  in §3.13.

### 3.13 Background Removal — **shipped** at [`/background-removal`](../../frontend/src/routes/background-removal.tsx)

The Wave 3 page with no Hub task behind it, and the only one in this category
whose blocking question was a **licence** rather than an export.

**The licence call, settled before the route was written**
([#24](https://github.com/bthek1/model_playground/issues/24) Phase 1).
`briaai/RMBG-1.4` is the better matte — visibly so on hair and fur — and it ships
under `bria-rmbg-1.4`, a Creative Commons licence for **non-commercial use only**,
with commercial use gated behind a paid agreement with BRIA. This repo is MIT.
Nothing here redistributes weights either way (the browser fetches them from the
Hub), but a default that most downstream users may not legally use is still a
trap. So:

- **`Xenova/modnet` (Apache-2.0) is the default.** It is also Transformers.js's
  own default for this pipeline, and 6.6 MB on WASM.
- **RMBG-1.4 is offered and labelled.** `components/vision/LicenceNote.tsx`
  renders the restriction in the same amber as the size-before-load guardrail,
  in the SELECT slot, at the moment the model is chosen.

**The taxonomy call.** The Hub has no `background-removal` task — Transformers.js
invented the pipeline as a subclass of image segmentation. Adding a row is
therefore a deliberate departure from the taxonomy's mirror of the Hub, and it
was taken: the page stands alone as useful, and an unlisted route is one nobody
finds. That is why the category is twenty rows rather than nineteen.

**The soft matte survives, and that is not something the pipeline guarantees.**
`ImageSegmentationPipeline` picks its post-processing by looking for a
`post_process_*_segmentation` method on the processor. Find one and it takes the
semantic branch, which is an **argmax**: every pixel becomes 0 or 255 and the
soft edge — the entire point of a matting model — is gone, silently. Both
entries publish a plain `ImageFeatureExtractor`, which has no such method, so the
pipeline falls through to the `!subtask` branch and returns a genuine 0–255
matte. **A matting checkpoint whose repo happened to ship a
`SegformerImageProcessor` config would be hard-thresholded and would still look
plausible.** Check `preprocessor_config.json` before adding a third entry.

Everything downstream of that is ours and lives in `vision/matte.ts`: the
composite is a real linear blend (`src * a + bg * (1 - a)`), the raw matte is
offered as its own view so the edge can be judged rather than assumed, and the
download is a PNG because it is the only common format with an alpha channel.
The unit test that matters asserts a **mid-alpha pixel blends** — a hard
threshold passes the fully-opaque and fully-transparent cases and fails only that
one.

**MODNet is a *portrait* matting model, and asked for anything else it does not
fail loudly.** The `@slow` spec measured **0.2% coverage** on the beetle
photograph: a near-empty matte, which renders as a clean, empty checkerboard and
looks like a page that has not run yet. `PORTRAIT_SAMPLES` exists because of that
measurement, the route lists it first, and the sample hint names which model
suits which picture. The spec now asserts a **coverage band** on a portrait, for
the same reason `/mask-generation` asserts one: all-on and all-off are what a
broken preprocessing path produces, and both render beautifully.

| Model | Licence | Notes |
|---|---|---|
| `Xenova/modnet` | Apache-2.0 | 6.5M params, portrait matting, **the default** |
| `briaai/RMBG-1.4` | `bria-rmbg-1.4`, non-commercial | 44M params, general-purpose, a better edge |

---

### 3.14 Super Resolution — **shipped** at [`/super-resolution`](../../frontend/src/routes/super-resolution.tsx), as the Image to Image carve-out

`Xenova/swin2SR-classical-sr-x2-64`, 12M params, Apache-2.0 upstream. One forward
pass per tile and no sampling loop at all, which is the whole reason this half of
Image to Image runs in a tab when the other half cannot. **The page says so in
its header** — the slug promises editing, and the route delivers upscaling.

**Tiling is the correctness surface, and it fails silently.** A transformer on a
full-resolution photo exhausts memory, so the image is cut into overlapping tiles
and stitched back. Butt them edge to edge and every seam is a visible line;
overlap them and feather each contribution to nearly zero across the overlap, and
they vanish. `vision/tile.ts` is pure arithmetic over plain buffers for exactly
this reason — the failure mode is geometric, and geometry is what a unit test can
pin and a rendered picture cannot. Three details earned their comments:

- **Normalise by accumulated weight**, not by trusting the weights to sum to one.
  That is what makes the identity case *exact*: run tiles through a model that
  returns them unchanged and reassembly reproduces the input pixel for pixel,
  which is the test that catches a wrong stride or a mis-placed tile.
- **`+ 0.5` in the feather.** Without it the outermost column of an interior seam
  weighs exactly zero, two tiles both contribute nothing, and the division paints
  a black line precisely where the blending was meant to hide one.
- **`Swin2SRImageProcessor` pads the input up to a multiple of 8** and upscales
  what it was given, so a tile can come back *larger* than `2 x tile`. Only the
  top-left region is read. Without that crop every subsequent tile lands slightly
  off, which reads as a soft, doubled image rather than as a bug.

**The comparison is the output.** A 2x image on its own proves nothing — every
upscaler produces one, and the eye has no reference. `CompareSlider` puts the
result under a draggable split against a **bicubic upscale of the same input**,
both rasterised at source resolution and revealed with `clip-path` so neither
half is resampled by the comparison itself.

**A run is many inferences, so the page quotes the cost first and offers Stop.**
`MAX_SOURCE_SIDE` is 512 rather than 1024 because tiles grow with *area*: 512 px
is about 6 tiles, 1024 px about 35, and on WASM that is fifteen seconds against a
minute and a half for the same demonstration. `useSuperRes` reports `{done,
total}` derived from the loop rather than through a new worker message, and its
`running` covers the whole sequence — the pipeline's own inflight count drops to
zero between tiles and would flicker the transport thirty times per upscale.

**`dtypes: { wasm: "fp32" }` started as a precaution and is now a measurement.**
`just fe-e2e-superres` upscales a 320x320 crop to 640x640 and scores it by PSNR
against the crop it was downscaled from, alongside a bicubic resize of the same
input:

| WASM precision | model | bicubic | per tile |
|---|---|---|---|
| fp32 | **27.80 dB** | 27.55 dB | 32–40 s |
| q8 | 27.39 dB | 27.55 dB | 24.2 s |

**At q8 the model loses to bicubic** — it is worse than not running at all — for a
25% saving in time and 33 MB in download. That is §3.2's quantized-MobileNetV4
failure in a different guise: dense regression puts int8 error straight into the
picture rather than letting an argmax absorb it. The WebGPU path is *not* covered
by that measurement; the spec runs on WASM.

Two things came out of the same run. `MS_PER_TILE.wasm` was 2,500 — out by more
than a factor of ten, which made the size guard actively misleading, promising 23
seconds for a run that took 273; it is 35,000 now, the middle of the observed
range. And `MAX_SOURCE_SIDE` moved from 1024 to 512:
tiles grow with *area*, so that is 4–9 tiles instead of 35, and it is the
difference between a demonstration and an abandoned tab.

---

### 3.15 Image to 3D — **shipped** at [`/image-to-3d`](../../frontend/src/routes/image-to-3d.tsx), as the depth-to-point-cloud carve-out

**The cheapest impressive demo in this file, and it adds no model at all**: §3.1's
Depth Anything V2, unprojected. It is also the one route where the hand-written
WGSL runtime and the Transformers.js runtime appear on the same page — see
[`docs/explanations/webgpu-inference.md`](../explanations/webgpu-inference.md) for
how they coexist. They do not mix: inference stays in `src/vision/` and produces a
plain `Float32Array`, rendering stays in `src/webgpu/` and never knows a model
exists, and the route is where they meet.

**The focal length is an assumption, and the page says so next to the slider.**
Depth Anything V2 predicts relative depth with no camera intrinsics attached — it
cannot know the lens that took the picture — so `f` is a control with a plausible
default (0.8x the long side, about a 64° horizontal field of view) and the
geometry is *plausible rather than metric*. Two photos' clouds are not comparable.
Depth Pro is the exception, which is why `unproject` takes `inverse` as a flag
read off the catalogue entry rather than assuming a convention.

**Two things in `vision/pointCloud.ts` fail silently, and the unit tests exist
for them specifically.** Both were caught during the build:

- **Inverse depth: a big value means *near*, so distance is its reciprocal.** The
  first implementation had it backwards. The result is a recognisable point cloud
  that is inside out — near things land far away and a scene reads as a bowl —
  and nothing throws. The test asserts the *ordering* of two pixels, not a count.
- **Bounds must be read back from the float32 buffer, not from the doubles that
  produced them.** A double rounded into a `Float32Array` can land a hair outside
  a bound computed before the write, and a consumer clamping to those bounds
  would drop a point.

A third choice is worth recording: the far plane is floored at `MIN_INVERSE = 0.1`,
giving a 10:1 depth range. Without a floor a normalised map's zero pixel makes the
reciprocal unbounded, the bounds put the whole visible scene in a speck, and the
canvas reads as "the model failed" rather than as a scaling choice.

**Rendering is instanced quads, not `point-list`.** WebGPU's point primitive is
always exactly one pixel — there is no `gl_PointSize` — and a few hundred thousand
one-pixel points on a high-DPI backing store read as faint noise. Each point is an
instance of a two-triangle quad, offset in clip space and scaled by `w` so it
keeps a constant size on screen. Depth testing is not optional either: without it
the points draw in buffer order, the back of the scene paints over the front, and
the result looks like fog.

**The vertex buffer is written once per inference; the camera once per frame.** A
cloud is megabytes, so orbiting rewrites 48 bytes of uniform rather than
re-uploading it — the difference between an orbit that tracks the pointer and one
that stutters. Both sliders (focal length, point density) **re-derive from the
cached depth map and never re-run the model**, the same pure-derivation rule
`/vad` applies to its threshold and §3.3 to its score floor.

**It degrades honestly.** `detectWebGPU()` never throws, so the failure is a
status rather than an exception — and a page that answered it with an empty canvas
would look like the model had failed rather than like the machine lacking a
device. On a non-`ready` status the route shows the depth map and says the 3-D
view needs WebGPU, warns in the LOAD slot *before* the download, and an E2E spec
asserts that path in a real Chromium with no GPU.

---
---

## 4. Feasibility summary

| Taxonomy task | In-browser? | Recommended model | Best backend | If not |
|---|---|---|---|---|
| **Depth Estimation** | **Shipped** — `/depth` | `onnx-community/depth-anything-v2-small` | WebGPU | DepthPro gated for metric |
| **Image Classification** | **Shipped** — `/image-classification` | `Xenova/vit-base-patch16-224` | WebGPU / WASM | - |
| **Object Detection** | **Shipped** — `/object-detection` | `onnx-community/dfine_n_coco-ONNX` | WebGPU | RF-DETR to server |
| **Image Segmentation** | **Shipped** — `/segmentation` (semantic) | `Xenova/segformer-b0-finetuned-ade-512-512` | WebGPU | Mask2Former / OneFormer to server |
| **Image to Text** | **Shipped** — `/image-to-text` | `onnx-community/Florence-2-base-ft` | WebGPU | GOT-OCR to server; SmolVLM is Image-Text-to-Text |
| **Video Classification** | **Shipped** — `/video-classification`, frame-level baseline | CLIP over sampled frames | WebGPU / WASM | real video transformers to server |
| **Zero Shot Image Classification** | **Shipped** — `/zero-shot-image-classification` | `Xenova/clip-vit-base-patch32` | WebGPU / WASM | - |
| **Mask Generation** | **Shipped** — `/mask-generation` | `Xenova/slimsam-77-uniform` | WebGPU / WASM | SAM-HQ, SAM 3, Grounded SAM to server |
| **Zero Shot Object Detection** | **Shipped** — `/zero-shot-object-detection` | `Xenova/owlv2-base-patch16-ensemble` | WebGPU / WASM | LLMDet to server |
| **Image Feature Extraction** | **Shipped** — `/image-features` | `Xenova/dinov2-small` | WebGPU / WASM | - |
| **Keypoint Detection** | **Shipped** — `/pose` | `dfine_n_coco` + `vitpose-base-simple` | WebGPU / WASM | Sapiens2, SuperPoint to server |
| **Image to Image** | **Shipped** — `/super-resolution`, SR only | `Xenova/swin2SR-classical-sr-x2-64` | WebGPU / WASM | editing / img2img diffusion to server |
| **Image to 3D** | **Shipped** — `/image-to-3d`, depth-to-cloud only | Depth Anything V2 plus WGSL | WebGPU | Zero123++ and full reconstruction to server |
| **Background Removal** | **Shipped** — `/background-removal` | `Xenova/modnet` (Apache-2.0) | WebGPU / WASM | RMBG-1.4 offered, non-commercial licence |
| **Text to Image** | No | - | - | server API |
| **Image to Video** | No | - | - | server API |
| **Unconditional Image Generation** | No | - | - | server API |
| **Text to Video** | No | - | - | server API |
| **Text to 3D** | No | - | - | server API |
| **Video to Video** | No | - | - | server API |

Rule of thumb: **one forward pass runs in a tab; fifty do not.** Every "no" in
that table is a diffusion model, and every "yes" is a single-pass network — and
`/super-resolution`, which runs one pass per tile, is the case that shows the rule
is about the *sampling loop* rather than about the number of inferences.

---

## 5. Memory and performance notes

The research these models come from targets 12 GB of VRAM and frees memory
aggressively. A tab is tighter, and the same discipline applies with different
mechanics.

- **One model live at a time**, with `/pose` as the single deliberate exception
  (§3.10 — top-down pose is a detector plus a pose model, and neither half is
  useful alone). `await model.dispose()` after nulling the reference. There is
  no `torch.cuda.empty_cache()`, so disposing is the entire mechanism. The pose
  engine disposes both halves with `Promise.allSettled`: a detector whose
  teardown throws must not skip the pose model's, which is the larger of the two.
- **A model the machine cannot run is not offered.** `VisionModel.backends` was
  declarable from Wave 0 and read by nothing until `/image-to-text` needed it;
  `model/useBackendProbe.ts` now answers "what would this load on" before
  anything downloads, and `ModelPicker` disables the row with the reason on it
  rather than letting a 275 MB download fail.
- **Resolution is the throttle, not the model.** A detector at 640x480 and the
  same detector at 1280x720 differ by roughly 4 times in cost. Downscale the
  frame before inference and draw the boxes on the full-size canvas.
  `/super-resolution` is the sharpest case: its cost is *tiles*, tiles grow with
  area, and `MAX_SOURCE_SIDE` is 512 rather than 1024 because that is 6 tiles
  instead of 35 for the same demonstration.
- **A GPU buffer is written once per result, not once per frame.** `/image-to-3d`
  uploads a multi-megabyte cloud when the inference finishes and rewrites 48
  bytes of camera uniform while the user orbits. Re-uploading per frame would
  spend the whole budget moving data that did not change.
- **Never queue frames.** One in flight at a time, as in section 2. A backlog is
  the reason a live demo feels laggy long before the model is the reason.
- **Encode once, decode many** wherever the architecture allows it, and it is now
  implemented in all three places it applies:
  - `/zero-shot-image-classification` (`src/vision/zeroshot/`) drives the two
    towers separately and caches several label sets at once — see §3.5,
    including what owning the final normalise/scale/softmax costs and the
    parity spec that controls it.
  - `/mask-generation` (`src/vision/sam/`) is the architectural case: the vision
    encoder runs once per image and every click decodes from the cached
    embedding. §3.7 — and note that `SamModel.forward` silently re-encodes when
    the embeddings are not passed in, which is the one bug on that page no
    output inspection can catch.
  - `/video-classification` reuses the zero-shot text cache across every frame
    of a clip. §3.11.
- **A control that changes the *view* re-derives; only one that changes the
  *question* re-runs.** Detection's threshold, segmentation's opacity, `/vad`'s
  threshold, `/background-removal`'s backdrop and matte view, and
  `/image-to-3d`'s focal length and point density are all pure derivations over a
  result already in hand. Each has a test asserting `run` was not called again.
- **Warm up on load.** The first inference compiles WebGPU shaders. On a
  detector that is 2 to 4 seconds the user should not be charged for.
- **Weights cache after the first download**, so the second visit is instant and
  works offline. `model/cache.ts` probes that cache, which is what lets the picker
  say "already downloaded" and a refresh resume without asking again.

---

## 6. Reference

- **Transformers.js pipelines used here**: `depth-estimation`,
  `image-classification`, `object-detection`, `image-segmentation`,
  `zero-shot-image-classification`, `zero-shot-object-detection`,
  `image-feature-extraction`, `image-to-text`, `mask-generation`,
  `background-removal`, `image-to-image`.
- **onnxruntime-web** for the models with no pipeline wrapper.
- **Raw WGSL** for one thing only: `/image-to-3d`'s point-cloud render pass
  (`webgpu/shaders/points.wgsl` + `webgpu/pointRenderer.ts`) — the first *render*
  pipeline in this repo, every other shader here being compute.
- **Page construction**: [`docs/guides/adding-a-task-page.md`](../guides/adding-a-task-page.md).
- **In-repo standards**:
  [`docs/standards/model-page-pattern.md`](../standards/model-page-pattern.md)
  (the four-slot contract),
  [`docs/standards/model-visualization.md`](../standards/model-visualization.md)
  (canvas heatmaps, schematics, theme tokens — read before drawing anything),
  [`docs/guides/adding-a-model.md`](../guides/adding-a-model.md) §8.
- **The shipped precedent**: the six audio routes, mapped in
  [`docs/roadmaps/audio.md`](./audio.md), and the fourteen vision routes in this
  category. A vision page is the same worker protocol with an image payload
  instead of a `Float32Array` — and, on the way back, a result that has been
  through `toCloneable`.
- Model recommendations here come from a companion collection of Python
  notebooks, which is a separate project and not a dependency of this repo. Every
  browser id above was checked against the Hugging Face API — re-check with
  `just fe-e2e-models` once a catalogue module exists to check.

