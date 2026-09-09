# Computer Vision Models in the Browser (WebGPU or CPU)

> The **Computer Vision** category of `components/layout/taskTaxonomy.ts`, task by
> task: what runs client-side on the user's GPU (WebGPU) or CPU (WebAssembly),
> which checkpoint to use, and which tasks stay on a server. No Python server in
> the inference path.

**Five of nineteen are built.** [`/image-classification`](../../frontend/src/routes/image-classification.tsx)
(§3.2), [`/depth`](../../frontend/src/routes/depth.tsx) (§3.1),
[`/object-detection`](../../frontend/src/routes/object-detection.tsx) (§3.3),
[`/segmentation`](../../frontend/src/routes/segmentation.tsx) (§3.4) and
[`/zero-shot-image-classification`](../../frontend/src/routes/zero-shot-image-classification.tsx)
(§3.5) ship; the other fourteen tasks in the sidebar still render the
`/tasks/$slug` placeholder. This file is therefore two things at once: a
description of the shipped `src/vision/` module (§1–§2), and the research the
next page gets written from (§3) — the verified checkpoint per task, the shape of
the page, and, for nine of them, the reason not to bother.

Each unbuilt task has a phased plan of its own, opened as a sub-issue of
[#2](https://github.com/bthek1/model_playground/issues/2); build order is in that
issue's build-order comment. The procedure is
[`docs/guides/adding-a-task-page.md`](../guides/adding-a-task-page.md).

Vision splits more cleanly than any other category. The
**discriminative half** (depth, classification, detection, segmentation,
embeddings, keypoints, open-vocabulary anything) runs beautifully in a tab: the
models are 20 MB to 200 MB, they are single-pass, and almost all of them have an
official ONNX export. The **generative half** (text-to-image, image-to-video,
text-to-video, text-to-3D, unconditional diffusion) does not run in a tab at
all, and the reason is structural rather than a missing export: latent diffusion
needs multi-gigabyte weights and tens of denoising steps.

So this file is honest about a hard split. Sections 3.1 to 3.11 are pages that
can be built; section 3.12 is the list of tasks that stay on a server, with the
reason for each.

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

### 3.6 Zero-Shot Object Detection, the same idea with boxes

Taxonomy task **Zero Shot Object Detection** · not built · **plan: [#16](https://github.com/bthek1/model_playground/issues/16)**. Upstream research: OWL-ViT, OWLv2, Grounding DINO, LLMDet.

| Upstream (PyTorch) | Browser model | Notes |
|---|---|---|
| `google/owlvit-base-patch32` | `Xenova/owlvit-base-patch32` | query lists and image queries |
| `google/owlv2-base-patch16-ensemble` | `Xenova/owlv2-base-patch16-ensemble` | the self-training jump, and the better default |
| `IDEA-Research/grounding-dino-tiny` | `onnx-community/grounding-dino-tiny-ONNX` | detection as phrase grounding |
| `iSEE-Laboratory/llmdet_tiny` | none | - |

```ts
const owl = await pipeline("zero-shot-object-detection", "Xenova/owlv2-base-patch16-ensemble",
                           loadOpts(backend));
const out = await owl(image, ["a hand", "a coffee cup", "a laptop"], { threshold: 0.2 });
```

One thing belongs in the UI above all else: **the prompt is the model.** Threshold and query wording move the results more than the checkpoint
choice does, so both belong in the RUN slot as live controls rather than being
hard-coded.

### 3.7 Mask Generation, interactive by construction

Taxonomy task **Mask Generation** · not built · **plan: [#17](https://github.com/bthek1/model_playground/issues/17)**. Upstream research: SAM, SAM-HQ, SAM 2.1, Grounded SAM.

This is the one task whose architecture was designed for exactly the interaction
a web page provides: **encode once, decode many.** The image encoder runs once
when the picture loads, then every click decodes a mask in a few milliseconds.
That is why SAM feels instant, and it is why the browser version feels as good
as the desktop one.

| Upstream (PyTorch) | Browser model | Notes |
|---|---|---|
| `facebook/sam-vit-base`, `Zigeng/SlimSAM-uniform-77` | `Xenova/slimsam-77-uniform` | ~40 MB, split into `vision_encoder` and `prompt_encoder_mask_decoder` |
| `facebook/sam2.1-hiera-tiny` | `onnx-community/sam2.1-hiera-tiny-ONNX` | the 2.1 line, larger, same split |
| `syscv-community/sam-hq-vit-base`, `facebook/sam3` | none | - |

Structure the engine around the split, because the split is the whole point:

```ts
// one call when the image changes
const embeddings = await visionEncoder(processedImage);

// one call per click, reusing the embeddings. Sub-100ms.
const { pred_masks, iou_scores } = await maskDecoder({
  image_embeddings: embeddings,
  input_points: [[[x, y]]],
  input_labels: [[1]],            // 1 = positive, 0 = negative
});
```

Two states, not one: LOAD covers the download, and a second "encoding" state
covers the per-image encode. Collapsing them means the user clicks and waits
with no explanation.

### 3.8 Image Feature Extraction, the invisible task that powers the rest

Taxonomy task **Image Feature Extraction** · not built · **plan: [#18](https://github.com/bthek1/model_playground/issues/18)**. Upstream research: DINOv2, CLIP, SigLIP 2, DINOv3.

No visible output of its own, which makes the page a similarity search: embed a
small gallery in the browser, embed the webcam frame, show the nearest
neighbours. The whole index lives in memory, so it is genuinely private.

| Upstream (PyTorch) | Browser model | Notes |
|---|---|---|
| `facebook/dinov2-base` | `Xenova/dinov2-small` | self-supervised features, ~22M params |
| `facebook/dinov3-vitb16-*` | `onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX` | the section 13 leader, small variant |
| `openai/clip-vit-base-patch32` | `Xenova/clip-vit-base-patch32` | language-aligned, use when text queries matter |

```ts
const feat = await pipeline("image-feature-extraction", "Xenova/dinov2-small", loadOpts(backend));
const emb = await feat(image, { pooling: "cls", normalize: true });
```

The question **which vector do you actually take** is a real UI control here.
CLS token against mean-pooled patches gives visibly different neighbours, and a
toggle makes that legible in a way prose cannot.

### 3.9 Image-to-Text, captioning and OCR

Taxonomy task **Image to Text** · not built · **plan: [#19](https://github.com/bthek1/model_playground/issues/19)**. Upstream research: BLIP, Florence-2, SmolVLM2, GOT-OCR 2.0.

| Upstream (PyTorch) | Browser model | Backend | Notes |
|---|---|---|---|
| `nlpconnect/vit-gpt2-image-captioning` | `Xenova/vit-gpt2-image-captioning` | WASM ok | the cheap caption, ~250 MB |
| `florence-community/Florence-2-base` | `onnx-community/Florence-2-base-ft` | WebGPU | one 0.23B model doing caption, dense caption, OCR and grounding by task token |
| `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `HuggingFaceTB/SmolVLM-256M-Instruct` | WebGPU | the 2.2B does not fit a tab; the 256M does |
| `Salesforce/blip-image-captioning-large` | none usable | - | the only mirror, `onnx-community/Salesforce_blip-image-captioning-base`, ships `split_0/1.onnx` rather than the transformers.js layout |
| `stepfun-ai/GOT-OCR-2.0-hf` | none | - | no export. Florence-2 covers OCR instead |

Florence-2 is the standout because one download covers captioning, dense
captioning, OCR and grounding. The task token selects the behaviour:

```ts
const florence = await pipeline("image-to-text", "onnx-community/Florence-2-base-ft",
                                { device: "webgpu", dtype: "fp16" });
await florence(image, { task: "<CAPTION>" });
await florence(image, { task: "<DETAILED_CAPTION>" });
await florence(image, { task: "<OCR>" });
await florence(image, { task: "<OD>" });        // boxes, drawable with drawBoxes()
```

### 3.10 Keypoint Detection, pose in the browser

Taxonomy task **Keypoint Detection** · not built · **plan: [#20](https://github.com/bthek1/model_playground/issues/20)**. Upstream research: RT-DETR + ViTPose, Sapiens2, SuperPoint, LightGlue.

Top-down pose is two models: detect people, then run the pose model on each
crop. Both halves have exports.

| Upstream (PyTorch) | Browser model | Notes |
|---|---|---|
| `PekingU/rtdetr_r50vd_coco_o365` (the detector) | `onnx-community/rtdetr_r50vd` or `onnx-community/dfine_n_coco-ONNX` | step 1, person boxes only |
| `usyd-community/vitpose-base-simple` | `onnx-community/vitpose-base-simple` | step 2, 17 COCO keypoints per crop |
| `facebook/sapiens2-pose-0.4b` | none | - | 308 keypoints, no export |
| `magic-leap-community/superpoint`, `ETH-CVG/lightglue_superpoint` | none | - | interest points and matching stay server-side for now |

Budget for two models live at once here, which is the one place this guide
relaxes the one-model rule. Keep the detector small precisely because of that:
the nano D-FINE plus ViTPose fits comfortably; two base-sized models does not.

### 3.11 Video Classification, and why it is really per-frame classification

Taxonomy task **Video Classification** · not built · **plan: [#21](https://github.com/bthek1/model_playground/issues/21)**. Upstream research: VideoMAE, TimeSformer, X-CLIP, V-JEPA 2.

**None of the four video transformers has an ONNX export.** They attend across
time as well as space, and nobody has shipped a browser-runnable one.

What you can honestly build is the control experiment: sample frames, classify
each one with CLIP zero-shot, and pool the scores over a sliding window. The
question worth asking is "does motion actually matter", and a page that only sees
single frames answers it from the other direction. Label it as what it is — a
frame-level baseline, not video classification.

### 3.12 The tasks that stay on a server

Nine tasks, and each has a structural reason rather than a missing export.
They keep their `/tasks/$slug` placeholders.

| Taxonomy task | Why not in a tab |
|---|---|
| Text to Image | SD 1.5 is 1.7 GB at fp16 before the text encoder, and needs 20 to 50 UNet passes. SDXL-Turbo cuts the steps but not the weights |
| Image to Image | same weights as above. The one exception is super-resolution, below |
| Image to Video | SVD and LTX are multi-gigabyte, and decode a 3-D VAE per clip |
| Unconditional Image Generation | DDPM at 1000 steps is a server job even at 32x32. Consistency models are the only part that comes close |
| Text to Video | at the edge of a 12 GB card, let alone a tab |
| Text to 3D | Shap-E is a diffusion model plus a NeRF decoder |
| Image to 3D | Zero123++ is SD-derived. The depth-to-point-cloud half is the exception, below |
| Video to Video | per-frame diffusion, so the cost of Image to Image multiplied by the frame count |
| Grounded SAM, OneFormer, RF-DETR | model-specific: no export exists yet, so they are missing *variants* rather than missing tasks |

Two useful carve-outs hide inside that list:

- **Super-resolution runs in the browser** ([plan #22](https://github.com/bthek1/model_playground/issues/22)). `Xenova/swin2SR-classical-sr-x2-64`
  is exported, and it is a single forward pass with no diffusion at all. It
  makes a genuinely good page, and it has no taxonomy entry of its own — ship it
  inside Image to Image.
- **The depth-to-point-cloud half of Image to 3D runs in the browser** ([plan #23](https://github.com/bthek1/model_playground/issues/23)).
  Depth Anything V2 gives the depth map, and the unprojection to a point cloud
  is arithmetic. Render it with WebGL or a WGSL compute shader. This is the
  cheapest impressive 3-D demo available.
- **Background removal** ([plan #24](https://github.com/bthek1/model_playground/issues/24)) — `briaai/RMBG-1.4`, official ONNX — has no taxonomy row
  of its own, but it is the preprocessing step everyone skips before image-to-3D,
  and it stands alone as a genuinely useful page.

---

## 4. Feasibility summary

| Taxonomy task | In-browser? | Recommended model | Best backend | If not |
|---|---|---|---|---|
| **Depth Estimation** | **Shipped** — `/depth` | `onnx-community/depth-anything-v2-small` | WebGPU | DepthPro gated for metric |
| **Image Classification** | **Shipped** — `/image-classification` | `Xenova/vit-base-patch16-224` | WebGPU / WASM | - |
| **Object Detection** | **Shipped** — `/object-detection` | `onnx-community/dfine_n_coco-ONNX` | WebGPU | RF-DETR to server |
| **Image Segmentation** | **Shipped** — `/segmentation` (semantic) | `Xenova/segformer-b0-finetuned-ade-512-512` | WebGPU | Mask2Former / OneFormer to server |
| **Image to Text** | Yes | `onnx-community/Florence-2-base-ft` | WebGPU | GOT-OCR to server |
| **Video Classification** | Frame-level only | CLIP over sampled frames | WebGPU | real video transformers to server |
| **Zero Shot Image Classification** | **Shipped** — `/zero-shot-image-classification` | `Xenova/clip-vit-base-patch32` | WebGPU / WASM | - |
| **Mask Generation** | Yes, excellent | `Xenova/slimsam-77-uniform` | WebGPU | SAM-HQ, SAM 3, Grounded SAM to server |
| **Zero Shot Object Detection** | Yes | `Xenova/owlv2-base-patch16-ensemble` | WebGPU | LLMDet to server |
| **Image Feature Extraction** | Yes, full | `Xenova/dinov2-small` | WebGPU / WASM | - |
| **Keypoint Detection** | Yes, pose only | `onnx-community/vitpose-base-simple` + detector | WebGPU | Sapiens2, SuperPoint to server |
| **Image to Image** | Super-resolution only | `Xenova/swin2SR-classical-sr-x2-64` | WebGPU | editing / img2img diffusion to server |
| **Image to 3D** | Depth-to-point-cloud only | Depth Anything V2 plus WebGL | WebGPU | Zero123++ and full reconstruction to server |
| **Text to Image** | No | - | - | server API |
| **Image to Video** | No | - | - | server API |
| **Unconditional Image Generation** | No | - | - | server API |
| **Text to Video** | No | - | - | server API |
| **Text to 3D** | No | - | - | server API |
| **Video to Video** | No | - | - | server API |

Rule of thumb: **one forward pass runs in a tab; fifty do not.** Every "no" in
that table is a diffusion model, and every "yes" is a single-pass network.

---

## 5. Memory and performance notes

The research these models come from targets 12 GB of VRAM and frees memory
aggressively. A tab is tighter, and the same discipline applies with different
mechanics.

- **One model live at a time**, with the pose page as the single deliberate
  exception. `await model.dispose()` after nulling the reference. There is no
  `torch.cuda.empty_cache()`, so disposing is the entire mechanism.
- **Resolution is the throttle, not the model.** A detector at 640x480 and the
  same detector at 1280x720 differ by roughly 4 times in cost. Downscale the
  frame before inference and draw the boxes on the full-size canvas.
- **Never queue frames.** One in flight at a time, as in section 2. A backlog is
  the reason a live demo feels laggy long before the model is the reason.
- **Encode once, decode many** wherever the architecture allows it. SAM is the
  obvious case; CLIP zero-shot is the other one, since the label embeddings are
  constant across frames and recomputing them per frame doubles the work for
  nothing. **Implemented for `/zero-shot-image-classification`** in
  `src/vision/zeroshot/`, which drives the two towers separately and caches
  several label sets at once — see §3.5, including what owning the final
  normalise/scale/softmax costs and the parity spec that controls it.
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
  `background-removal`.
- **onnxruntime-web** for the models with no pipeline wrapper.
- **Page construction**: [`docs/guides/adding-a-task-page.md`](../guides/adding-a-task-page.md).
- **In-repo standards**:
  [`docs/standards/model-page-pattern.md`](../standards/model-page-pattern.md)
  (the four-slot contract),
  [`docs/standards/model-visualization.md`](../standards/model-visualization.md)
  (canvas heatmaps, schematics, theme tokens — read before drawing anything),
  [`docs/guides/adding-a-model.md`](../guides/adding-a-model.md) §8.
- **The shipped precedent**: the six audio routes, mapped in
  [`docs/roadmaps/audio.md`](./audio.md), and the five vision routes in this
  category. A vision page is the same worker protocol with an image payload
  instead of a `Float32Array` — and, on the way back, a result that has been
  through `toCloneable`.
- Model recommendations here come from a companion collection of Python
  notebooks, which is a separate project and not a dependency of this repo. Every
  browser id above was checked against the Hugging Face API — re-check with
  `just fe-e2e-models` once a catalogue module exists to check.

