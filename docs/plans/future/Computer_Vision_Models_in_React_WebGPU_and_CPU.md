# Computer Vision Models in the Browser (WebGPU or CPU)

> The **Computer Vision** category of `components/layout/taskTaxonomy.ts`, task by
> task: what runs client-side on the user's GPU (WebGPU) or CPU (WebAssembly),
> which checkpoint to use, and which tasks stay on a server. No Python server in
> the inference path.

**Nothing in this category is built yet.** All nineteen Computer Vision tasks in
the sidebar currently render the `/tasks/$slug` placeholder. This file is the
research a plan gets written from: the verified checkpoint per task, the shape of
the page, and — for nine of them — the reason not to bother. Build order is
suggested in §3; the procedure is
[`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md), and the shared plumbing
(backend probe, worker protocol, size guardrail) is already shipped and described
in [`Audio_Models_in_React_WebGPU_and_CPU.md`](Audio_Models_in_React_WebGPU_and_CPU.md)
§1–2.

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
reason for each. **Image Classification (§3.2) is the one to build first** — it
is the simplest possible instance of the four-slot pattern, and the second vision
page costs a fraction of the first because the worker, the image helpers and the
canvas drawing all already exist by then.

Every model id below was checked against the Hugging Face API. Re-check before
shipping with `just fe-e2e-models`.

---

## 1. The core stack

`@huggingface/transformers` and `onnxruntime-web` are already dependencies —
nothing needs installing. Vision adds no new runtime.

Backend selection and dtype are **shared, not re-derived**. Import them:

```ts
import { pickBackend, loadOpts } from "@/audio/backend";
```

Yes, from `audio/` — that module is misnamed for a second modality, and the right
move when the first vision page lands is to `git mv` it to `src/model/backend.ts`
and update the five audio imports, rather than to write a second copy of the
probe. `audio/size.ts` (the size-before-load guardrail) deserves the same
treatment at the same time.

The rule that carries hardest into a tab: **hold one large model live at a time
and free it before loading the next.** A WebGPU context can OOM the tab exactly
the way a careless notebook cell OOMs a 12 GB container.

---

## 2. Image I/O helpers (the browser's PIL + torchvision)

Python opens a `PIL.Image` and hands it to a processor. In the browser the
equivalent is `RawImage`, which Transformers.js accepts everywhere a pipeline
takes an image, and which handles the decode for you.

These belong in `src/vision/image.ts` — the direct counterpart of the shipped
[`audio/io.ts`](../../../frontend/src/audio/io.ts), and the module every vision
route will import.

```ts
// src/vision/image.ts
import { RawImage } from "@huggingface/transformers";

/** A file the user picked, or a drag-and-drop, or a fetch. */
export async function fromFile(file: File): Promise<RawImage> {
  return RawImage.fromBlob(file);
}

/** One frame out of a live <video> element, for the webcam demos. */
export function fromVideo(video: HTMLVideoElement): RawImage {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d")!.drawImage(video, 0, 0);
  return RawImage.fromCanvas(c);
}

/** Open the camera. The browser analogue of cv2.VideoCapture(0, CAP_V4L2). */
export async function openCamera(video: HTMLVideoElement, width = 640, height = 480) {
  video.srcObject = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: width }, height: { ideal: height } },
  });
  await video.play();
  return () => (video.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
}
```

### Do not resize by hand

Every pipeline below calls `AutoProcessor`, which reads
`preprocessor_config.json` from the same repo and applies the model's own resize,
crop, rescale and normalise. That file **is** the input contract, published.
Reimplementing it by hand is the single most common source of a model that runs,
produces plausible-looking output, and is quietly wrong — which is exactly how
the two DSP bugs in `audio/enhance/` shipped past a green test suite, and that
route at least had the excuse that its model publishes no processor at all.

### Drawing the output

Most of these tasks produce something to draw over the input rather than text,
so the OUTPUT slot is a `<canvas>` sized to the source image.

```ts
// src/vision/draw.ts — see also components/viz/heatmap.tsx, which already
// draws a diverging canvas heatmap and owns the theme tokens.

/** Boxes from an object-detection pipeline, in absolute pixels. */
export function drawBoxes(
  ctx: CanvasRenderingContext2D,
  dets: { box: { xmin: number; ymin: number; xmax: number; ymax: number }; label: string; score: number }[],
) {
  ctx.lineWidth = 2;
  ctx.font = "14px system-ui";
  for (const { box, label, score } of dets) {
    const w = box.xmax - box.xmin, h = box.ymax - box.ymin;
    ctx.strokeStyle = "#22d3ee";
    ctx.strokeRect(box.xmin, box.ymin, w, h);
    ctx.fillStyle = "#22d3ee";
    ctx.fillText(`${label} ${score.toFixed(2)}`, box.xmin + 4, box.ymin + 16);
  }
}

/** A single-channel map (depth, a segmentation logit) to a colourised canvas. */
export function drawHeatmap(ctx: CanvasRenderingContext2D, data: Float32Array, w: number, h: number) {
  let lo = Infinity, hi = -Infinity;
  for (const v of data) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < data.length; i++) {
    const t = (data[i] - lo) / (hi - lo || 1);        // normalise first, always
    img.data[i * 4 + 0] = t * 255;
    img.data[i * 4 + 1] = (1 - Math.abs(t - 0.5) * 2) * 255;
    img.data[i * 4 + 2] = (1 - t) * 255;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
```

The `drawHeatmap` normalisation is not cosmetic. Relative depth models output an
inverse-depth map on an arbitrary scale: **alignment comes first**, and without a
per-frame min/max the canvas is either black or white.

Before writing a new drawing helper, check
[`components/viz/`](../../../frontend/src/components/viz/) and the standard in
[`../../standards/model-visualization.md`](../../standards/model-visualization.md).
`HeatmapTile` and `DivergingLegend` are already there, already theme-aware, and
already used by `/training` and `/tensor`.

### Run in a Web Worker, and cap the frame rate yourself

Vision demos are usually live, which makes the worker mandatory rather than
advisory: a 40 ms inference on the main thread is a page that cannot scroll.
The worker skeleton is the one in
[`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md) section 5, over the
shared `ModelRequest`/`ModelResponse` envelope. One `src/vision/pipeline.worker.ts`
serves every task in this file that is a plain Transformers.js pipeline; the task
string travels in the `load` message, exactly as `audio/pipeline.worker.ts` does.

One extra rule that only applies to video: **do not queue frames.** Grab a frame
only when the worker has returned the previous result. A naive
`requestAnimationFrame` loop that posts every frame builds an unbounded backlog
and the overlay drifts seconds behind the picture.

```ts
let busy = false;
async function tick() {
  if (!busy) {
    busy = true;
    worker.postMessage({ type: "run", payload: { image: fromVideo(video) } });
  }
  requestAnimationFrame(tick);            // busy is cleared in onmessage
}
```

---

## 3. Task by task

### 3.1 Depth Estimation, the best-looking browser demo in the category

Taxonomy task **Depth Estimation** · not built. Upstream research: Depth Anything V2, Depth Pro, ZoeDepth.

Depth Anything V2 Small is 25M parameters, single pass, and has an official ONNX
export. It runs at interactive rates on WebGPU and produces the most immediately
impressive output of any task here.

| Upstream (PyTorch) | Browser model | Backend | Notes |
|---|---|---|---|
| `depth-anything/Depth-Anything-V2-Small-hf` | `onnx-community/depth-anything-v2-small` | WebGPU / WASM | relative depth, the default |
| the same, older mirror | `Xenova/depth-anything-small-hf` | WebGPU / WASM | equivalent, keeps working |
| `apple/DepthPro-hf` | `onnx-community/DepthPro-ONNX` | WebGPU | metric depth plus focal length, but large. Gate it |
| `Intel/zoedepth-nyu-kitti` | none | - | no export. Use Depth Pro for metric |

```ts
const depth = await pipeline("depth-estimation", "onnx-community/depth-anything-v2-small",
                             loadOpts(backend));
const { predicted_depth, depth: asImage } = await depth(image);
// predicted_depth is the raw tensor -> drawHeatmap()
// asImage is an already-normalised RawImage if you just want a picture
```

Carry the caveat into the UI copy: this is **relative** depth. The
numbers are not metres, and comparing two frames without alignment is
meaningless. The page should say so next to the colour bar.

### 3.2 Image Classification, the reference implementation of the pattern

Taxonomy task **Image Classification** · not built. Upstream research: ResNet-50, ViT-B/16, ConvNeXt V2, DINOv2.

The simplest possible task page, which makes it the right one to build first
when you are learning the four-slot pattern.

| Upstream (PyTorch) | Browser model | Size (q8) | Notes |
|---|---|---|---|
| `google/vit-base-patch16-224` | `Xenova/vit-base-patch16-224` | ~88 MB | the transformer baseline |
| `microsoft/resnet-50` | `Xenova/resnet-50` | ~26 MB | the CNN baseline |
| - | `onnx-community/mobilenetv4_conv_small.e2400_r224_in1k` | ~10 MB | the "instant load" option, and the honest speed/accuracy floor |
| - | `Xenova/mobilevitv2-1.0-imagenet1k-256` | ~19 MB | a middle point |

```ts
const clf = await pipeline("image-classification", "Xenova/vit-base-patch16-224",
                           { ...loadOpts(backend), topk: 5 });
const preds = await clf(image);      // [{ label, score }, ...]
```

One finding is worth a UI decision: **the aggregate number hides the failure.**
Show the top 5 with their scores,
not a single label, so a 0.31/0.29 near-tie is visible rather than presented as
a confident answer.

### 3.3 Object Detection, the flagship live demo

Taxonomy task **Object Detection** · not built. Upstream research: DETR, YOLOS, RT-DETRv2, D-FINE, RF-DETR.

| Upstream (PyTorch) | Browser model | Backend | Notes |
|---|---|---|---|
| `facebook/detr-resnet-50` | `Xenova/detr-resnet-50` | WebGPU / WASM | the set-prediction original, slow but canonical |
| `hustvl/yolos-small` | `Xenova/yolos-small` | WebGPU / WASM | a plain ViT that detects |
| `PekingU/rtdetr_v2_r18vd` | `onnx-community/rtdetr_r50vd` | WebGPU | the r18 variant has no export; r50 does |
| `ustc-community/dfine-small-coco` | `onnx-community/dfine_s_coco-ONNX` | WebGPU | the localisation specialist from section 11 |
| the same, smaller | `onnx-community/dfine_n_coco-ONNX` | WebGPU / WASM | nano. The one to default to for a live webcam |
| `Roboflow/rf-detr-*` | none | - | no ONNX export as of this writing. Table only |

```ts
const det = await pipeline("object-detection", "onnx-community/dfine_n_coco-ONNX",
                           loadOpts(backend));
const out = await det(image, { threshold: 0.4, percentage: false });
drawBoxes(ctx, out);
```

`percentage: false` returns absolute pixels, which is what `drawBoxes` above
expects. Getting this backwards produces boxes clustered in the top-left corner,
which is the single most common bug on a first detection page.

### 3.4 Image Segmentation, works, with a caveat about which models exist

Taxonomy task **Image Segmentation** · not built. Upstream research: SegFormer, Mask2Former, OneFormer, EoMT-DINOv3.

Semantic segmentation ports cleanly. Instance and panoptic are thinner: only
the DETR panoptic head has an export; Mask2Former, OneFormer and EoMT have no
browser path.

| Upstream (PyTorch) | Browser model | Task | Notes |
|---|---|---|---|
| `nvidia/segformer-b0-finetuned-ade-512-512` | `Xenova/segformer-b0-finetuned-ade-512-512` | semantic, 150 ADE classes | the efficient baseline, ~14 MB |
| `facebook/mask2former-*-panoptic` | `Xenova/detr-resnet-50-panoptic` | panoptic | a different model, same output shape |
| - | `Xenova/face-parsing` | semantic, faces | a strong small demo, and it works live |
| - | `mattmdjaga/segformer_b2_clothes` | semantic, garments | the other good live demo |
| `shi-labs/oneformer_*`, `tue-mps/eomt-*` | none | - | no export. Table only |

```ts
const seg = await pipeline("image-segmentation", "Xenova/segformer-b0-finetuned-ade-512-512",
                           loadOpts(backend));
const masks = await seg(image);   // [{ label, score, mask: RawImage }, ...]
```

Each mask comes back as a single-channel `RawImage`. Composite them into one
canvas with a per-label colour rather than rendering 150 separate images.

### 3.5 Zero-Shot Image Classification, the best "your own labels" page

Taxonomy task **Zero Shot Image Classification** · not built. Upstream research: CLIP, OpenCLIP, SigLIP 2, MetaCLIP 2.

The most satisfying page in the category, because the user types the labels. It
is also the cheapest live demo: the text side is encoded once and reused for
every frame.

| Upstream (PyTorch) | Browser model | Notes |
|---|---|---|
| `openai/clip-vit-base-patch32` | `Xenova/clip-vit-base-patch32` | the reference, ~150 MB fp16 |
| `google/siglip-base-patch16-224` | `Xenova/siglip-base-patch16-224` | sigmoid loss, better calibrated scores |
| `google/siglip2-base-patch16-224` | `onnx-community/siglip2-base-patch16-224-ONNX` | the current-generation choice |
| `laion/CLIP-ViT-B-32-laion2B-*` | none | - |

```ts
const zs = await pipeline("zero-shot-image-classification", "Xenova/clip-vit-base-patch32",
                          loadOpts(backend));
const out = await zs(image, ["a photo of a cat", "a photo of a dog", "an empty room"]);
```

**Ship the prompt-template experiment.** It is the most instructive result in
this whole file: `"a photo of a {}"` beats a bare
`"{}"` by several points, and a page that lets the user toggle the template
teaches that in one click. Very few demos anywhere show this.

### 3.6 Zero-Shot Object Detection, the same idea with boxes

Taxonomy task **Zero Shot Object Detection** · not built. Upstream research: OWL-ViT, OWLv2, Grounding DINO, LLMDet.

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

Taxonomy task **Mask Generation** · not built. Upstream research: SAM, SAM-HQ, SAM 2.1, Grounded SAM.

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

Taxonomy task **Image Feature Extraction** · not built. Upstream research: DINOv2, CLIP, SigLIP 2, DINOv3.

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

Taxonomy task **Image to Text** · not built. Upstream research: BLIP, Florence-2, SmolVLM2, GOT-OCR 2.0.

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

Taxonomy task **Keypoint Detection** · not built. Upstream research: RT-DETR + ViTPose, Sapiens2, SuperPoint, LightGlue.

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

Taxonomy task **Video Classification** · not built. Upstream research: VideoMAE, TimeSformer, X-CLIP, V-JEPA 2.

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

- **Super-resolution runs in the browser.** `Xenova/swin2SR-classical-sr-x2-64`
  is exported, and it is a single forward pass with no diffusion at all. It
  makes a genuinely good page, and it has no taxonomy entry of its own — ship it
  inside Image to Image.
- **The depth-to-point-cloud half of Image to 3D runs in the browser.**
  Depth Anything V2 gives the depth map, and the unprojection to a point cloud
  is arithmetic. Render it with WebGL or a WGSL compute shader. This is the
  cheapest impressive 3-D demo available.
- **Background removal** (`briaai/RMBG-1.4`, official ONNX) has no taxonomy row
  of its own, but it is the preprocessing step everyone skips before image-to-3D,
  and it stands alone as a genuinely useful page.

---

## 4. Feasibility summary

| Taxonomy task | In-browser? | Recommended model | Best backend | If not |
|---|---|---|---|---|
| **Depth Estimation** | Yes, excellent | `onnx-community/depth-anything-v2-small` | WebGPU | metric depth, gate DepthPro |
| **Image Classification** | Yes, full | `Xenova/vit-base-patch16-224` | WebGPU / WASM | - |
| **Object Detection** | Yes, full | `onnx-community/dfine_n_coco-ONNX` | WebGPU | RF-DETR to server |
| **Image Segmentation** | Yes, semantic only | `Xenova/segformer-b0-finetuned-ade-512-512` | WebGPU | Mask2Former / OneFormer to server |
| **Image to Text** | Yes | `onnx-community/Florence-2-base-ft` | WebGPU | GOT-OCR to server |
| **Video Classification** | Frame-level only | CLIP over sampled frames | WebGPU | real video transformers to server |
| **Zero Shot Image Classification** | Yes, excellent | `Xenova/clip-vit-base-patch32` | WebGPU / WASM | - |
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
  nothing.
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
- **Page construction**: [`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md).
- **In-repo standards**:
  [`../../standards/model-page-pattern.md`](../../standards/model-page-pattern.md)
  (the four-slot contract),
  [`../../standards/model-visualization.md`](../../standards/model-visualization.md)
  (canvas heatmaps, schematics, theme tokens — read before drawing anything),
  [`../../guides/adding-a-model.md`](../../guides/adding-a-model.md) §8.
- **The shipped precedent**: the five audio routes, mapped in
  [`Audio_Models_in_React_WebGPU_and_CPU.md`](Audio_Models_in_React_WebGPU_and_CPU.md).
  A vision page is the same worker protocol with `RawImage` instead of
  `Float32Array`.
- Model recommendations here come from a companion collection of Python
  notebooks, which is a separate project and not a dependency of this repo. Every
  browser id above was checked against the Hugging Face API — re-check with
  `just fe-e2e-models` once a catalogue module exists to check.
