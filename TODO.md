# TODO

## Computer Vision

**Fourteen of twenty shipped.** [`/image-classification`](frontend/src/routes/image-classification.tsx),
[`/depth`](frontend/src/routes/depth.tsx),
[`/object-detection`](frontend/src/routes/object-detection.tsx),
[`/segmentation`](frontend/src/routes/segmentation.tsx),
[`/zero-shot-image-classification`](frontend/src/routes/zero-shot-image-classification.tsx),
[`/zero-shot-object-detection`](frontend/src/routes/zero-shot-object-detection.tsx),
[`/image-features`](frontend/src/routes/image-features.tsx),
[`/mask-generation`](frontend/src/routes/mask-generation.tsx),
[`/image-to-text`](frontend/src/routes/image-to-text.tsx),
[`/pose`](frontend/src/routes/pose.tsx),
[`/video-classification`](frontend/src/routes/video-classification.tsx),
[`/background-removal`](frontend/src/routes/background-removal.tsx),
[`/super-resolution`](frontend/src/routes/super-resolution.tsx) and
[`/image-to-3d`](frontend/src/routes/image-to-3d.tsx).
Twenty rather than nineteen because background removal added a taxonomy row the
Hub does not have. The rest of the category stays on a server, with a reason per
task — see [`docs/roadmaps/vision.md`](docs/roadmaps/vision.md) §3.12.

- [x] depth estimation — [#12](https://github.com/bthek1/model_playground/issues/12)
- [x] object detection, live from the camera — [#13](https://github.com/bthek1/model_playground/issues/13)
- [x] image segmentation — [#14](https://github.com/bthek1/model_playground/issues/14)
- [x] zero-shot image classification — [#15](https://github.com/bthek1/model_playground/issues/15)
      (including the text-embedding cache: `src/vision/zeroshot/` drives the two
      towers separately, pinned against the full graph by `just fe-e2e-zeroshot`)
- [x] zero-shot object detection — [#16](https://github.com/bthek1/model_playground/issues/16)
- [x] image feature extraction — [#18](https://github.com/bthek1/model_playground/issues/18)
- [x] mask generation (SAM, encode once / decode many) — [#17](https://github.com/bthek1/model_playground/issues/17)
- [x] image to text (captioning, OCR, grounding) — [#19](https://github.com/bthek1/model_playground/issues/19)
- [x] keypoint detection (pose) — [#20](https://github.com/bthek1/model_playground/issues/20)
- [x] video classification, **as a frame-level baseline** — [#21](https://github.com/bthek1/model_playground/issues/21)

The Wave 3 carve-outs, each covering *part* of its slug and each saying so:

- [x] super-resolution inside Image to Image — [#22](https://github.com/bthek1/model_playground/issues/22)
      (overlapping tiles feathered at the seams, against a bicubic baseline;
      `just fe-e2e-superres` is what decides the WASM precision pin)
- [x] depth-to-point-cloud, the browser half of Image to 3D — [#23](https://github.com/bthek1/model_playground/issues/23)
      (no new model — /depth's checkpoint plus an unprojection, rendered through
      the first WGSL *render* pass in the repo)
- [x] background removal — [#24](https://github.com/bthek1/model_playground/issues/24)
      Both re-scoping risks were real and both were settled before the build:
      **licence** — RMBG-1.4 is CC non-commercial, so Apache-2.0 MODNet is the
      default and RMBG is offered with the restriction on screen; **taxonomy** —
      a Computer Vision row was added, a deliberate departure from the Hub's
      task list, taking the category from nineteen rows to twenty.


- [ ] deploy using pulumi
- [ ] setup LSTM stack


```
Wave 0 — #10 (platform) → #11 (Image Classification, which graduates #2 to docs/roadmaps/vision.md), with a note to merge the two if one person builds both.
Wave 1 — DONE. #12/#13/#14/#15. In the event the drawing primitives were already
in place from Wave 0, so the shared work was the *page* layer instead:
useImagePick, useCameraFrames, ImageSourcePanel, OverlayCanvas — plus toCloneable,
without which no Tensor-returning task can cross the worker boundary at all.
Wave 2 — DONE. #16/#18/#17/#19/#20/#21. Three of the six turned out not to be
plain `pipeline()` calls, which is now the documented criterion for owning an
engine (docs/guides/adding-a-model.md §10): SAM, image-to-text and pose own one;
zero-shot detection and image features rode the generic worker unchanged, and
video classification reused the zero-shot engine wholesale. The shared work was in the
*platform* rather than the page layer — ModelPicker now enforces
`VisionModel.backends` through `useBackendProbe`, `VisionModel.graphs` lets the
Hub-id spec check the files a model actually downloads, `model/progress.ts` is
keyed on repo+file so two models share one bar, and PhraseList/OverlayCanvas.onPick
came out of the third and second callers respectively.
Wave 3 — the carve-outs #24, #22, #23, with #24's licence/taxonomy questions called out as re-scoping risks.
```
