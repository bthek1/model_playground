# TODO

## Computer Vision

Shipped: [`/image-classification`](frontend/src/routes/image-classification.tsx),
[`/depth`](frontend/src/routes/depth.tsx),
[`/object-detection`](frontend/src/routes/object-detection.tsx),
[`/segmentation`](frontend/src/routes/segmentation.tsx),
[`/zero-shot-image-classification`](frontend/src/routes/zero-shot-image-classification.tsx).
The rest of the category is planned per task — see
[`docs/roadmaps/vision.md`](docs/roadmaps/vision.md) and the plan issues.

- [x] depth estimation — [#12](https://github.com/bthek1/model_playground/issues/12)
- [x] object detection, live from the camera — [#13](https://github.com/bthek1/model_playground/issues/13)
- [x] image segmentation — [#14](https://github.com/bthek1/model_playground/issues/14)
- [x] zero-shot image classification — [#15](https://github.com/bthek1/model_playground/issues/15)
      (including the text-embedding cache: `src/vision/zeroshot/` drives the two
      towers separately, pinned against the full graph by `just fe-e2e-zeroshot`)
- [ ] keypoint detection (pose) — [#20](https://github.com/bthek1/model_playground/issues/20)

                
- [ ] deploy using pulumi
- [ ] setup LSTM stack


```
Wave 0 — #10 (platform) → #11 (Image Classification, which graduates #2 to docs/roadmaps/vision.md), with a note to merge the two if one person builds both.
Wave 1 — DONE. #12/#13/#14/#15. In the event the drawing primitives were already
in place from Wave 0, so the shared work was the *page* layer instead:
useImagePick, useCameraFrames, ImageSourcePanel, OverlayCanvas — plus toCloneable,
without which no Tensor-returning task can cross the worker boundary at all.
Wave 2 — #16, #18, #17, #19, #20, #21, ordered by how much new shape each introduces (own engine, generative decoder, two models live).
Wave 3 — the carve-outs #24, #22, #23, with #24's licence/taxonomy questions called out as re-scoping risks.
```