# TODO

## Computer Vision

Shipped: [`/image-classification`](frontend/src/routes/image-classification.tsx).
The rest of the category is planned per task — see
[`docs/roadmaps/vision.md`](docs/roadmaps/vision.md) and the plan issues.

- [ ] depth estimation — [#12](https://github.com/bthek1/model_playground/issues/12)
- [ ] object detection, live from the camera — [#13](https://github.com/bthek1/model_playground/issues/13)
- [ ] keypoint detection (pose) — [#20](https://github.com/bthek1/model_playground/issues/20)

                
- [ ] deploy using pulumi
- [ ] setup LSTM stack


```
Wave 0 — #10 (platform) → #11 (Image Classification, which graduates #2 to docs/roadmaps/vision.md), with a note to merge the two if one person builds both.
Wave 1 — #12/#13/#14/#15, each adding exactly one drawing primitive; #15 flagged as the high-leverage one since #16 and #21 are mostly "#15 plus one thing".
Wave 2 — #16, #18, #17, #19, #20, #21, ordered by how much new shape each introduces (own engine, generative decoder, two models live).
Wave 3 — the carve-outs #24, #22, #23, with #24's licence/taxonomy questions called out as re-scoping risks.
```