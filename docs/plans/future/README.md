# Roadmap: the task taxonomy, category by category

These files answer one question for every task in the sidebar
([`components/layout/taskTaxonomy.ts`](../../../frontend/src/components/layout/taskTaxonomy.ts)):
**can this run in a browser tab, and if so, on which checkpoint?**

They are **not plans.** A plan is phased, has a Testing section, lives in
[`../in-progress/`](../in-progress/) while it is being built and moves to
[`../completed/`](../completed/) when it lands. These are the research a plan
gets written from — one file per taxonomy category, plus the procedure that
turns any row in them into a route.

## Start here

| File | What it is |
|---|---|
| [`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md) | **The procedure.** Triage → checkpoint → worker → engine → hook → route → registrations → tests → docs. Read this before any of the tables. |
| [`Audio_Models_in_React_WebGPU_and_CPU.md`](Audio_Models_in_React_WebGPU_and_CPU.md) | **The reference implementation.** Five shipped routes, and §1–2 are the shared plumbing (backend probe, I/O, worker split) every other category cites. |

## Category status

| Category | Guide | Built | Left |
|---|---|---|---|
| Audio | [Audio](Audio_Models_in_React_WebGPU_and_CPU.md) | 5 of 6 — TTS, text-to-audio, ASR, audio-to-audio, classification | Voice Activity Detection |
| Theory | — (shipped, no guide) | 2 of 3 — [`/training`](../../../frontend/src/routes/training.tsx), [`/tensor`](../../../frontend/src/routes/tensor.tsx) | Discrete Maths |
| Computer Vision | [Computer Vision](Computer_Vision_Models_in_React_WebGPU_and_CPU.md) | 0 of 19 | 10 buildable, 9 server-side |
| Natural Language Processing | [NLP](NLP_Models_in_React_WebGPU_and_CPU.md) | 0 of 12 | 11 buildable, 1 (Table QA) not |
| Multimodal | [Multimodal](Multimodal_Models_in_React_WebGPU_and_CPU.md) | 0 of 9 | 5 buildable, 4 server-side |
| Tabular | [Tabular](Tabular_Models_in_React_WebGPU_and_CPU.md) | 0 of 3 | all 3, as *training* pages rather than inference |
| Reinforcement Learning | [RL](Reinforcement_Learning_in_React_WebGPU_and_CPU.md) | 0 of 2 | both, once the environment is written in TypeScript |
| Other (Graph ML) | [Graph](Graph_Models_in_React_WebGPU_and_CPU.md) | 0 of 1 | 1, on the raw WGSL path |

Everything not built renders the generic
[`/tasks/$slug`](../../../frontend/src/routes/tasks.$slug.tsx) placeholder, which
is deliberate: the full taxonomy is visible without every task existing.

## Suggested build order

The first page in a category is several times the cost of the second, because it
establishes the module (`src/vision/`, `src/text/`), the worker and the shared
components. So build the cheapest page in a category first, not the most
impressive one.

1. **NLP — Text Classification.** The smallest useful page in the app: a string
   in, a score list out, a 67 MB model, no decode step at all.
2. **Computer Vision — Image Classification.** Establishes `RawImage`, the camera
   helper and the canvas drawing that nine other vision pages reuse.
3. **Audio — Voice Activity Detection.** Closes the one category that is nearly
   finished, and gives `useLiveAsr` a gate it currently lacks.
4. **Tabular or Graph.** Both extend the existing WGSL path rather than the
   Transformers.js one, so they exercise a different half of the app.
5. **Multimodal, last.** These pages need the vision helpers *and* the audio
   helpers, and they are the only ones where WebGPU is a hard requirement.

## Two conventions these files assume

- **The shared backend probe lives in
  [`audio/backend.ts`](../../../frontend/src/audio/backend.ts)** and the size
  guardrail in [`audio/size.ts`](../../../frontend/src/audio/size.ts). Both are
  misnamed for a second modality. When the first non-audio page lands, `git mv`
  them to `src/model/` and update the audio imports — do not write a second copy.
- **Model recommendations come from a companion collection of Python notebooks**,
  which is a separate project. The checkpoints, input contracts and head-to-head
  results are quoted here so you never have to open it, and **nothing in this repo
  depends on it.** Where a file says "upstream research", read it that way.

Every browser model id in these files was checked against the Hugging Face API
when it was written. Re-check before shipping one:

```bash
just fe-e2e-models   # seconds, no downloads, fails loudly on a dead id
```

## Related

- [`../../standards/model-page-pattern.md`](../../standards/model-page-pattern.md) — the four-slot contract every page implements
- [`../../standards/model-visualization.md`](../../standards/model-visualization.md) — how a model and its internals are drawn
- [`../../guides/adding-a-model.md`](../../guides/adding-a-model.md) — §8 Transformers.js, §9 a bare ONNX graph
- [`../../explanations/webgpu-inference.md`](../../explanations/webgpu-inference.md) — how inference actually runs
- [`../../guides/e2e-testing.md`](../../guides/e2e-testing.md) — the Playwright layer, including the `@slow` specs that are the only guard against a broken model
