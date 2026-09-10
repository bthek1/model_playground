# AI Model Visualization Standards

How Model Playground presents **AI models and their internal structure** to the user.
This is the visual counterpart to the [WebGPU inference explanation](../explanations/webgpu-inference.md):
that document covers *how* a model runs; this one covers *how we show it*.

The goal is a single, coherent visual language so that every model — a 7,850-parameter
linear classifier today, a quantized transformer tomorrow — is rendered with the same
grammar of stages, arrows, heatmaps, chips, and charts. A user should be able to look at
any model page and understand, at a glance: **what it is, how it's built, what its
parameters look like, and how it performs.**

The reference implementation of everything here is the **Training route** — a full-bleed
schematic whose background *is* the model
([`components/training/ModelArchitecture.tsx`](../../frontend/src/components/training/ModelArchitecture.tsx),
[`ModelWeights.tsx`](../../frontend/src/components/training/ModelWeights.tsx)). When in doubt,
match those components.

---

## 1. Principles

1. **Show real data, never a stock diagram.** A layer block should be sized/labelled from
   the actual `ModelCard.config`; a weight tile must be drawn from the live `Float32Array`,
   not a placeholder image. The linear classifier's weight templates *sharpen into digit
   shapes as training runs* precisely because they read the real weight stream. Prefer a
   truthful, ugly number to a pretty fake one.
2. **Progressive disclosure.** Lead with the shape (input → op → output). Put parameter
   budgets, tensor shapes, and hyperparameters one level in (chips, popovers, dialogs).
   Put raw dumps (full weight matrices, per-token logits) behind an explicit expand.
3. **Theme-aware by construction.** Everything must read correctly in light *and* dark.
   Use the CSS theme tokens (`--card`, `--muted-foreground`, `--chart-1…5`) — never
   hard-code a hex that only works on one background. Where raw canvas colors are
   unavoidable (signed-weight heatmaps), scale **alpha with magnitude** so near-zero
   values fade into the card and the encoding survives both themes.
4. **Degrade gracefully.** WebGPU may be `unsupported` / `no-adapter` / `no-device`
   (see [webgpu-inference.md](../explanations/webgpu-inference.md)). A model view must still
   render its *structure* and *metadata* with no GPU — only the live/animated parts
   (running inference, updating weights) depend on a device.
5. **Cross-checked and honest.** Numbers shown to the user (GFLOP/s, latency, tokens/sec)
   are the same ones we `POST` to `/api/registry/runs/`. If a value is estimated or a
   kernel is unverified against its CPU reference, say so — don't imply precision we
   don't have.

---

## 2. What to visualize — the anatomy of a model

Every model view is composed from up to five layers of detail. Not every model has all
five; render the ones that apply, in this order of prominence.

| Layer | What it answers | Source of truth | Typical rendering |
|-------|-----------------|-----------------|-------------------|
| **Identity** | *What is this?* name, task, license, size, public/private | `ModelCard` (`name`, `task`, `license`, `size_bytes`, `is_public`) | Card header + badges; `titleCase(task)`, `formatBytes(size_bytes)` |
| **Structure** | *How is it built?* layers, ops, dataflow, I/O shapes | `ModelCard.config` (tensor shapes, entry points, workgroup sizes) + WGSL kernels | Left-to-right **stage/arrow schematic** (§4) |
| **Parameters** | *What did it learn?* weights & biases per layer | live `Float32Array` weight stream (training) / decoded weights (`weights_url`) | **Heatmap tiles** + diverging legend (§5) |
| **Runtime** | *What is it doing now?* inputs, activations, outputs, tokens | worker inference results | Input preview, activation maps, output probabilities/logits |
| **Performance** | *How well does it run?* latency, throughput, loss/accuracy | `InferenceRun.metrics`, training history | **Charts** (§6): loss/accuracy curves, latency, GFLOP/s |

**Per-task emphasis** (from [`model-task-categories.md`](../model-task-categories.md)):

- **`custom` / tensor ops** — emphasise *structure*: the matmul/elementwise dataflow, tensor
  shapes, workgroup grid. (See the Tensor route + `tensorops.ts`.)
- **`vision`** — emphasise *runtime*: input image, per-class weight templates or feature/attention
  maps, output class probabilities. (See MNIST training.)
- **`llm`** — emphasise *runtime + structure*: tokenizer view (token boundaries + ids),
  per-token output distribution, KV-cache/layer stack. Stream tokens; never block the UI.
- **`embedding` / `audio`** — emphasise *outputs*: vector heatmap / similarity, spectrogram.

---

## 3. The visual vocabulary (building blocks)

These are the reusable primitives. New model views should **compose these**, not invent
parallel ones. The canonical implementations live in
[`components/viz/`](../../frontend/src/components/viz/) — `schematic.tsx` (Stage / Arrow /
ParamChip) and `heatmap.tsx` (HeatmapTile / DivergingLegend). Both the Training route
(`components/training/`) and the Tensor route (`routes/tensor.tsx`) compose them.

### Stage
A labelled box for one phase of the pipeline (INPUT LAYER, WEIGHTS + BIAS, OUTPUT). Title
in `text-[11px] font-semibold tracking-wide`, sub-label in `text-[10px] text-muted-foreground`,
container `rounded-lg border bg-card/60 p-2.5`. See `viz/schematic.tsx::Stage`.

### Arrow
The connector between stages, carrying the operation label (`W·x + b`, `softmax`). Mono
label, big `⟶` glyph that **rotates to point down when stages stack vertically**
(`rotate-90 … xl:rotate-0`) so the schematic reflows on narrow screens. See
`viz/schematic.tsx::Arrow`.

### ParamChip
A compact `label value` pill for a single scalar fact — parameter counts, tensor dims,
hyperparameters. Mono `text-[11px]`; use the `accent` variant (`bg-primary/15 text-primary`)
for the headline number (e.g. total params). See `viz/schematic.tsx::ParamChip`.

### Heatmap tile
A `<canvas>` (not `<img>`, not a div grid) drawn per-pixel from a `Float32Array`, sized with
a Tailwind `size-*` class (or inline `width`/`height` to preserve aspect ratio) and
`[image-rendering:pixelated]` so individual weights/activations stay crisp when scaled up. One
tile per class/channel/head. `viz/heatmap.tsx::HeatmapTile` accepts a strided `at` accessor so
a class column can be drawn out of a larger buffer without copying it (see
`ModelWeights.tsx::ClassTemplate`).

### Legend
Every heatmap needs one: the value→color mapping made explicit, with numeric end-labels
(`−maxAbs … +maxAbs`), a gradient swatch, a plain-words gloss (`blue = against · red = for`),
and optional state (`after epoch 4`). Never ship a colored visualization without its legend.
See `viz/heatmap.tsx::DivergingLegend`.

### Stat tile / KPI
A single metric with label + value (+ optional delta). Value in `tabular-nums` mono so digits
don't jitter as they update. Used for latency, throughput, param counts, accuracy.

### Time-aligned track
A full-width `<canvas>` drawn from a `Float32Array` against a **time** x-axis, for signals the
user reads left-to-right: the waveform of a clip, a live mic tap, a per-frame model score. These
live in [`components/audio/`](../../frontend/src/components/audio/) rather than `viz/` because
they are audio-domain, but they share one grammar and it is the grammar that matters:

- **Paint in the canvas's own resolved `color`** (`getComputedStyle(canvas).color`), never a
  hard-coded fill, so a Tailwind text class themes them in both schemes. Same rule as the
  heatmap's alpha-encoded magnitude, reached a different way.
- **Stacked tracks must share one x-axis.** `/vad` draws `Waveform` and `VadTimeline` in a
  single `space-y-1` column at the same width, with the same number of samples behind them, so
  a peak and its speech score line up vertically. A track whose axis silently differs from the
  one above it is worse than no track.
- **Redraw on resize** (`ResizeObserver`) and degrade when the 2-D context is missing — every
  one of these renders under happy-dom in the unit tests, where `getContext` returns null.
- **Give it `role="img"` and an `aria-label` that states the data**, not the picture:
  "Speech probability for 343 frames, threshold 0.50". It is also what the route tests query by.

`Waveform` / `LiveWaveform` (`audio/Waveform.tsx`) and `VadTimeline` (`audio/VadTimeline.tsx`)
are the implementations. A new time-series track composes these, or joins them here.

---

### Overlays on a source image (vision)

Boxes, class masks and single-channel maps drawn *over* a photo live in
[`vision/draw.ts`](../../frontend/src/vision/draw.ts), for the same reason the
audio timeline components live in `components/audio/`: they are domain-specific,
but they share this grammar. Three rules carry over, and one is new:

- **A sequential map is not a diverging one.** Depth has no meaningful zero, so
  `drawHeatmap` uses a ramp that is monotonic in lightness rather than the
  red/blue `paintDiverging` above. Signed tensors still use `viz/heatmap.tsx`.
- **Normalise per map before painting.** Relative depth arrives on an arbitrary
  scale; without rescaling to the values present the canvas is uniformly black or
  uniformly white, and the page reads as broken rather than wrong.
- **A label's colour is deterministic** (`colorForLabel`), so a class keeps its
  colour between frames of a live feed. A class that changes colour every frame is
  worse than no colour at all.
- **Composite masks through a scratch canvas**, then `drawImage`. `putImageData`
  *replaces* pixels including alpha, so writing an overlay straight onto the
  target erases the picture underneath instead of tinting it.
- **Opacity is how an overlay expresses doubt.** A pose heatmap always has a maximum
  somewhere, so an occluded ankle comes back as a confident-looking *guess*, not as an
  absence. `pose/skeleton.ts` fades a joint below its confidence floor rather than
  hiding it — and draws each limb at the **lower** of its two joints' confidences,
  because an edge is only as believable as its weakest end and averaging would let one
  solid joint carry a guessed one into looking certain. Never fade to nothing: "the
  model put it here and does not believe it" is information, and a joint that vanishes
  is indistinguishable from one the model never returned.
- **A skeleton is not a generic overlay**, which is why `pose/skeleton.ts` owns it rather
  than `draw.ts`. The three forms in `draw.ts` are meaningful for any task; a skeleton is
  meaningless without its specific 17-keypoint ordering, and the index *is* the label —
  `post_process_pose_estimation` returns `labels: number[]` and nothing else names them.
  Domain drawing that carries a domain contract lives with the domain.

**The coordinate space is part of the drawing, and getting it wrong is silent.**
Every overlay here is painted onto a canvas sized to the *source* image while the model
ran on a downscaled frame, so something has to map between them —
`scaleDetections` for boxes, `scalePeople` for a skeleton and its box together (scaling
one without the other shrinks the skeleton away from its own outline). The same applies
in reverse for input: `OverlayCanvas`'s `onPick` converts a click from CSS pixels back to
source pixels, because the canvas is scaled down by CSS and a raw offset is wrong by that
factor. None of these produce an error when wrong — they produce a plausible picture. A
canvas assertion cannot catch it either, so the checks live in pure unit tests over the
arithmetic and in `@slow` specs that assert geometry.

---

## 4. Color system

Model Playground uses **OKLCH** theme tokens defined in
[`frontend/src/index.css`](../../frontend/src/index.css). Reach for a token before a raw color.

### Categorical — series, layers, classes
Use the five chart tokens `--chart-1 … --chart-5` (exposed as `color-chart-1…5` and CSS
`var(--chart-N)`). They form a blue-family ramp tuned for both themes. For **more than five
categories**, don't invent ad-hoc hexes — cycle the ramp with varied lightness, or switch to a
sequential encoding. Keep a category's color stable across every chart on the page.

### Sequential — a single magnitude (activations, attention, similarity, |weight|)
A one-hued ramp from `--muted` (low) to `--primary` (high), or a perceptually-uniform OKLCH
lightness sweep at fixed hue. Low values must recede into the card background, not compete
with it.

### Diverging — signed values (weights, gradients, logit deltas)
Two hues meeting at a **neutral zero**. The project standard is **red = positive, blue =
negative**, endpoints Tailwind red-500 `rgb(239,68,68)` / blue-500 `rgb(59,130,246)`, with a
near-transparent slate midpoint. Critically, **alpha encodes magnitude** (`|t|·255`) so zero is
transparent — this is what makes the same canvas legible in light and dark themes. Normalise by
the largest absolute value in the tensor (`maxAbs`) so tiles are directly comparable, and show
`±maxAbs` in the legend. The shared implementation is `viz/heatmap.tsx` (`paintDiverging` +
`HeatmapTile` + `DivergingLegend`); `ModelWeights.tsx` (weight templates) and the Tensor route's
result view both use it.

### Status & semantics
`--primary` = active/headline · `--destructive` = error/failed run · `--muted-foreground` =
secondary text and inactive. Match the playground's capability panel (`routes/playground.tsx`).
The established **status-pill** pattern (see the training HUD) is
`bg-{color}-500/15 text-{color}-600 dark:text-{color}-400` — amber for running/pending,
emerald for done/ready, `destructive` for failed. HUD/overlay surfaces use
`bg-card/60`–`/70 backdrop-blur-md`.

### Rules
- **Never rely on color alone.** Pair it with a label, shape, position, or number
  (color-vision-deficiency safety). The weight legend spells out "blue = against, red = for".
- **One encoding per channel.** Don't use hue for both class *and* sign in one view.
- **Test both themes** before shipping — toggle `ThemeToggle` and confirm contrast.

---

## 5. Charts

Charts cover the **Performance** layer (loss/accuracy curves, latency, throughput
distributions) and aggregate views over `InferenceRun` metrics.

- **ECharts** for anything interactive, dense, or animated (training curves, large scatter,
  zoomable timelines). Import **only** through the lazy wrapper
  [`components/charts/EChart.tsx`](../../frontend/src/components/charts/EChart.tsx) via
  `lazy(() => import("@/components/charts/EChart"))` — the `echarts` bundle is heavy and must
  stay code-split out of the initial chunk.
- Small, composable marks (a sparkline, a single bar in a card) are hand-written **inline SVG** —
  a second charting library is not worth its weight for a 40px sparkline. Recharts used to be
  listed here as that option and was never imported once; it has been removed.
- Feed chart series from the **theme tokens** (`var(--chart-1)`…) so charts recolor with the
  theme; don't hard-code series colors. ECharts can't read CSS vars directly, so resolve them at
  runtime with `getCSSVar()` ([`lib/theme.ts`](../../frontend/src/lib/theme.ts)) keyed on the
  active theme. `LossChart`/`AccuracyChart` in `routes/training.tsx` do this via a
  `useChartTheme()` hook — copy that pattern for new ECharts views.
- Charts are the **Performance** layer only. Do **not** use a generic bar/line chart to depict
  model *structure* — structure is the stage/arrow schematic (§3). A histogram of weight values
  is fine (that's data about parameters); a "chart" standing in for the architecture is not.
- Keep axes labelled with units (`ms`, `tok/s`, `GFLOP/s`, `epoch`) and numbers `tabular-nums`.

---

## 6. Layout & composition patterns

Three composition patterns, in increasing ambition:

1. **Card** — the default. A `card.tsx` with header (identity), body (schematic or chart),
   footer (metrics). Use for registry listings, a single metric, a compact model summary.
2. **Schematic-as-stage** — the model *is* the page. A full-bleed background schematic on a
   pan/zoom canvas ([`training/PanZoom.tsx`](../../frontend/src/components/training/PanZoom.tsx):
   drag to pan, wheel to scroll, ⌘/Ctrl-wheel to zoom, double-click to fit), with controls and
   readouts floating in `dialog`/`popover`/`sheet` overlays and the loss/accuracy charts in a
   collapsible HUD. Use for the "explore this model" experience. This is the Training route.
3. **Split inspect** — schematic on one side, live runtime detail (inputs/outputs/activations)
   on the other. Use for interactive inference (LLM chat + token view, vision + camera).

Whichever pattern: the `AppLayout` sidebar/navbar stays untouched, overlays use the shadcn/ui
Base UI primitives (remember: `render` prop, no `asChild`; `ref` is a plain prop — see CLAUDE.md),
and heavy per-pixel drawing goes to `<canvas>`, heavy compute goes to the Web Worker.

---

## 7. Accessibility & performance

**Accessibility**
- Every canvas/heatmap gets a descriptive `aria-label` (`Weight template for digit 3`).
- Never encode meaning in color alone (§4). Provide the legend and numeric labels.
- Numeric readouts use `tabular-nums` so values don't reflow as they animate.
- Interactive overlays (dialogs, popovers) inherit Base UI's focus management — don't
  re-implement it.

**Performance**
- Per-pixel imagery (weights, activations, attention) → `<canvas>` + `putImageData`, **not**
  hundreds of DOM nodes. Use `[image-rendering:pixelated]` and a small intrinsic size scaled up
  with CSS.
- **Memoise normalization** (e.g. `maxAbs`) with `useMemo` keyed on the tensor — recompute only
  when weights change, not every render.
- Live weight/activation streams must arrive via the worker's transferred `ArrayBuffer`s
  (zero-copy); never round-trip large tensors through React state. The training path already
  does this — the worker streams a `WeightSnapshot` (`{ epoch, weights, bias }`, see
  `webgpu/workerClient.ts`) that `ModelWeights` renders directly.
- Charts stay lazy-loaded (§5). Don't pull `echarts` into a route's initial chunk.

---

## 8. Checklist — adding a visualization for a new model

When you add a model (see [`../guides/adding-a-model.md`](../guides/adding-a-model.md)), its
visualization should:

- [ ] Render **Identity** from the `ModelCard` (name, `titleCase(task)`, `formatBytes(size)`, license, visibility).
- [ ] Render **Structure** as a stage/arrow schematic driven by real `config` shapes — no GPU required.
- [ ] Render **Parameters** (if applicable) as canvas heatmap tiles with a diverging legend and `±maxAbs`.
- [ ] Render **Runtime** (if interactive) from worker results, streamed, off the main thread.
- [ ] Render **Performance** with the lazy `EChart` wrapper (or inline SVG for a sparkline), series colored from `--chart-*`.
- [ ] Reuse the `Stage` / `Arrow` / `ParamChip` / heatmap / `Legend` primitives — don't fork them.
- [ ] Pass **both themes** and **every WebGPU status** (structure still shows with no device).
- [ ] Give canvases `aria-label`s and pair every color with a label/number.

---

## Related

- In-browser inference pipeline: [`../explanations/webgpu-inference.md`](../explanations/webgpu-inference.md)
- Adding a model (kernel + registry entry): [`../guides/adding-a-model.md`](../guides/adding-a-model.md)
- Model/run API shapes: [`api-contracts.md`](api-contracts.md)
- Task taxonomy: [`../model-task-categories.md`](../model-task-categories.md)
- System architecture: [`../explanations/architecture.md`](../explanations/architecture.md)
</content>
