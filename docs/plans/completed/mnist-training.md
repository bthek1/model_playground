# Plan: MNIST Training UI Upgrade

**Status:** Complete — six core phases plus five follow-ups: dataset persistence
(Phase 7), a model-architecture explainer (Phase 8), promoting that schematic to
*be* the stage background (Phase 9), showing every parameter as the 10 weight
templates while retiring the Weights popover (Phase 10), and making the background
an interactive pan/zoom canvas (Phase 11). All implemented and unit-tested.
`fe-build` green; `fe-test` green at **206 tests / 35 files** (was 177); `fe-lint`
clean for the touched files (one pre-existing, unrelated error remains in
`webgpu/capabilities.ts`, untouched by this work). Remaining: manual end-to-end
pass on a real GPU browser to watch the weight templates sharpen into digit shapes
as training runs, and to feel the pan/zoom.
**Date:** 2026-07-08

---

## Goal

Rebuild the **Training** page ([`frontend/src/routes/training.tsx`](../../../frontend/src/routes/training.tsx))
around a **full-bleed animated visualization of the model itself** — its neurons,
weights, and connections — rendered as a live **background** that reacts to
training in real time. Configuration and controls move off the page flow and into
**floating overlays and popups**, so the network visualization is the primary
surface, not one card among many.

The **sidebar and navbar stay exactly as they are** — this is a change to the
`/training` route content only, inside the existing
[`AppLayout`](../../../frontend/src/components/layout/AppLayout.tsx) shell.

### What changes, at a glance

| Today | After this upgrade |
|-------|--------------------|
| Vertical stack of cards in `max-w-5xl` | Full-bleed viz canvas fills the `<main>` area |
| Dataset + hyperparameter cards always on screen | Settings live in **popup dialogs / popovers** |
| `ModelWeights` = 10 static weight tiles in a card | Weight templates **and** a live network-structure background |
| Stats + charts stacked below | Stats + charts float as **collapsible HUD overlays** |
| Sidebar / Navbar | **Unchanged** |

## Background & constraints

- **Raw WebGPU only for compute** (repo rule) — training math is untouched.
  The background visualization is **presentation**, so it may use a 2D `<canvas>`
  (or a lightweight WebGL/WebGPU render if perf demands). It reads the same
  `WeightSnapshot` stream the current `ModelWeights` component already consumes —
  no new data pathway into the trainer.
- **No new heavy work on the UI thread.** The viz animates from snapshots that
  already arrive rAF-batched via [`useLinearTraining`](../../../frontend/src/hooks/useLinearTraining.ts);
  the render loop is its own `requestAnimationFrame`, decoupled from React state.
- **Model is linear, 784 → 10.** 784 input nodes drawn literally is noise, so the
  structure viz is *stylized*: a sampled/aggregated node graph (see Phase 3), not
  a 784-dot hairball. Correctness of the weights display is preserved by keeping
  the existing 28×28 templates as a first-class element.
- **Base UI, not Radix** (repo rule). New overlay primitives compose via `render`,
  accept `ref` as a plain prop, and must not reintroduce `forwardRef`. Add them
  with `npx shadcn@latest add dialog popover` in the **`base-nova`** style.
- **Theme-aware.** The background must read correctly in both light and dark —
  derive colours from CSS variables / the existing `POS`/`NEG` diverging scheme,
  never hard-coded backgrounds.
- **Graceful WebGPU degradation.** When `useWebGPU().supported` is false, training
  can't run; the background still renders (static, zero-weights) and the start
  control stays disabled with the existing explanation.

## Phases

### Phase 1 — Overlay primitives (`src/components/ui/`)

Add the shadcn **`dialog`** and **`popover`** components (base-nova / Base UI).
Verify each is a thin wrapper that spreads `{...props}`, uses `render` for
composition, and carries no `forwardRef` — mirror the existing
[`sheet.tsx`](../../../frontend/src/components/ui/sheet.tsx) conventions. These are
generic, reusable infra (keep them domain-free).

- `dialog.tsx` — modal for the multi-field settings forms (Dataset, Hyperparameters).
- `popover.tsx` — anchored, non-modal for quick single-control tweaks and the
  legend/help.

### Phase 2 — Page shell restructure (`src/routes/training.tsx`)

Convert the page from a scrolling card stack to a **full-bleed stage**:

- A positioned container that fills `<main>` (`relative h-full`), with the
  background viz as an absolutely-positioned layer behind a foreground overlay
  layer (`pointer-events-none` on the layer, `pointer-events-auto` on controls).
- Keep the existing `useLinearTraining` + `useWebGPU` wiring and the `settings`
  state exactly as-is — only the *presentation* of these moves.
- **Foreground HUD** (floating, glassy `Card`-style panels):
  - top-left: title + run status (idle / training / done);
  - top-right: **Settings** button → Dataset dialog, **Tune** button → Hyperparameters dialog;
  - bottom-center: transport controls (Load MNIST · Start · Stop) as a floating bar;
  - a **live stat strip** (epoch, batch loss, train/test acc) as a compact overlay.

### Phase 3 — Network-structure background (`src/components/training/NetworkBackground.tsx`)

A new canvas component that visualizes the model's **parametrics, connections, and
structure** and animates with training:

- **Nodes:** 10 output neurons laid out (arc / column), each rendered from its
  28×28 weight template (reuse the `ModelWeights` colour math) so the node *is* a
  picture of what it detects. A stylized, down-sampled band of input nodes on the
  opposite side.
- **Connections:** edges input-band → output-neurons whose **opacity/width encodes
  weight magnitude** and **colour encodes sign** (reuse `POS`/`NEG` diverging
  scale). **Inputs are pooled into a coarse grid** (e.g. 7×7 = 49 pools over the
  28×28 field; each pool's edge weight is the summed/mean weight of its member
  pixels for that class), giving **49 → 10 = 490 legible edges** instead of 7,840.
  The pool grid also doubles as the stylized input-node band.
- **Live animation:** on each `WeightSnapshot`, tween edge weights toward the new
  values; a subtle pulse/flow along active edges conveys "training". Bias per class
  shown as a node glow.
- Own `requestAnimationFrame` loop, `ResizeObserver` for sizing, DPR-aware, pauses
  when not visible. Purely presentational — reads a `Float32Array` of weights + bias.

### Phase 4 — Settings as popups (`src/components/training/`)

Move the two config cards into overlays, preserving field semantics/validation:

- `DatasetDialog.tsx` — Load MNIST, train/test size fields, progress bar, loaded-count.
- `HyperparamsDialog.tsx` — learning rate, batch size, epochs (the current
  `NumberField`s), disabled while `training`.
- Extract the inline `NumberField` / `Stat` helpers into
  `src/components/training/controls.tsx` for reuse across overlays. Keep `DEFAULTS`.
- Settings changes still flow through the same `setSettings`/`start(settings)` path.

### Phase 5 — Metrics & templates as HUD overlays

- Keep `ModelWeights` (the 28×28 templates) but present it in a **collapsible
  floating panel** (popover or docked card) rather than a full-width card.
- Loss + Accuracy `EChart`s move into a collapsible bottom drawer/panel so they're
  available without dominating the stage. Lazy-load unchanged.
- Ensure all overlays are dismissible, keyboard-accessible, and don't trap scroll.

### Phase 6 — Polish & docs

- Motion: respect `prefers-reduced-motion` (freeze flow animation, keep static graph).
- Responsive: on narrow viewports, overlays stack / collapse; background still fills.
- Update this plan's status and add a short note to
  [`docs/explanations/webgpu-inference.md`](../../explanations/webgpu-inference.md)
  describing the visualization (docs travel with code).

### Phase 7 — Dataset persistence across reloads *(follow-up)*

**Problem:** the decoded MNIST pool lived only in a `useRef` inside
`useLinearTraining`, so a page reload dropped it and forced a full re-download +
re-decode. It should survive reloads.

- New `frontend/src/lib/mnistCache.ts` — persists the decoded `MnistPool` in
  **IndexedDB** (the pool is ~31 MB for 10k images — far past localStorage's
  limit; IndexedDB structured-clones typed arrays natively). All helpers are
  best-effort: absent IndexedDB (SSR/tests) or a quota error degrades to
  "no cache" instead of throwing. A read is validated (`isValidPool`) before use.
- `useLinearTraining` now:
  - **restores** a cached pool on mount (marks the dataset `ready` without a
    network call), and
  - in `loadData`, reuses an in-memory or cached pool that already covers the
    requested count, only downloading (and then `saveCachedPool`-ing) when the
    cache is missing or too small.

### Phase 8 — Model-architecture explainer *(follow-up)*

**Goal:** make the model *make intuitive sense* — show the concrete architecture:
`INPUT (784) ─► z = W·x + b ─► softmax ─► 10 class probs`, with the parameter
budget (7,840 W + 10 b = 7,850) and **colour encoding actual parameter values**.

- New `frontend/src/components/training/ModelArchitecture.tsx` — a three-stage
  schematic (Input · Weights+Bias · Output). Each output class is drawn as its
  **weight vector** pooled into a 28-cell coloured strip (red = +ve / for the
  digit, blue = −ve / against, alpha = magnitude) plus a signed, colour-coded bias
  chip. Reads the live weights, so strips fill in with structure as training runs
  (`transition-colors` eases the change each epoch); faint before training with a
  "Start training…" hint. Reuses the POS/NEG diverging scheme; theme-aware;
  responsive (stages stack, arrows rotate on narrow screens).

### Phase 9 — Architecture schematic *is* the background *(follow-up)*

Replaced the abstract pooled node-graph background with the Phase-8 schematic
itself: the `ModelArchitecture` diagram is now the full-bleed, centered background
of the Training stage (non-interactive layer), and the HUD panels float over it.

- `routes/training.tsx` renders `ModelArchitecture` as the background layer;
  the separate **Model** dialog button is removed (the schematic is always on
  screen, so it needs no dialog).
- **Removed** `frontend/src/components/training/NetworkBackground.tsx` (+ its test)
  — superseded; the schematic is the single, legible model visualization now.

### Phase 10 — Show every parameter; retire the Weights popover *(follow-up)*

The schematic's middle stage now shows **every parameter**, not a pooled summary:

- `ModelArchitecture` renders the **INPUT** as a real 28×28 pixel grid, and the
  **WEIGHTS + BIAS** stage as the 10 per-class `ModelWeights` templates (each a
  28×28 image of all 784 weights, red = for / blue = against, plus the bias) —
  reusing the existing `ModelWeights` component. All 7,840 weights + 10 biases are
  individually visible inline in the diagram; a hint shows until the weights carry
  signal. `snapshot.epoch` is forwarded through for the template legend.
- **Removed the Weights popover** from `routes/training.tsx` (and its `Grid3x3` /
  `ModelWeights` imports) — it's redundant now that the templates live in the
  background schematic. The Charts popover stays.

### Phase 11 — Pan/zoom canvas background *(follow-up)*

The background was a plain `overflow-auto` scroll box; make it an interactive
diagram canvas instead.

- New `frontend/src/components/training/PanZoom.tsx` — a transform-based surface:
  **drag to pan**, **wheel to scroll** (x/y), **⌘/Ctrl-wheel or +/− buttons to
  zoom** (cursor-anchored), **double-click / fit button to recenter**, with a live
  zoom-% readout. Content is a single translate+scale layer; all gestures mutate
  `{scale, x, y}`. `MIN_SCALE`/`MAX_SCALE` clamp. Wheel is a native non-passive
  listener so it can `preventDefault` the page scroll.
- `routes/training.tsx` wraps the schematic in `<PanZoom>` (a `w-max` content
  layer so the diagram takes its natural width and pans as one canvas — no nested
  scroll), replacing the old centered scroll box.
- Scaled the diagram up now that the canvas can pan/zoom: `ModelWeights` gained a
  `tileSize` prop (default `size-16`); the schematic uses `size-32` templates
  (128px), a `w-40` input grid, `size-10` output nodes, and big `⟶` connectors so
  the parameters and connections read clearly.

## Files touched

**New**
- `frontend/src/components/ui/dialog.tsx`, `frontend/src/components/ui/popover.tsx`
- `frontend/src/components/training/DatasetDialog.tsx`, `HyperparamsDialog.tsx`
- `frontend/src/components/training/controls.tsx`
- `frontend/src/lib/mnistCache.ts` *(Phase 7)*
- `frontend/src/components/training/ModelArchitecture.tsx` *(Phase 8; the stage background as of Phase 9)*
- `frontend/src/components/training/PanZoom.tsx` *(Phase 11 — pan/zoom canvas)*

**Modified**
- `frontend/src/routes/training.tsx` (shell + overlays; schematic background,
  Weights popover removed, wrapped in `PanZoom` — Phases 9–11)
- `frontend/src/components/training/ModelWeights.tsx` — now also embedded in the
  `ModelArchitecture` schematic (Phase 10); component itself unchanged
- `frontend/src/hooks/useLinearTraining.ts` (cache restore + reuse/save — Phase 7)

**Removed**
- `frontend/src/components/training/NetworkBackground.tsx` (+ test) — the pooled
  node-graph background, superseded by the schematic in Phase 9.

**Unchanged (explicitly)**
- `AppLayout.tsx`, `Sidebar.tsx`, `Navbar.tsx`, `navItems.ts`
- `linearModel.ts`, `ModelWeights.tsx`, worker/compute pipeline

## Testing

**Done — automated (Vitest + Testing Library, all green):**

- `src/components/training/controls.test.tsx` *(new, 8 tests)* — `pct`
  formatting; `NumberField` renders label/value, emits parsed ints and unrounded
  floats through `onChange`, respects `disabled`; `Stat` renders label/value and
  omits the optional sub line. *(Note: happy-dom's number input coerces empty →
  `0`, so the `Number.isFinite` guard isn't reachable via the field in-env; it
  stays as defensive code.)*
- `src/components/training/HyperparamsDialog.test.tsx` *(new, 4 tests)* — closed
  until trigger click; edits route through `set(...)`; integer fields round;
  all fields locked while `training`.
- `src/components/training/DatasetDialog.test.tsx` *(new, 6 tests)* — opens on
  trigger; Load MNIST calls `loadData(trainSize + testSize)`; size edits call
  `set(...)`; ready-summary shows the loaded count; dataset error surfaces;
  load button disabled while loading.
- `src/__tests__/routes/training.test.tsx` *(updated)* — rewritten for the
  popup-driven UI: heading renders; the **architecture schematic** is present as
  the background (title + `7,850` budget); WebGPU-unavailable notice toggles;
  **Dataset** popup gates Load MNIST and the dataset error; **Start** disabled
  until ready and dispatches `start(settings)`; **Stop** enabled while training;
  stat strip shows streamed metrics; **Charts** / **Weights** popovers reveal the
  lazy charts and the per-digit weight templates.
- `src/lib/mnistCache.test.ts` *(new, Phase 7, 3 tests)* — with `indexedDB`
  stubbed absent, `loadCachedPool` → `null` and `saveCachedPool` / `clearCachedPool`
  resolve without throwing (the degrade-gracefully contract).
- `src/hooks/useLinearTraining.test.ts` *(updated, Phase 7, +4 tests)* — mocks
  `@/lib/mnistCache`: a fresh download is persisted via `saveCachedPool`; a cached
  pool is **restored on mount** (ready, no download) and **reused in `loadData`**
  (no download, no re-save); a too-small cached pool triggers a re-download.
- `src/components/training/ModelArchitecture.test.tsx` *(new, Phase 8; updated
  Phase 10, 4 tests)* — title + parameter budget (7,840 / 7,850); the 28×28 input
  grid; a 28×28 weight template for every class (10 canvases); the "start training"
  prompt until the weights carry signal. *(Phase 9 removed the 3 `NetworkBackground`
  tests with the component; Phase 10 dropped the pooled-strip/bias-chip assertions
  in favour of the per-class templates and moved the route templates test off the
  now-removed Weights popover.)*

- `src/components/training/PanZoom.test.tsx` *(new, Phase 11, 3 tests)* — renders
  children + zoom controls with a 100% readout; a pan drag gesture doesn't throw;
  the zoom-in button is clickable. *(happy-dom has no layout, so `fit()` no-ops and
  the transform stays at its initial value — the tests assert mount + interaction
  safety, not pixel math.)*

Full suite: **206 tests / 35 files** green (was 177).

**Done — build/lint:** `just fe-build` (type-check + production build) green;
`fe-lint` clean on all touched files (one pre-existing, unrelated
`no-useless-assignment` error remains in `webgpu/capabilities.ts`, untouched here).

**Pending — manual (real GPU browser, can't run headless):** load MNIST, start
training, confirm the background edges animate toward sharpening digit templates,
stats + charts update live, overlays open/close without blocking training, sidebar
and navbar remain intact and functional. **Then reload the page and confirm the
dataset stays `ready` (no re-download) — the Phase 7 IndexedDB persistence.**

## Open questions

1. **Background renderer:** 2D canvas (simplest, recommended) vs a WebGPU/WebGL
   particle render for the "flow" effect? Start with 2D; revisit only if it can't
   hold 60fps.
2. ~~Connection sampling strategy~~ — **Decided: pooled input regions** (7×7
   grid → 490 edges). Pool size (7×7 vs a finer 14×14) can be tuned during impl.
3. **Overlay defaults:** should the metrics/charts panel start open or collapsed
   on first visit?
