# Plan: MNIST Training UI Upgrade

**Status:** Complete — all phases implemented; `fe-build` + `fe-test` (177 tests,
incl. new `NetworkBackground` + rewritten `training` route tests) green; `fe-lint`
clean for the touched files (one pre-existing, unrelated error remains in
`webgpu/capabilities.ts`). Remaining: manual end-to-end pass on a real GPU browser
to watch the background edges animate as training runs.
**Date:** 2026-07-07

---

## Goal

Rebuild the **Training** page ([`frontend/src/routes/training.tsx`](../../frontend/src/routes/training.tsx))
around a **full-bleed animated visualization of the model itself** — its neurons,
weights, and connections — rendered as a live **background** that reacts to
training in real time. Configuration and controls move off the page flow and into
**floating overlays and popups**, so the network visualization is the primary
surface, not one card among many.

The **sidebar and navbar stay exactly as they are** — this is a change to the
`/training` route content only, inside the existing
[`AppLayout`](../../frontend/src/components/layout/AppLayout.tsx) shell.

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
  already arrive rAF-batched via [`useLinearTraining`](../../frontend/src/hooks/useLinearTraining.ts);
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
[`sheet.tsx`](../../frontend/src/components/ui/sheet.tsx) conventions. These are
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
- Update [`docs/plans/mnist-training.md`](mnist-training.md) status and add a short
  note to [`docs/explanations/webgpu-inference.md`](../explanations/webgpu-inference.md)
  or a training-UI section describing the visualization (docs travel with code).

## Files touched

**New**
- `frontend/src/components/ui/dialog.tsx`, `frontend/src/components/ui/popover.tsx`
- `frontend/src/components/training/NetworkBackground.tsx`
- `frontend/src/components/training/DatasetDialog.tsx`, `HyperparamsDialog.tsx`
- `frontend/src/components/training/controls.tsx`

**Modified**
- `frontend/src/routes/training.tsx` (shell + overlays)
- `frontend/src/components/training/ModelWeights.tsx` (only if factoring out colour math)

**Unchanged (explicitly)**
- `AppLayout.tsx`, `Sidebar.tsx`, `Navbar.tsx`, `navItems.ts`
- `useLinearTraining.ts`, `linearModel.ts`, worker/compute pipeline

## Testing

- **Unit (Vitest + Testing Library):**
  - `dialog` / `popover` open, close (Esc + outside-click), and render children.
  - `NetworkBackground` mounts, acquires a 2D context (mock canvas), and redraws
    on a new weights prop without throwing; degrades to a static frame with
    zero weights.
  - `DatasetDialog` / `HyperparamsDialog`: field edits call `set(...)`; fields
    disabled while `training`; defaults from `DEFAULTS`.
  - Update `src/__tests__/routes/training.test.tsx`: page renders inside the
    layout, Settings/Tune buttons open their dialogs, Start disabled until dataset
    ready, stat strip reflects streamed metrics.
- **Build/lint:** `just fe-build` (type-check) + `just fe-lint` green.
- **Manual (real GPU browser — can't run headless):** load MNIST, start training,
  confirm the background edges animate toward sharpening digit templates, stats +
  charts update live, overlays open/close without blocking training, sidebar and
  navbar remain intact and functional.

## Open questions

1. **Background renderer:** 2D canvas (simplest, recommended) vs a WebGPU/WebGL
   particle render for the "flow" effect? Start with 2D; revisit only if it can't
   hold 60fps.
2. ~~Connection sampling strategy~~ — **Decided: pooled input regions** (7×7
   grid → 490 edges). Pool size (7×7 vs a finer 14×14) can be tuned during impl.
3. **Overlay defaults:** should the metrics/charts panel start open or collapsed
   on first visit?
