# Sidebar Task Taxonomy

**Status:** Complete

Restructure the app sidebar from a flat 3-item list into a categorized, collapsible
navigation tree modelled on the Hugging Face task taxonomy (Audio, Computer Vision,
Multimodal, NLP, Other, Reinforcement Learning, Tabular, Theory).

---

## Context

Today the sidebar is a flat list defined in
[`frontend/src/components/layout/navItems.ts`](../../../frontend/src/components/layout/navItems.ts)
and rendered by [`Sidebar.tsx`](../../../frontend/src/components/layout/SidebarNav.tsx).
`SidebarNav` is shared by the desktop `Sidebar` and the mobile drawer in
[`Navbar.tsx`](../../../frontend/src/components/layout/Navbar.tsx). Expand/collapse of the
whole rail is tracked by `sidebarOpen` in [`store/ui.ts`](../../../frontend/src/store/ui.ts).

We want to group navigation under the following two-level taxonomy. Only a handful of
these tasks have real routes today (`/playground`, `/tensor`, `/training`); the rest are
**catalog placeholders** — they should render but route to a generic "coming soon" / task
landing view rather than 404.

### Target structure

- **Audio** — Text to Speech · Text to Audio · Automatic Speech Recognition · Audio to Audio · Audio Classification · Voice Activity Detection
- **Computer Vision** — Depth Estimation · Image Classification · Object Detection · Image Segmentation · Text to Image · Image to Text · Image to Image · Image to Video · Unconditional Image Generation · Video Classification · Text to Video · Zero Shot Image Classification · Mask Generation · Zero Shot Object Detection · Text to 3D · Image to 3D · Image Feature Extraction · Keypoint Detection · Video to Video
- **Multimodal** — Audio Text to Text · Image Text to Text · Image Text to Image · Image Text to Video · Visual Question Answering · Document Question Answering · Video Text to Text · Visual Document Retrieval · Any to Any
- **Natural Language Processing** — Text Classification · Token Classification · Table Question Answering · Question Answering · Zero Shot Classification · Translation · Summarization · Feature Extraction · Text Generation · Fill Mask · Sentence Similarity · Text Ranking
- **Other** — Graph Machine Learning
- **Reinforcement Learning** — Reinforcement Learning · Robotics
- **Tabular** — Tabular Classification · Tabular Regression · Time Series Forecasting
- **Theory** — Mathematics Foundations · Discrete Maths · Deep Learning Equations

---

## Design decisions

- **Data-driven.** The taxonomy is one exported constant; the component renders it. Adding
  a task is a one-line data edit, not a component change. Keep it generic/reusable per the
  repo's template ethos.
- **Two levels, collapsible categories.** Each category is a disclosure header; its tasks
  render underneath when expanded. Slugs derive from the task label
  (`"Text to Speech" → "text-to-speech"`).
- **Every task links somewhere.** Tasks with a real route link to it; the rest link to a
  generic task route `/tasks/$slug` that renders a placeholder landing page. No dead links,
  no 404s.
- **Collapsed rail = icons only.** When `sidebarOpen` is false, show one icon per
  *category* (with the category name as `title`); clicking expands the rail rather than
  showing a nested flyout (keeps scope small; flyout can be a follow-up).
- **Per-category expanded state** lives in the UI store so it persists across the
  desktop/mobile shared `SidebarNav` and across navigation. Active task's category
  auto-expands on load.
- **Reuse existing primitives.** Base UI + shadcn `Collapsible` (add via
  `npx shadcn@latest add collapsible`) or a plain `<button>`-toggled section — no Radix,
  use the `render` prop convention. Icons from `lucide-react`.

---

## Phase 1 — Taxonomy data model

- Add `frontend/src/components/layout/taskTaxonomy.ts` exporting:
  - `TaskItem { label: string; slug: string; to: string }`
  - `TaskCategory { label: string; icon: LucideIcon; tasks: TaskItem[] }`
  - `taskCategories: TaskCategory[]` covering the full structure above.
- `to` resolves to the real route when one exists (map `Text Generation`/others to
  `/playground`, `Tensor Arithmetic`-style utilities as needed), otherwise `/tasks/${slug}`.
- Pick one representative `lucide-react` icon per category (e.g. `AudioLines`, `Eye`,
  `Layers`, `Type`, `Boxes`, `Gamepad2`, `Table`, `Sigma`).
- Keep the legacy `navItems.ts` exports until Phase 4 removes their last caller, to avoid a
  broken build mid-way.

## Phase 2 — Store: per-category expansion

- Extend [`store/ui.ts`](../../../frontend/src/store/ui.ts) with:
  - `expandedCategories: Record<string, boolean>`
  - `toggleCategory(label: string)` and `setCategoryExpanded(label, open)`.
- Default: all categories collapsed **except** the one containing the active route (compute
  lazily in the component, or seed on mount). Keep the Immer pattern already in the file.

## Phase 3 — Sidebar rendering

- Rewrite `SidebarNav` in
  [`Sidebar.tsx`](../../../frontend/src/components/layout/Sidebar.tsx) to render
  `taskCategories`:
  - **Expanded rail:** category header row (icon + label + chevron that rotates on open),
    then the task links indented beneath when the category is expanded. Active task keeps
    the existing `aria-current="page"` + `bg-sidebar-accent` styling.
  - **Collapsed rail:** one centered category icon per group with `title={category.label}`;
    clicking it calls `setSidebarOpen(true)` and expands that category.
  - Wrap the nav in an `overflow-y-auto` container — the list is long and must scroll within
    the fixed-height `aside`/drawer without pushing the logo/footer.
- Keep `SidebarNav` signature (`collapsed`, `onNavigate`) so `Navbar.tsx`'s mobile drawer
  keeps working unchanged; `onNavigate` still fires on task-link click to close the drawer.

## Phase 4 — Generic task route + cleanup

- Add `frontend/src/routes/tasks.$slug.tsx` (TanStack Router file route): looks the slug up
  in the taxonomy, renders the task label + a "not yet available" placeholder for
  unimplemented tasks, or redirects to the real route if one is mapped.
- Repoint any remaining `navItems` importers to the taxonomy; delete `navItems.ts` once
  unused (confirm with `grep -rn navItems frontend/src`).
- Regenerate the route tree (`just fe-dev` / vite plugin) so `tasks.$slug` is registered.

## Phase 5 — Docs

- Update [`CLAUDE.md`](../../../CLAUDE.md) and
  [`.github/copilot-instructions.md`](../../../.github/copilot-instructions.md) frontend
  sections to note the taxonomy-driven sidebar and where to add a task.
- If any API/route contract changes, reflect it in
  [`docs/standards/api-contracts.md`](../../standards/api-contracts.md) (none expected —
  this is frontend-only).
- Move this plan to `docs/plans/completed/` when `Status` reaches `Complete`.

---

## Testing

- **Unit (`just fe-test`):** update
  [`Sidebar.test.tsx`](../../../frontend/src/components/layout/Sidebar.test.tsx):
  - Renders every category header.
  - Expanding a category reveals its tasks; collapsing hides them (assert via store toggle).
  - Active task within a category gets `aria-current="page"`.
  - Collapsed rail shows category icons with `title` and hides task labels.
  - The category containing the active route is expanded on initial render.
- **Route test:** `/tasks/$slug` renders the correct label for a known slug and a graceful
  placeholder for one without a real route.
- **Lint/build:** `just fe-lint` and `just fe-build` clean (type-check the new data model
  and route).
- **Manual:** `just fe-dev` — verify long list scrolls, expansion persists across
  navigation, mobile drawer (`Navbar`) still opens/closes on task click, and collapsed rail
  round-trips to expanded.

---

## Out of scope / follow-ups

- Flyout submenus on the collapsed rail (hover to reveal a category's tasks).
- Search/filter box over tasks.
- Wiring individual tasks to real WebGPU model routes (each is its own
  [adding-a-model](../../guides/adding-a-model.md) effort).
- Backend `ModelCard.pipeline_tag` alignment with this taxonomy.
