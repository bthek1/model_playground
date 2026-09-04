# Model Page Horizontal Layout

**Status:** Complete
**Created:** 2026-09-04
**Completed:** 2026-09-04
**Scope:** `frontend/src/components/model/` (shell + four slots), every task route,
`docs/standards/model-page-pattern.md`, route tests + E2E.

---

## 1. Why

Every task page today is a single 896 px column (`max-w-4xl`) with the four stages —
SELECT → LOAD → RUN → OUTPUT — stacked vertically ([`ModelPage.tsx`](../../frontend/src/components/model/ModelPage.tsx)).
That reads well on a phone and badly on the 16:9 display everyone actually uses:

- **The output is below the fold.** On a 1080p laptop, SELECT + LOAD + a textarea
  consume the viewport, so the result — the reason the page exists — lands off-screen
  and the user scrolls back and forth between the input they are editing and the
  output they are judging. For ASR and Audio-to-Audio (before/after waveforms) that
  round trip happens on every single run.
- **Two-thirds of the window is empty.** A 1440–1920 px viewport shows a 896 px column
  centred in whitespace. The bands that need width (waveforms, heatmaps, transcript
  with timestamps) are the ones squeezed; the bands that need almost none (SELECT's
  button row, LOAD's one-line status) get the same full width.
- **Setup and work are weighted equally.** SELECT and LOAD are done once per session.
  RUN and OUTPUT are used on every iteration. The vertical stack gives all four the
  same prominence and pushes the repeated pair furthest down.

The four-stage *pipeline* is not the problem and does not change. What changes is the
**arrangement**: setup collapses into a compact rail, and the iteration loop —
input beside output — sits side by side in the space that frees up.

Non-goals: no change to the state machines (§2 of the standard), the hook contract
(§3), the worker protocol, or any task's behaviour. This is layout only.

---

## 2. Target layout

```
┌───────────────────────────────────────────────────────────────────────────────┐
│  ModelPageHeader     icon · title · one-line what/where          [backend chip]│
├──────────────────────┬────────────────────────────────────────────────────────┤
│  SETUP RAIL (~320px) │  WORKBENCH (fills)                                     │
│                      │  ┌──────────────────────┬───────────────────────────┐  │
│  ① MODEL             │  │ ③ INPUT              │ ④ OUTPUT                  │  │
│  ┌────────────────┐  │  │                      │                           │  │
│  │ model list     │  │  │ the task's input     │ empty → running → result  │  │
│  │ (vertical)     │  │  │ surface              │                           │  │
│  │ hint · params  │  │  │                      │ meta · actions            │  │
│  │ size · ⚠ large │  │  │ ── transport row ──  │                           │  │
│  └────────────────┘  │  │ [Run]  [Clear]       │                           │  │
│                      │  │                      │                           │  │
│  ② LOAD              │  │ input errors here    │ run errors here           │  │
│  ┌────────────────┐  │  └──────────────────────┴───────────────────────────┘  │
│  │ idle → button  │  │                                                        │
│  │ loading → bar  │  │  Both columns are independently scrollable; the        │
│  │ ready → chip   │  │  header and the rail never scroll away.                │
│  │ error → retry  │  │                                                        │
│  └────────────────┘  │                                                        │
└──────────────────────┴────────────────────────────────────────────────────────┘
```

**Breakpoints** (Tailwind v4 defaults):

| Width | Layout |
|---|---|
| `< md` (< 768) | Today's single column, all four bands stacked in pipeline order. |
| `md … xl` | Two rows: setup rail becomes a **horizontal** SELECT + LOAD strip across the top; INPUT and OUTPUT side by side below. |
| `≥ xl` (1280+) | The full three-column arrangement above: rail + input + output. |

The page container goes from `max-w-4xl` to `max-w-[1600px]` with a viewport-height
flex shell (`h-[calc(100dvh-…)]`) so the two work columns scroll inside themselves
rather than the page scrolling as one long sheet.

**What survives unchanged:** all four stages are always present and always in
pipeline order (left-to-right, top-to-bottom); each is still a labelled `region`
with its step number; `data-testid="slot-1"`…`slot-4` keep their meanings; errors
still render in the slot that produced them.

**Ordering claim to defend in review:** reading order in the DOM stays 1 → 2 → 3 → 4
regardless of breakpoint, so keyboard and screen-reader traversal matches the numbered
pipeline even when the visual arrangement is two-dimensional. The grid places items by
area, not by reordering source.

---

## 3. Phases

### Phase 1 — Layout primitives in the shell

Rework [`ModelPage.tsx`](../../frontend/src/components/model/ModelPage.tsx) only. No
route touched; every existing page picks the new arrangement up for free.

- Replace the `max-w-4xl space-y-8` column with a CSS-grid shell:
  `grid-template-areas` per breakpoint (`"setup"/"input"/"output"` → `"setup setup"/"input output"`
  → `"setup input output"`), so the DOM order never changes.
- Add a `SetupRail` wrapper grouping slots 1 and 2 with a shared border, so the rail
  reads as one "configure the model" surface rather than two floating bands.
- Give `ModelSlot` a `dense` variant: smaller heading, tighter spacing, used inside the
  rail where the content is a list and a status line.
- Make the two work columns `min-h-0 overflow-y-auto` inside a `h-full` flex parent —
  without `min-h-0` a grid child refuses to shrink and the inner scroll never engages.
- Add an optional `aside` prop for the header's right edge (backend chip / device
  status), so `DeviceStatus` pages have somewhere to put a one-line answer that isn't
  a full band.

### Phase 2 — Slot components adapted to their new width

- **`ModelPicker`**: add a `layout: "row" | "list"` prop. `list` (rail default) renders
  models as full-width stacked buttons with label, hint and size on each row — a wrapped
  button row in a 320 px rail is unreadable. `row` keeps today's behaviour for the
  `md…xl` horizontal strip. Size note and large-model warning are unchanged in content.
- **`ModelStatus`**: compact variant — `ready` becomes a single chip (`✓ Ready · WEBGPU`),
  progress bar spans the rail width, error keeps its full message + Retry.
- **`InputPanel`**: gains a sticky bottom transport row so the Run button stays visible
  when a long input scrolls. `disabledHint` unchanged.
- **`OutputPanel`**: fills its column height (`h-full`), with the result area scrolling
  and `meta`/`actions` pinned to the card header/footer. The empty state grows a centred
  vertical treatment now that it owns a tall column.

### Phase 3 — Route audit

Each route only supplies slot content, so most need nothing. Audit and fix:

- [`text-to-speech.tsx`](../../frontend/src/routes/text-to-speech.tsx) — textarea grows to fill the input column.
- [`asr.tsx`](../../frontend/src/routes/asr.tsx) — live transcript is the tall element; verify the take-relative timestamp list scrolls inside OUTPUT, not the page.
- [`audio-classification.tsx`](../../frontend/src/routes/audio-classification.tsx) — label bars gain width; no change expected.
- [`audio-to-audio.tsx`](../../frontend/src/routes/audio-to-audio.tsx) — before/after waveforms stack vertically in the output column (they currently sit side by side); this is the biggest win and the biggest risk.
- [`text-to-audio.tsx`](../../frontend/src/routes/text-to-audio.tsx) — the MusicGen opt-in gate stays in LOAD, unmoved.
- [`tensor.tsx`](../../frontend/src/routes/tensor.tsx) — operand matrices in INPUT, heatmap in OUTPUT; check the heatmap's fixed sizing against a narrower-but-taller column.
- [`tasks.$slug.tsx`](../../frontend/src/routes/tasks.%24slug.tsx) — the empty placeholder must still read as the same kind of page.
- [`training.tsx`](../../frontend/src/routes/training.tsx) — **exempt** (§7 of the standard: full-bleed canvas + HUD). Untouched; the exemption note gets a sentence saying it also opts out of the horizontal shell.

### Phase 4 — Docs and standard

- Rewrite §4 of [`model-page-pattern.md`](../standards/model-page-pattern.md) around the
  new diagram, with the breakpoint table and the "DOM order is pipeline order" rule.
- Add a §4a "Layout" subsection: which slot belongs in the rail, which in the workbench,
  and why the split is setup-vs-iteration rather than 1–2-vs-3–4 by accident.
- Update §8 to note the testids are unchanged, and add the `md` breakpoint to what route
  tests assume (jsdom/happy-dom has no layout — tests assert presence and order, never
  geometry).
- Mirror the change in [`.github/copilot-instructions.md`](../../.github/copilot-instructions.md)
  (the CLAUDE.md counterpart) where it summarises the four-slot shell.

---

## 4. Risks

| Risk | Mitigation |
|---|---|
| Inner scroll containers break the page on short viewports | Below `md` the shell falls back to today's single flowing column with no height clamp. |
| A grid child won't shrink, so the output column pushes the page wide | `min-h-0` / `min-w-0` on both work columns; caught by an E2E assertion that the body has no horizontal scroll. |
| Tests key off band *position* rather than testid | `slot-N` is bound to the step number, not to source position; grid placement never reorders the DOM. |
| Audio-to-Audio's dual waveforms are width-hungry | Stack them vertically in the output column and let the column scroll; verify with the `@slow` enhancement spec still passing visually. |
| The rail makes SELECT feel permanent while it must disable mid-flight | The `disabled` gate on `ModelPicker` is unchanged. Dimming the rail as a whole was dropped: it would have dimmed LOAD's progress bar, the one thing the user is watching at that moment. |

---

## 5. Testing

**Unit / component (Vitest, `just fe-test`)**

- `ModelPage`: renders exactly four bands, `slot-1`…`slot-4` present, and in that DOM
  order — the assertion that guards grid placement against source reordering.
- `ModelPage`: `aside` renders when supplied and the header is unchanged without it.
- `ModelPicker`: both `layout` values render every model, the size note and the
  large-model warning; `disabled` still blocks selection in either layout.
- `ModelStatus` compact variant: all five states (idle / loading / warmup / ready /
  error) still render their identifying text, and `model-ready` still carries the backend.
- `OutputPanel`: `output-empty`, `output-running`, result and error states unchanged
  under the new height handling.

**Route tests (Vitest, per §8 of the standard) — re-run all of them unchanged.**
The list is the regression suite for this plan: nothing downloads on mount, `load`/`retry`
fire from the LOAD slot, four slots with `output-empty` before any run, run controls
disabled until `ready`, errors in the producing slot. If a route test needs editing,
that is a signal the layout change leaked into behaviour.

**E2E (Playwright, `just fe-e2e`)**

- Extend `e2e/specs/model-page.spec.ts`: at 1440×900 assert `slot-3` and `slot-4` have
  non-overlapping bounding boxes at the same vertical offset (side by side), and that
  `document.body.scrollWidth <= clientWidth` (no horizontal overflow).
- At 375×812 assert the same four slots render stacked (increasing `y`).
- Assert the output panel is within the initial viewport without scrolling on a
  1440×900 page — the concrete statement of §1's complaint.
- `just fe-e2e-enhance` (`@slow`) re-run for Audio-to-Audio, since its output column
  changed the most.

**Manual**

- 1920×1080, 1440×900, 1280×800, and a 13" laptop at 125% scale.
- Keyboard traversal: Tab order must be model → load → input → run → output actions.
- Light and dark, since the rail introduces a new border/background surface.

---

## 6. Done when

- [ ] `ModelPage` renders the grid shell with the three documented breakpoints
- [ ] `SetupRail` + `dense` slot variant in place; slots 1–2 live in the rail
- [ ] `ModelPicker` list layout, compact `ModelStatus`, filling `OutputPanel`
- [ ] Every task route audited; `training.tsx` explicitly exempt and untouched
- [ ] All existing route tests pass **without edits**
- [ ] New layout assertions in `model-page.spec.ts` pass at both viewports
- [ ] §4 of `model-page-pattern.md` rewritten; copilot instructions mirrored
- [ ] Status moved to `Complete` and this file `git mv`d to `docs/plans/completed/`

---

## 7. What changed against the plan

Three things the plan did not anticipate, all found while building:

- **`ModelStatus` lost its `size` prop entirely.** With the picker directly above it in
  the rail, both quoted the same download and the rail said the number twice. The
  size-before-load guardrail is unweakened — it is now stated once, in SELECT, where the
  choice is actually made. Removing the prop also made `meta` dead in three routes.
- **The transport row is sticky, not bottom-pinned.** `mt-auto` put Run at the foot of
  the input column, which reads well for a full-height textarea and leaves a chasm on
  ASR, whose input is three sample-clip buttons. `sticky bottom-0` gives the same
  reachability guarantee with none of the gap.
- **The setup rail hugs its content (`xl:self-start xl:max-h-full`)** instead of
  stretching. A full-height bordered panel holding two short bands was a large grey
  rectangle of nothing.

Also: `/tensor`'s operand schematic lost its `xl:flex-row`. A ~600px input column has no
room to run Matrix A beside Matrix B; it stacks at every width now, arrow included.

**Verification as run:** `just fe-test` 571 passed / 78 files, with **no route test
edited** — the criterion this plan set for "layout only". `tsc --noEmit` clean, `eslint`
clean (5 pre-existing `react-refresh` warnings in unrelated files). `just fe-e2e` 68
passed / 5 skipped, including seven new arrangement specs. Visually checked at
1440×900, 1000×800 and 375×812 across `/text-to-speech`, `/asr` and `/tensor`.

**Where each assertion landed.** Structure is Vitest, geometry is Playwright — happy-dom
reports zeroes for every box, so a layout assertion there would pass on any layout at all:

| Decision | Asserted in |
|---|---|
| Four bands, DOM order 1→2→3→4 under grid-area placement | `ModelPage.test.tsx` |
| Rail holds slots 1–2; slots 3–4 in separate columns | `ModelPage.test.tsx` |
| Header `aside` renders, and stays out of the four slots | `ModelPage.test.tsx` |
| `dense` changes spacing only — step, heading, region unchanged | `ModelPage.test.tsx` (`ModelSlot`) |
| Picker `list` rows carry the hint; `row` chips keep it in `title` | `ModelPicker.test.tsx` |
| Size guardrail identical in both layouts; disabled in both | `ModelPicker.test.tsx` |
| LOAD no longer repeats the download estimate | `ModelStatus.test.tsx` |
| Transport follows the fields, carries the error, no `mt-auto` | `InputPanel.test.tsx` |
| INPUT/OUTPUT side by side, vertically aligned @1440×900 | `model-page.spec.ts` |
| Result above the fold @1440×900 | `model-page.spec.ts` |
| Setup strip above two work columns @1000×800 | `model-page.spec.ts` |
| Four bands stack in increasing `y` @375×812 | `model-page.spec.ts` |
| No horizontal document overflow, desktop and phone | `model-page.spec.ts` |
| No chasm between a short input and its transport | `model-page.spec.ts` |
| OUTPUT fills its column; the rail hugs its content | `model-page.spec.ts` |

The chasm spec was checked against the behaviour it guards: reinstating `mt-auto`
produces a 456px gap against a 120px threshold, and the spec fails.

**Docs updated:** `model-page-pattern.md` (§4 rewritten, §4a breakpoints added, §5, §7,
§8, §9), `.github/copilot-instructions.md`, `CLAUDE.md`, `.github/agents/frontend.agent.md`,
`docs/guides/e2e-testing.md`.
