# Model Page Pattern

Every task page in Model Playground is the same four-stage pipeline:

```
  ┌────────┐    ┌──────┐    ┌─────┐    ┌────────┐
  │ SELECT │───▶│ LOAD │───▶│ RUN │───▶│ OUTPUT │
  └────────┘    └──────┘    └─────┘    └────────┘
   which        weights      give it     show what
   model?       → memory     an input    came back
```

Text-to-Speech, ASR, Audio Classification, Image Classification, Depth Estimation,
Object Detection — the modality changes, the pipeline does not. This document is the
contract every model page implements, so that a new task is *filling in four slots*
rather than inventing a page.

It is the interaction counterpart to [`model-visualization.md`](model-visualization.md):
that document covers **how we draw a model**, this one covers **how a user drives it**.

---

## 1. Principles

1. **The four stages are always visible, in order.** A user should never wonder which
   step they are on. Stages that don't apply are rendered as a completed/skipped state,
   not omitted — omission makes two task pages look structurally different when they
   aren't.
2. **Loading is explicit and consented.** Model weights are the user's bandwidth and the
   tab's memory budget. Show the size *before* the download starts, and start it on an
   action — never as a side effect of navigating to the page.
3. **One state machine, one vocabulary.** Every task hook exposes the same `status`,
   `progress`, `backend`, `error`, `run`, `running`, `result`. A reviewer who has read
   one task hook has read all of them.
4. **The worker is the boundary.** Stages LOAD and RUN happen in a Web Worker; the page
   only ever sees messages. Nothing model-shaped blocks the UI thread.
5. **Degrade, don't disappear.** No WebGPU → WASM. No mic → file upload. Model failed to
   load → an error in the LOAD slot with a retry, not a blank page.

---

## 2. The state machine

Two orthogonal machines run per page. Keep them orthogonal — do not collapse them into
one enum.

### Machine A — model lifecycle (`status`, one per worker)

```
   ┌──────┐  load()   ┌─────────┐  ready   ┌───────┐
   │ idle │──────────▶│ loading │─────────▶│ ready │
   └──────┘◀──────────└────┬────┘          └───┬───┘
      ▲       cancel()     │ progress ↺        │
      │                    │                   │ run()
      │                    │ error (no id)     │   ↺
      │                    ▼                   │
      │               ┌───────┐  retry()       │
      └───────────────│ error │◀───────────────┘
        model change  └───────┘   load failure only
```

- `idle` is the **default**. The worker is not created until `load()`.
- `cancel()` abandons a download in flight and returns to `idle` — a load can take
  minutes, and a user who changed their mind should not have to reload the page.
  Cancelling is not failing: no error is shown, and the partial download stays in the
  browser cache, so resuming later picks up where it left off.
- `retry(overrides?)` merges `overrides` into the `load` message. That is how the
  "Retry on CPU" action after a GPU failure pins `{ backend: "wasm" }` — §6 still holds,
  the backend is resolved once per worker, the user has simply chosen it instead of the
  probe.
- `progress` events are a self-loop on `loading` — they never change `status`.
- Changing the selected model tears the worker down and returns to `idle`.
- `error` here means **the model could not be loaded**. A failed *inference* does not
  leave `ready`.

### Machine B — inference (per request, id-correlated)

```
   run(input) ──▶ id = ++nextId
                  pending.set(id, {resolve, reject})
                  inflight += 1
                       │
                       ├── {result, id} ──▶ resolve · inflight -= 1
                       ├── {error, id}  ──▶ reject  · inflight -= 1 · status stays "ready"
                       └── teardown     ──▶ reject all pending · clear
```

`running` is `inflight > 0`. Track a **count, not a boolean** — a boolean is wrong the
moment two requests overlap (the first to return clears the flag while the second is
still in flight).

The `id` on an error message is the discriminator: `id != null` is a request failure
(Machine B), `id == null` is a load failure (Machine A).

---

## 3. The hook contract

Every task hook returns this shape. Extra task-specific fields are additive; nothing
here is optional or renamed per task.

```ts
export interface ModelTask<TInput, TOutput, TOpts = void> {
  // --- Machine A: load ---
  status: "idle" | "loading" | "ready" | "error";
  idle: boolean;
  loading: boolean;
  ready: boolean;
  /** The last raw worker progress event. Null outside `loading`. */
  progress: ModelProgress | null;
  /** Aggregate, monotonic progress across every file. Null outside `loading`. */
  loadProgress: LoadProgress | null;
  /** How long the load that produced `ready` took, in ms. */
  loadedInMs: number | null;
  /** Resolved execution backend once `ready` — "webgpu" | "wasm". */
  backend: Backend | null;
  /** Start the download. No-op unless `idle`. */
  load: () => void;
  /** Re-attempt a failed load, same model. No-op unless `error`. */
  retry: (overrides?: Record<string, unknown>) => void;
  /** Abandon a load in flight, returning to `idle`. No-op unless `loading`. */
  cancel: () => void;

  // --- Machine B: run ---
  run: (input: TInput, opts?: TOpts) => Promise<TOutput>;
  /** True while ANY request is in flight (inflight count > 0). */
  running: boolean;
  /** Latest successful output, for pages that display one result at a time. */
  result: TOutput | null;

  /** Load error (status === "error") or the most recent run error. */
  error: string | null;
}
```

The identical worker plumbing — creation, teardown, the pending map, the
`progress`/`ready`/`result`/`error` switch — lives once in `useModelWorker`. A task hook
is a thin typed wrapper around it, plus whatever is genuinely task-specific (ASR's
capture loop, TTS's voice option).

**`useEnhance` is the reference implementation** — it returns this contract verbatim
and nothing else. The older hooks predate it and still expose a task-named alias for
`run` (`useTts`'s `synthesize`, `useAsr`'s `transcribe`, `useAudioClassifier`'s
`classify`); those are being migrated route by route. Do not add a new alias — a
reviewer who has read one hook should have read them all, and a renamed `run` is
exactly what breaks that.

---

## 4. The four slots

The page shell composes them; a route supplies content. The stages are split by
**how often they are used**, not stacked evenly: SELECT and LOAD are done once per
session and collapse into a compact setup rail, while RUN and OUTPUT — the pair the
user touches on every iteration — sit side by side, so a result never lands below
the fold and there is no scroll between the input being edited and the output being
judged.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│  header      icon · title · one-line what/where                        [aside] │
├──────────────────────┬────────────────────────────────────────────────────────┤
│  SETUP RAIL (~20rem) │  WORKBENCH                                             │
│                      │  ┌──────────────────────┬───────────────────────────┐  │
│  1. MODEL            │  │ 3. INPUT             │ 4. OUTPUT                 │  │
│     <ModelPicker>    │  │    <InputPanel>      │    <OutputPanel>          │  │
│     one row per      │  │                      │                           │  │
│     model · hint     │  │    the input surface │    empty → what a result  │  │
│     params · size    │  │                      │      will look like       │  │
│     · ⚠ large        │  │    ─ transport ─     │    running → skeleton     │  │
│                      │  │    every control     │    result → output + its  │  │
│  2. LOAD             │  │    gated on `ready`  │      actions              │  │
│     <ModelStatus>    │  │                      │    error → the message    │  │
│     idle → button    │  │  input errors here   │  run errors here          │  │
│     loading → bar    │  └──────────────────────┴───────────────────────────┘  │
│     warmup → "…"     │                                                        │
│     ready → chip     │  Each column scrolls its own overflow; the header and   │
│     error → retry    │  the rail never scroll away.                           │
└──────────────────────┴────────────────────────────────────────────────────────┘
```

### 4a. Breakpoints

| Width | Arrangement |
|---|---|
| `< md` (< 768) | One column, all four bands stacked in pipeline order. The page scrolls as one sheet — no height clamp, no inner scroll containers. |
| `md … xl` | Setup becomes a **horizontal** strip across the top (MODEL beside LOAD); INPUT and OUTPUT are side by side below it. |
| `≥ xl` (1280+) | The full arrangement above: rail, then the two workbench columns. |

**DOM order is pipeline order at every breakpoint.** The shell is a CSS grid that
places children by *area*; it never reorders source. That is what keeps Tab and a
screen reader walking Model → Load → Input → Output no matter how the bands are
arranged visually, and it is asserted directly (`ModelPage.test.tsx`).

Two mechanical rules make the columns behave: both workbench columns carry
`min-h-0 min-w-0` — without them a grid child refuses to shrink, the inner scroll
never engages and a wide result pushes the whole page sideways — and the height
clamp only applies from `md` up, so a short phone viewport is never trapped inside
a scroll container.

**Slot rules**

- SELECT is disabled while `loading` or `running` — switching mid-flight would discard
  work the user is waiting on.
- LOAD is the only slot that may show a progress bar. Inference progress belongs in
  OUTPUT.
- RUN controls are `disabled={!ready || running}`. There is no "queue it up" affordance.
  The transport row is `sticky bottom-0` inside the input column, so a tall input
  surface can scroll without carrying Run out of reach — sticky, not bottom-pinned:
  pinning leaves a chasm on every task whose input is short.
- OUTPUT always renders, and fills its column. An empty state that describes the coming
  result is worth more than a collapsed section, and it stops the page from jumping
  when the result lands.
- Errors render **in the slot that produced them**. A load error goes in LOAD, a decode
  error in RUN, an inference error in OUTPUT.
- The header's `aside` is for one-line answers that don't deserve a band — a backend
  chip, a device line. It is not a fifth stage.

---

## 5. Size before load

The guardrail that motivates `idle`. Before any bytes move, the user sees the estimate:

```
  Whisper base · 74M params · ~140 MB
  ⚠ Large model — slow first load and high memory use.
    Weights are cached after the first download.
```

- A model whose weights are **already in the browser cache** says so — a `Cached` badge
  on its row, "already downloaded" in place of the size, and "Load model (cached)" on the
  button. Nothing else in this slot answers the question the user is actually asking,
  which is whether this click costs 200 MB or nothing. The probe is `model/cache.ts`; it
  reads Cache Storage and resolves to "not cached" on any failure, because the cost of
  that error direction is one extra click, while the other spends bandwidth unasked.
- The estimate is quoted **once, by `ModelPicker`** — it sits directly above LOAD in the
  setup rail, so the guardrail is satisfied where the choice is made. `ModelStatus` does
  not repeat the number; in a 20rem rail that just said it twice.
- Estimate from measured `bytes` per backend when available; fall back to params × dtype.
- The estimate is **backend-dependent** — a WASM q8 encoder with an fp32 decoder is not
  the same download as fp16 on WebGPU. Resolve the backend before quoting a number.
- Past `LARGE_MODEL_BYTES`, the warning is mandatory.

---

### 5a. Progress while it loads

The LOAD slot reports the **aggregate**, never a raw per-file event. Transformers.js
reports progress per file (4–8 of them), so rendering its payload directly makes the bar
restart at 0 for each one — it visibly runs forwards, snaps back, and then sits at 100%
through a warm-up that has not started. `model/progress.ts` folds the events into one
picture, and its rules are the interesting part:

- percent is **by bytes, across all files with a known size**, and **monotonic** — a
  newly-announced file enlarges the denominator and must not drop the bar;
- a file with no announced size is excluded rather than guessed at; while no size is
  known the bar is **indeterminate** (and `aria-valuenow` is omitted, so assistive tech
  says "busy" instead of announcing a number we do not have);
- **warm-up is its own phase**, indeterminate, labelled as the first inference — not the
  tail of the download;
- **no ETA.** A rate extrapolated from the first seconds of a multi-file download is
  wrong by a factor of several. Bytes downloaded and elapsed time are facts; show those.

### 5b. Surviving a refresh

A Worker and its GPU/WASM session cannot outlive a page load — nothing persists an
in-memory model, and no amount of UI should imply otherwise. What persists is:

- **the selection** (`store/models.ts`, `localStorage`), validated against the catalogue
  on read — a stored id we no longer ship falls back to the default and is dropped;
- **the intent to load it**, recorded when the user presses Load and cleared on cancel or
  a model change;
- **the weights**, in the browser's cache, which is what makes a resume free.

The resume rule, in `model/useModelSelection.ts`: auto-load on mount **only when the
stored intent and a cache hit agree**. An uncached model stays `idle` and asks, however
recently it was used. While a resume runs, the slot says "Restoring from cache…" — the
load is real and is described as one.

## 6. Backend selection

Resolved once inside the worker, before `ready`. Never throws, never re-negotiates.

```
  navigator.gpu? ──yes──▶ requestAdapter() ──ok──▶ webgpu  (fp16)
        │                        │
        no                  null/throws
        └────────────────────────┴──────────────▶ wasm    (q8)
```

Once `ready` reports a backend it is fixed for that worker's life. A GPU failure after
load surfaces as a run error, not a silent fallback — silently switching would make the
timings the user is reading meaningless.

---

## 7. Where the pattern bends

Not every page downloads weights, and that's fine — the stages still hold:

| Page | SELECT | LOAD | RUN | OUTPUT |
|---|---|---|---|---|
| TTS / ASR / Audio Classification | model catalogue | weight download in worker | text / mic / file | audio, transcript, labels |
| Audio to Audio | model catalogue | ONNX graph + constants file | mic / file (48 kHz) | before/after waveforms, A-B play, WAV |
| Tensor Arithmetic | the operation | WGSL pipeline compile (fast, auto) | operand matrices | heatmap + numeric grid |
| Linear Training | architecture + hyperparams | dataset fetch + kernel compile | train loop | live weights + loss curve |
| `/tasks/$slug` placeholder | — | — | — | "not available yet" |

Where LOAD is fast and free (a shader compile), it may auto-run — pass `autoLoad`. The
slot still renders, so the page keeps the same four-band rhythm as its neighbours; use
[`DeviceStatus`](../../frontend/src/components/model/DeviceStatus.tsx) there, which answers
the question those pages actually raise — is there a GPU, or nothing to compute on. The
placeholder route uses the same shell with empty slots, so an unimplemented task reads
as *the same kind of page*, not a different app.

**Linear Training is the documented exception.** It does not render `ModelPage` at all,
so it opts out of the setup-rail/workbench arrangement along with everything else. It
keeps its own layout: a full-bleed
pan/zoom canvas whose background *is* the model, with a floating HUD. Forcing it into
stacked bands would destroy the thing the visualization standard names as its reference
implementation, and its "run" is a long-lived loop with start/stop rather than a
request/response. It still honours the *stages* — the Dataset dialog is LOAD, Tune + Start
is RUN, the live stats and charts are OUTPUT — and it shares `DeviceStatus` for the
can-this-run answer. A page may earn this exemption; it may not quietly invent a fifth
stage or skip OUTPUT.

---

## 8. Test hooks

The slots expose a small, stable set of `data-testid`s. They are a **contract**:
Vitest route tests and Playwright specs both key off them, so renaming one breaks
tests in two suites at once. Add to this table rather than inventing an ad-hoc id.

| Test id | Where | Means |
|---|---|---|
| `slot-1` … `slot-4` | `ModelPage` | The four bands, in pipeline order. Always exactly four. |
| `model-size-note` | `ModelPicker` | The selected model's hint, params and per-backend download. |
| `model-size-warning` | `ModelPicker` | The large-model guardrail. Absent below `LARGE_MODEL_BYTES`. |
| `model-ready` | `ModelStatus` | The model loaded; carries the resolved backend and the load time. |
| `load-progress` | `ModelStatus` | A load in flight. `data-phase` is `connecting` \| `downloading` \| `warmup`. |
| `load-cancel` | `ModelStatus` | Abandon the download. Present only while loading. |
| `model-cached-<id>` | `ModelPicker` | That model's weights are already downloaded. |
| `model-evict` | `ModelPicker` | Clear the selected model's cached weights. |
| `device-ready` | `DeviceStatus` | A GPU device was acquired (compile-only routes). |
| `output-panel` | `OutputPanel` | The OUTPUT card. Present regardless of whether there is a result. |
| `output-empty` | `OutputPanel` | No result and nothing running — the "what you'll get" state. |
| `output-running` | `OutputPanel` | A first run is in flight. Absent when a previous result is still shown. |
| `error-note` | `ErrorNote` | Any error. Also `role="alert"` — prefer the role in assertions. |

The testids are unchanged by the horizontal arrangement — `slot-N` is bound to the
step number, not to a position in the layout.

**Layout itself is a Playwright assertion, never a Vitest one.** jsdom/happy-dom has no
geometry, so a route test asserts presence and order only; the arrangement half of §4 is
checked in `e2e/specs/model-page.spec.ts`, which asserts at 1440×900 that INPUT and
OUTPUT are horizontally disjoint and vertically aligned, that the OUTPUT panel starts
above the fold, and that the document never scrolls horizontally — plus, at 375×812,
that the four bands stack in increasing `y`.

Prefer role and text queries where they work; these exist for the states that have
no natural accessible name (a band, an empty panel). Note that a band is a labelled
`region`, so its heading text participates in accessible-name lookups —
`getByLabelText(/text/i)` will match both a band named "Text" and a field inside it.
That ambiguity is one reason §4's band labels stay generic.

**What every task page's tests should cover**, beyond the task's own behaviour:

- Nothing downloads on mount — assert the hook was called with `autoLoad: false`
  *and* that `load` was not called.
- `load()` fires from the LOAD slot, `retry()` from its error state, `cancel()` from the
  progress row.
- The refresh pair: a stored selection is restored; a stored intent **plus** a cache hit
  resumes the load, and a stored intent **without** one does not.
- All four slots render, with `output-empty` visible, before any run.
- Run controls are disabled until `ready`.
- A load error renders in LOAD, a run error in OUTPUT, and the page stays usable.

## 9. Checklist — adding a task page

- [ ] Model catalogue entry: `id`, `label`, `hint`, `params`, measured `bytes` per backend
- [ ] Worker built on the shared protocol (`load` / `run` messages, id-correlated)
- [ ] Hook wraps `useModelWorker`; returns the §3 contract verbatim
- [ ] Page uses `ModelPage` + the four slots — no bespoke layout, no grid of its own
- [ ] The input surface fills its column (`flex-1`) if it is the column's main element
- [ ] `idle` default: nothing downloads until the user asks
- [ ] Selection persisted through `useModelSelection`, with the route's own `routeKey`
- [ ] Size estimate + large-model warning shown before load
- [ ] Every RUN control gated on `ready`
- [ ] OUTPUT has an empty state, a running state, and an error state
- [ ] Errors land in the slot that produced them
- [ ] Unit tests for the hook's state machine; a route test covering §8's list
- [ ] Sidebar taxonomy entry mapped in `REAL_ROUTES`

---

## Related

- [`model-visualization.md`](model-visualization.md) — how a model and its internals are drawn
- [`../explanations/webgpu-inference.md`](../explanations/webgpu-inference.md) — how inference runs
- [`../guides/adding-a-model.md`](../guides/adding-a-model.md) — kernel + registry entry
- [`../plans/completed/model-page-restructure.md`](../plans/completed/model-page-restructure.md) — the migration to this pattern
- [`../plans/completed/model-page-horizontal-layout.md`](../plans/completed/model-page-horizontal-layout.md) — the move from one column to rail + workbench
- [`../plans/completed/model-load-persistence-and-progress.md`](../plans/completed/model-load-persistence-and-progress.md) — aggregate progress, cancel, and the refresh story

## Reference implementations

| Piece | File |
|---|---|
| State machines | [`frontend/src/model/useModelWorker.ts`](../../frontend/src/model/useModelWorker.ts) |
| Contract types | [`frontend/src/model/types.ts`](../../frontend/src/model/types.ts) |
| Shell + slots | [`frontend/src/components/model/`](../../frontend/src/components/model/) |
| A weight-downloading page | [`routes/text-to-speech.tsx`](../../frontend/src/routes/text-to-speech.tsx) |
| A compile-only page | [`routes/tensor.tsx`](../../frontend/src/routes/tensor.tsx) |
| The empty case | [`routes/tasks.$slug.tsx`](../../frontend/src/routes/tasks.$slug.tsx) |
| The contract, asserted | [`frontend/e2e/specs/model-page.spec.ts`](../../frontend/e2e/specs/model-page.spec.ts) |
