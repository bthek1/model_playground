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
   └──────┘           └────┬────┘          └───┬───┘
      ▲                    │ progress ↺        │
      │                    │                   │ run()
      │                    │ error (no id)     │   ↺
      │                    ▼                   │
      │               ┌───────┐  retry()       │
      └───────────────│ error │◀───────────────┘
        model change  └───────┘   load failure only
```

- `idle` is the **default**. The worker is not created until `load()`.
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
  /** Weight-download / warm-up progress. Null outside `loading`. */
  progress: ModelProgress | null;
  /** Resolved execution backend once `ready` — "webgpu" | "wasm". */
  backend: Backend | null;
  /** Start the download. No-op unless `idle`. */
  load: () => void;

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

---

## 4. The four slots

The page shell composes them; a route supplies content.

```
┌─────────────────────────────────────────────────────────┐
│  <ModelPageHeader>   icon · title · one-line what/where │
├─────────────────────────────────────────────────────────┤
│  1. SELECT   <ModelPicker>                              │
│              buttons · hint · params · download size    │
│              · large-model warning                      │
├─────────────────────────────────────────────────────────┤
│  2. LOAD     <ModelStatus>                              │
│              idle    → "Load model (142 MB)" button     │
│              loading → file · % · progress bar          │
│              warmup  → "Warming up…"                    │
│              ready   → "Ready · running on WEBGPU"      │
│              error   → message + Retry                  │
├─────────────────────────────────────────────────────────┤
│  3. RUN      <InputPanel>  (task-specific)              │
│              the input surface + transport controls;    │
│              every control disabled unless `ready`      │
├─────────────────────────────────────────────────────────┤
│  4. OUTPUT   <OutputPanel>                              │
│              empty → what a result will look like       │
│              running → skeleton, never a layout jump    │
│              result → the output + its actions          │
│              error → the message, page still usable     │
└─────────────────────────────────────────────────────────┘
```

**Slot rules**

- SELECT is disabled while `loading` or `running` — switching mid-flight would discard
  work the user is waiting on.
- LOAD is the only slot that may show a progress bar. Inference progress belongs in
  OUTPUT.
- RUN controls are `disabled={!ready || running}`. There is no "queue it up" affordance.
- OUTPUT always renders. An empty state that describes the coming result is worth more
  than a collapsed section, and it stops the page from jumping when the result lands.
- Errors render **in the slot that produced them**. A load error goes in LOAD, a decode
  error in RUN, an inference error in OUTPUT.

---

## 5. Size before load

The guardrail that motivates `idle`. Before any bytes move, the user sees the estimate:

```
  Whisper base · 74M params · ~140 MB
  ⚠ Large model — slow first load and high memory use.
    Weights are cached after the first download.
```

- Estimate from measured `bytes` per backend when available; fall back to params × dtype.
- The estimate is **backend-dependent** — a WASM q8 encoder with an fp32 decoder is not
  the same download as fp16 on WebGPU. Resolve the backend before quoting a number.
- Past `LARGE_MODEL_BYTES`, the warning is mandatory.

---

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
| Tensor Arithmetic | the operation | WGSL pipeline compile (fast, auto) | operand matrices | heatmap + numeric grid |
| Linear Training | architecture + hyperparams | dataset fetch + kernel compile | train loop | live weights + loss curve |
| `/tasks/$slug` placeholder | — | — | — | "not available yet" |

Where LOAD is fast and free (a shader compile), it may auto-run — pass `autoLoad`. The
slot still renders, so the page keeps the same four-band rhythm as its neighbours. The
placeholder route uses the same shell with empty slots, so an unimplemented task reads
as *the same kind of page*, not a different app.

---

## 8. Checklist — adding a task page

- [ ] Model catalogue entry: `id`, `label`, `hint`, `params`, measured `bytes` per backend
- [ ] Worker built on the shared protocol (`load` / `run` messages, id-correlated)
- [ ] Hook wraps `useModelWorker`; returns the §3 contract verbatim
- [ ] Page uses `ModelPage` + the four slots — no bespoke layout
- [ ] `idle` default: nothing downloads until the user asks
- [ ] Size estimate + large-model warning shown before load
- [ ] Every RUN control gated on `ready`
- [ ] OUTPUT has an empty state, a running state, and an error state
- [ ] Errors land in the slot that produced them
- [ ] Unit tests for the hook's state machine; a route test for the four slots
- [ ] Sidebar taxonomy entry mapped in `REAL_ROUTES`

---

## Related

- [`model-visualization.md`](model-visualization.md) — how a model and its internals are drawn
- [`../explanations/webgpu-inference.md`](../explanations/webgpu-inference.md) — how inference runs
- [`../guides/adding-a-model.md`](../guides/adding-a-model.md) — kernel + registry entry
- [`../plans/in-progress/model-page-restructure.md`](../plans/in-progress/model-page-restructure.md) — the migration to this pattern
