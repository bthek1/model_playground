# Model Load: Persistence, Progress & Feedback

**Status:** Complete (2026-09-04)
**Created:** 2026-09-04
**Scope:** `frontend/src/model/`, `frontend/src/components/model/`, `frontend/src/store/`,
the audio engines (`frontend/src/audio/`), and the task routes.
**Standard touched:** [`docs/standards/model-page-pattern.md`](../../standards/model-page-pattern.md)
(§2 Machine A, §4 slot rules, §5 size-before-load).

---

## Problem

Three complaints about the LOAD stage, all landing in the same two files
(`model/useModelWorker.ts`, `components/model/ModelStatus.tsx`):

1. **A load does not survive a page refresh.** The worker dies with the page, the hook
   resets to `idle`, and the route's `useState(DEFAULT_*_MODEL)` throws away the model
   the user picked. The *bytes* are already cached by the browser (Transformers.js keeps
   them in the `transformers-cache` Cache Storage; the ONNX/DFN3 files sit in the HTTP
   cache), so the second load is fast and free — but the UI presents it as if nothing had
   ever happened, and asks for the same consent click again.
2. **The progress bar is weak.** `ModelStatus` renders a bar, but it is fed a raw
   Transformers.js `progress_callback` payload: per-**file** percent that restarts at 0
   for each of the 4–8 files a model pulls, no aggregate, no byte counts, and a hardcoded
   `8%` sliver before the first event arrives. On a 220 MB ASR download the bar visibly
   goes forwards, snaps back, and stalls in warm-up with no motion at all.
3. **Feedback is thin.** No elapsed time, no way to cancel a download in flight, no
   indication that a model is already cached (so "Load model" reads equally expensive for
   a 3 MB warm model and a 1 GB cold one), and load errors surface the raw exception
   string with no hint about the common causes (offline, 401 on a bad model id, OOM).

### What "survive a refresh" can and cannot mean

A `Worker` and its GPU/WASM session cannot outlive a page load — nothing persists an
in-memory ONNX session. What *is* achievable, and what this plan delivers:

- the **selection** (model id, and per-route options like TTS voice) is restored;
- the fact that the model's weights are **already in Cache Storage** is detected, shown,
  and used to **auto-resume** the load without re-asking for consent, because the
  consent in §5 of the standard is consent to *spend bandwidth*, and a cache hit spends
  none;
- the restore is visibly a *re-load from cache* ("Restoring from cache…"), not a lie
  about a session that survived.

A cold (uncached) model still returns to `idle` and still asks. That is the guardrail
working, not a bug.

---

## What shipped

All five phases, with two deliberate deviations from the draft, both recorded below:

- **The stage rail in Phase 5 was dropped.** `ModelPage` already numbers each band ①–④
  with its label; a second numbered strip in the same rail restated it in fewer words and
  earned no space.
- **The `autoLoad` signature stayed boolean.** The draft floated
  `autoLoad: boolean | "when-cached"`; the resume decision is resolved in
  `useModelSelection` and handed down as a plain boolean, which keeps `useModelWorker`
  ignorant of caches. It starts `false` while the probe is outstanding — `null` and
  `false` both mean "don't load", so the flip costs no teardown.

One thing found during implementation and worth keeping in mind: zustand's `persist`
captures its storage at module-init time, which in the test environment is Node's stub
`localStorage`. `store/models.ts` therefore passes a `safeStorage` adapter that resolves
`globalThis.localStorage` per call and swallows the private-mode throw — the same
adapter that makes preferences non-fatal in a browser that blocks site data.

## Phase 1 — Aggregate, honest progress

**New:** `frontend/src/model/progress.ts` — a pure module, unit-tested, no React.

```ts
export interface LoadProgress {
  phase: "connecting" | "downloading" | "warmup";
  /** 0–100 across ALL files, by bytes. Null while indeterminate. */
  percent: number | null;
  loaded: number;          // bytes
  total: number;           // bytes, 0 while unknown
  files: { done: number; count: number };
  /** The file currently moving, for the detail line. */
  current?: string;
}

export function reduceProgress(prev: FileTable, e: ModelProgress): FileTable;
export function summarize(t: FileTable, elapsedMs: number): LoadProgress;
```

- Keep a `Map<file, {loaded, total, status}>` rather than the last event. Aggregate
  percent is `Σloaded / Σtotal` over files whose `total` is known; a file with an unknown
  total contributes to neither, and `percent` stays `null` (indeterminate bar) until at
  least one total is known.
- **Never go backwards.** Clamp the reported percent to a running maximum — a new file
  appearing must not drop the bar.
- `warmup` is its own phase, not 100 % of the download: the bar goes indeterminate
  (striped/animated) with the label "Warming up (first inference)…", which is what it
  actually is.
- Do not compute an ETA from a partial download; a wrong ETA is worse than none. Show
  `18.4 / 221 MB` and elapsed seconds instead.

**Changed:** `useModelWorker` keeps the file table in a ref, exposes the existing raw
`progress` (unchanged, for compatibility) **plus** `loadProgress: LoadProgress | null`,
and records `loadStartedAt` / `loadedInMs`.

**Changed:** `ModelStatus` renders from `loadProgress`:

```
  ⟳ Downloading · encoder_model_fp16.onnx        3 of 6 files
  ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░  46%      101 / 221 MB · 12s
                                          [ Cancel ]
```

with `role="progressbar"` + `aria-valuenow/min/max` (indeterminate: omit `aria-valuenow`),
and the whole block in an `aria-live="polite"` region that announces phase changes only —
not every percent tick.

## Phase 2 — Cancel and better load errors

- **`cancel()` on the hook.** Terminates the worker, rejects the pending table, returns
  Machine A to `idle`. Machine A gains one transition — `loading --cancel()--> idle` — and
  the standard's §2 diagram is updated to match. The button lives in the LOAD slot next to
  the bar; the partial download stays in the browser cache, so a cancel is cheap to undo.
- **`classifyLoadError(err)`** in `model/errors.ts` maps the raw string to a cause and a
  next step: offline / `404|401` on the model repo (bad id — this has bitten us before,
  see the `onnx-community/*` note in CLAUDE.md) / out-of-memory / WebGPU device lost /
  everything else (raw message, verbatim, never swallowed). `ErrorNote` shows the friendly
  line with the raw message in a `<details>`.
- A WebGPU load failure offers **"Retry on CPU (WASM)"** alongside Retry: the worker
  accepts a `backend` override in the `load` message and skips `pickBackend()` when given
  one. §6 of the standard stays true — the backend is still resolved once per worker, it
  is just the user picking it after a failure.

## Phase 3 — Cache awareness

**New:** `frontend/src/model/cache.ts`

```ts
export async function isModelCached(modelId: string): Promise<boolean>;
export async function cachedModels(): Promise<Set<string>>;
export async function evictModel(modelId: string): Promise<void>;
```

Implemented over the Cache Storage API: open `transformers-cache` (the name
`@huggingface/transformers` uses), list `keys()`, and match request URLs containing
`/${modelId}/`. Every function is defensive — `caches` is absent on a non-secure origin
and throws in some privacy modes — and resolves to "not cached" rather than rejecting.
The DFN3/ORT route caches through the HTTP cache instead, so it reports `false` and
simply loses the badge; no route breaks.

- **`ModelPicker`** shows a `Cached` chip on rows whose weights are present, and the size
  line reads `221 MB · already downloaded` for those. This directly answers "is this click
  going to cost me 200 MB?".
- **`ModelStatus`** in `idle` with a cache hit reads **"Load model (cached)"** and drops
  the "Downloaded once…" line.
- A small **"Clear cached weights"** affordance (per model, in the picker row) calling
  `evictModel` — needed to test cold loads and to reclaim space.

## Phase 4 — Restoring across a refresh

**New:** `frontend/src/store/models.ts` — a Zustand store persisted to `localStorage`
(UI/preference state only; no server data, per CLAUDE.md).

```ts
interface ModelPrefs {
  /** route key → last selected model id, e.g. "asr" → "onnx-community/whisper-base" */
  selected: Record<string, string>;
  /** route key → the user asked for this to be loaded */
  autoResume: Record<string, boolean>;
}
```

- **New hook `useModelSelection(routeKey, models, fallback)`** replaces each route's
  `useState(DEFAULT_*_MODEL)`. It rehydrates the stored id, **validates it against the
  catalogue** (a stored id that is no longer shipped falls back to the default — silently,
  and the stale entry is dropped), and writes back on change.
- `load()` sets `autoResume[routeKey] = true`; `cancel()` and a model change clear it.
- **Resume rule, in `ModelPage`'s owner (the route), not inside `useModelWorker`:** on
  mount, auto-load only when `autoResume` **and** `isModelCached(id)` are both true. A
  cached model resumes; an uncached one shows a distinct idle state —
  *"You were using this model. Loading it again will re-download ~221 MB."* with the same
  explicit button. This keeps §1.2 of the standard intact: nothing uncached ever downloads
  without a click.
- While resuming, `ModelStatus` says **"Restoring from cache…"**, so the state is never
  presented as a session that survived.
- `autoLoad` in `useModelWorker` becomes `autoLoad?: boolean | "when-cached"`, or — the
  simpler option, preferred — stays boolean and the route passes the resolved decision.
  Decide in Phase 4 implementation; the route-side resolution keeps the hook dumb and is
  the default choice.

## Phase 5 — UI polish across the slots

- **Ready chip** gains the load time and the source: `Model ready · WEBGPU · 2.3s from cache`.
- **Setup rail** gets a compact stage rail: `① Model ✓ · ② Load ⟳ · ③ Run · ④ Output`, so
  the four stages read as a pipeline at rail width. Purely presentational; DOM order still
  is pipeline order (`ModelPage.test.tsx` asserts this — do not reorder source).
- **Disabled controls say why.** Run controls carry a `title`/`aria-description`
  ("Load the model first") instead of being inertly grey.
- **`OutputPanel` running state** shows an elapsed timer after 2s so a slow first inference
  doesn't look hung.
- The load bar is **`ModelStatus`'s alone** — §4's rule stands; nothing in this plan puts a
  second bar in RUN or OUTPUT.

---

## Files

| File | Change |
|---|---|
| `frontend/src/model/progress.ts` | **new** — file table + aggregate summary (pure) |
| `frontend/src/model/errors.ts` | **new** — `classifyLoadError` |
| `frontend/src/model/cache.ts` | **new** — Cache Storage probe / evict |
| `frontend/src/store/models.ts` | **new** — persisted selection + resume intent |
| `frontend/src/model/useModelSelection.ts` | **new** — validated, persisted selection |
| `frontend/src/model/useModelWorker.ts` | aggregate progress, `cancel()`, timing |
| `frontend/src/model/types.ts` | `LoadProgress`, `cancel` in `ModelTask` |
| `frontend/src/components/model/ModelStatus.tsx` | new bar, phases, cancel, cached copy |
| `frontend/src/components/model/ModelPicker.tsx` | cached chip, evict, size copy |
| `frontend/src/components/model/ErrorNote.tsx` | friendly cause + raw `<details>` |
| `frontend/src/audio/*Engine.ts` | pass `backend` override; keep `warmup` event |
| `frontend/src/routes/{asr,text-to-speech,audio-classification,audio-to-audio,text-to-audio}.tsx` | use `useModelSelection`, resolve resume |
| `docs/standards/model-page-pattern.md` | §2 cancel transition, §5 cached state, §8 testids |
| `.github/copilot-instructions.md` / `CLAUDE.md` | keep in sync if the contract changes |

New testids for §8's contract: `load-progress`, `load-cancel`, `model-cached`,
`model-restoring`.

---

## Testing

**Unit (Vitest) — `just fe-test`**

- `progress.test.ts`: multi-file aggregation; percent never decreases when a new file
  appears; unknown totals keep it indeterminate; `warmup` maps to the warmup phase;
  a file reaching 100 % increments `files.done` exactly once.
- `errors.test.ts`: each cause maps correctly; an unmatched error keeps its raw message.
- `cache.test.ts`: `caches` undefined → `false`, never throws; `caches.open` rejecting →
  `false`; URL matching only inside the model's own path segment.
- `useModelWorker.test.ts`: `cancel()` from `loading` returns to `idle`, terminates the
  worker and rejects pending; `cancel()` is a no-op in `ready`/`idle`; `loadedInMs` is set
  on `ready`.
- `useModelSelection.test.ts`: rehydrates a stored id; a stored id absent from the
  catalogue falls back to the default and is dropped from the store.
- `ModelStatus.test.tsx`: renders each phase; `role="progressbar"` values; indeterminate
  omits `aria-valuenow`; cancel button only during `loading`; "cached" copy in `idle`.
- `ModelPicker.test.tsx`: cached chip only for cached ids.
- Route tests keep asserting the existing invariant — **nothing downloads on mount**:
  hook called with the resolved auto-load `false` and `load` not called, for an uncached
  model. Add the mirror case: cached + `autoResume` → `load` **is** called once.

**E2E (Playwright) — `just fe-e2e`**

- `model-page.spec.ts`: with a mocked cache probe reporting "not cached", a reload leaves
  the page in `idle` with the *previously selected* model still selected — the selection
  survives, the download does not start (assert zero Hub requests).
- Cached path: seed a fake `transformers-cache` entry in the page context, reload, assert
  `model-restoring` appears and `model-ready` follows.
- Cancel: start a load against a throttled/stalled route, click Cancel, assert the page is
  back to `idle` and the worker request is aborted.

**`@slow` — `just fe-e2e-slow`**

- One real model (`whisper-tiny.en`): load, reload the page, assert it resumes from cache
  without a network download and reaches `model-ready` markedly faster than the cold load.

**Manual**

- DevTools → Application → Cache Storage shows `transformers-cache` growing; "Clear cached
  weights" removes exactly that model's entries.
- Offline (DevTools → Network → Offline): a cached model still loads; an uncached one shows
  the offline-specific error.

---

## Risks / open questions

- **Cache-name coupling.** `transformers-cache` is `@huggingface/transformers`' internal
  cache name; a library upgrade could change it. Contain it in `cache.ts` behind one
  constant, and treat a miss as "not cached" — the failure mode is a lost badge and one
  extra consent click, never a broken page.
- **Auto-resume vs. consent.** Resuming without a click is only defensible on a cache hit.
  If the probe is ever wrong (says cached, isn't), the user pays a download they did not
  ask for. Mitigate by re-checking nothing: the probe is the only gate, so keep it strict —
  match the model path, not a substring.
- **Storage pressure.** The browser may evict Cache Storage under pressure; a stored
  `autoResume` then meets an empty cache and correctly falls back to the idle prompt.
- Phases 1–2 are independently shippable and carry most of the perceived improvement;
  3–4 are the refresh story; 5 is polish. Ship in order.
