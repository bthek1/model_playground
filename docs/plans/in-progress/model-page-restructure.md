# Plan: Model Page Restructure — Select → Load → Run → Output

**Status:** In Progress — Phase 1 complete (2026-09-02). Phases 2-6 pending.
**Date:** 2026-09-02

---

## Goal

Every task page in the playground is the same pipeline — *pick a model, load it, run it
on an input, show the output* — but each one currently reimplements it. This plan
extracts that pipeline into a shared state machine (`useModelWorker`) and a shared page
shell (`ModelPage` + four slots), migrates the existing routes onto it, and fixes three
defects the duplication has been hiding. The outcome: adding a new task page becomes
"write a catalogue entry, a worker adapter, and an input surface" — roughly 80 lines
instead of 200–400 — and every page behaves identically under load, error, and retry.

The target contract is specified in [`docs/standards/model-page-pattern.md`](../../standards/model-page-pattern.md).

## Background

Three hooks — [`useTts`](../../../frontend/src/hooks/useTts.ts),
[`useAsr`](../../../frontend/src/hooks/useAsr.ts),
[`usePipeline`](../../../frontend/src/hooks/usePipeline.ts) — are near-identical: same
worker lifecycle, same `Map<id, {resolve, reject}>` pending table, same
`progress`/`ready`/`result`/`error` switch, same `"loading" | "ready" | "error"` enum.
`useAudioClassifier` wraps `usePipeline`; `useLiveAsr` adds a capture loop on top of the
same core. The three audio routes then repeat the same JSX rhythm (picker → status →
controls → result card) with no shared shell, and [`asr.tsx`](../../../frontend/src/routes/asr.tsx)
has grown to 389 lines because of it.

Three defects fall out of the duplication:

1. **Eager loading.** Each hook's effect posts `{type:"load"}` on mount. The
   [`ModelPicker`](../../../frontend/src/components/audio/ModelPicker.tsx) size warning —
   "Large model — expect a slow first load" — is therefore shown *after* the download has
   already started. The guardrail does not guard anything.
2. **`running` is a boolean over an N-entry map.** With two overlapping requests, the
   first to return sets `running = false` while the second is still in flight, so the UI
   reports idle mid-inference. Each promise still resolves correctly; only the flag lies.
3. **No recovery from a load error.** `status: "error"` is terminal for the worker — the
   only escape is changing the selected model. There is no retry.

There is also a structural gap: `/tasks/$slug` renders a placeholder card that shares
nothing with a real task page, so unimplemented tasks read as a different app rather than
an empty version of the same page.

## Phases

### Phase 1 — Extract the state machine (no visible change) ✅

*Done 2026-09-02. The full suite passes with **378 tests** (up from 364 — the 14 new
`useModelWorker` tests); every pre-existing hook, route and E2E test passed **unmodified**,
which is this phase's proof of correctness. `tsc` clean, 0 lint errors, build green.*

- [x] Add `frontend/src/model/types.ts`: `ModelStatus` (`idle | loading | ready | error`),
      `ModelProgress`, `ModelTask<TInput, TOutput, TOpts>` — the §3 contract
- [x] Add `frontend/src/model/useModelWorker.ts` — worker creation/teardown keyed on
      `(factory, model, …deps)`, the id-correlated pending map, the response switch,
      `load()` / `retry()` actions
- [x] Track **`inflight` as a count**; derive `running = inflight > 0` (fixes defect 2)
- [x] Add `retry()`: `error → loading` without a model change (fixes defect 3)
- [x] Keep `autoLoad` defaulting to `true` in this phase so behaviour is unchanged
- [x] Rewrite `useTts`, `usePipeline`, `useAsr` as thin wrappers; their public shapes stay
      backward-compatible (`status`, `loading`, `ready`, `progress`, `backend`, `running`,
      `error`, plus task methods)
- [x] Unify the worker message protocols: fold `TtsRequest`/`TtsResponse` and
      `AsrRequest`/`AsrResponse` into the generic `ModelRequest`/`ModelResponse` in
      `model/types.ts`, leaving only the payload types task-specific

*Exit criteria: full unit + route + E2E suites pass unchanged. No JSX touched.* — **met.**

**Notes from the build:**

- `useModelWorker` keys the worker on a `key` string (`model`, or `` `${task}:${model}` ``)
  rather than a spread dependency array, so `loadMessage` can be a fresh object each render
  without respawning the worker. `createWorker`/`loadMessage` live in a ref for the same reason.
- The protocol unification is **structural, not a rename**: `ModelRequest<TLoad, TRun>` and
  `ModelResponse<TResult>` carry the envelope, and each task supplies its own payload
  (`{ audio, args }`, `{ input, args }`, `{ text, opts }`). All three response unions turned out
  to be identical apart from the result type, so `AsrResponse = ModelResponse<AsrResult>` and
  friends are exact aliases — no worker or engine code changed, and the engine tests never moved.
- `run()` forwards a transfer list **only when given one**, preserving TTS's transfer-free
  `postMessage` call — a blanket `[]` would have been observably different.
- **One test edit, and it is type-only:** three route tests build a mock hook result object, and
  `UseTtsResult` / `UseAudioClassifierResult` grew `load` and `retry`. The mocks gained
  `load: vi.fn(), retry: vi.fn()`. No assertion, behaviour, or expectation changed — the
  additive-interface consequence the phase rule is not aimed at.
- Defect 2's regression test was verified to actually catch the old behaviour: reverting the
  decrement to a boolean-style `setInflight(0)` fails "keeps running true until BOTH overlapping
  requests resolve", and nothing else.
- `status` gained `"idle"` in the three public hook types. Nothing reaches it yet — `autoLoad`
  stays `true` until Phase 3 — but the enum is now the standard's.

### Phase 2 — The page shell and shared slots

- [ ] New `frontend/src/components/model/` — promoted from `components/audio/`, made
      modality-agnostic:
  - [ ] `ModelPage.tsx` — header + the four labelled bands
  - [ ] `ModelPicker.tsx` (moved from `audio/`, unchanged API)
  - [ ] `ModelStatus.tsx` (moved from `audio/`) — gains `idle` and `error` renderings
  - [ ] `InputPanel.tsx` — titled band for the task's input surface + transport controls
  - [ ] `OutputPanel.tsx` — empty / running / result / error states in one place
- [ ] Re-export from `components/audio/` for one phase so the migration can land per-route
- [ ] `ModelStatus` accepts `onLoad` / `onRetry` and renders the "Load model (~142 MB)"
      button in `idle` and a Retry in `error`

### Phase 3 — Deferred load (the behaviour change)

- [ ] Flip `autoLoad` to `false` for weight-downloading tasks; `idle` becomes the default
- [ ] Surface the size estimate in the `idle` LOAD slot, not only in the picker (fixes
      defect 1)
- [ ] Keep `autoLoad: true` for compile-only tasks (tensor ops) where LOAD is fast and free
- [ ] Decide and document the "already cached" case — if the weights are in the Cache API,
      the idle prompt still appears but is labelled as cached rather than a fresh download
- [ ] Update route tests and E2E specs that assume a model starts loading on mount

*This is the one phase with a user-visible behaviour change. It ships alone.*

### Phase 4 — Migrate the audio routes

- [ ] `text-to-speech.tsx` → `ModelPage` + four slots (smallest, do it first as the pattern
      reference)
- [ ] `audio-classification.tsx` → same; fold `preparing`/`ioError` into the RUN slot's
      local state so `busy` stops being hand-rolled per route
- [ ] `asr.tsx` → same; extract the capture/playback/waveform block into
      `components/audio/AsrTransport.tsx` so the route is layout, not logic (targets ≤150
      lines, from 389)
- [ ] Delete the per-route duplicated header/description blocks in favour of `ModelPage`
      props

### Phase 5 — Non-audio routes and the placeholder

- [ ] `tensor.tsx` — adopt the shell with SELECT = operation, LOAD = auto (compile), RUN =
      operands, OUTPUT = existing heatmap + grid. Visualization internals unchanged
- [ ] `training.tsx` — adopt the shell; LOAD covers dataset fetch + kernel compile, OUTPUT
      keeps the live weights/loss views. This route legitimately diverges most; adopt the
      bands, do not force its interaction model
- [ ] `tasks.$slug.tsx` — render `ModelPage` with disabled slots and a "not available yet"
      OUTPUT, so an unimplemented task looks like an empty task page
- [ ] `playground.tsx` — point at the shared pattern rather than describing per-task wiring

### Phase 6 — Docs, E2E, and the guide

- [ ] `docs/guides/adding-a-model.md` — add a "new task page" section built on the
      checklist in the standard
- [ ] `docs/explanations/architecture.md` — reference the pattern in the frontend section
- [ ] `e2e/pages/ModelPage.ts` — one page object for the four slots; fold
      `TensorPage.ts` into it
- [ ] `e2e/specs/model-page.spec.ts` — one parameterised spec asserting the four-slot
      contract across every real task route
- [ ] Mark this plan `Complete` and `git mv` it to `docs/plans/completed/`

## Testing

**Unit tests**
- `useModelWorker`: every Machine A transition — `idle → loading → ready`, `loading →
  error → retry() → loading`, model change → teardown → `idle`; `progress` does not change
  `status`
- `useModelWorker`: Machine B — id correlation with two overlapping requests; `running`
  stays true until *both* resolve (regression test for defect 2); an `{error, id}` leaves
  `status === "ready"`; teardown rejects all pending with "Worker terminated"
- The existing `useTts` / `useAsr` / `usePipeline` / `useAudioClassifier` tests must pass
  unmodified through Phase 1 — that is the phase's proof of correctness
- `ModelStatus`: renders idle/loading/warmup/ready/error, and fires `onLoad`/`onRetry`
- `OutputPanel`: empty, running, result, and error renderings

**Integration / route tests** (`src/__tests__/routes/`)
- Each migrated route: renders four slots; RUN controls disabled until `ready`; nothing
  downloads before `load()` is clicked (assert no worker `load` message on mount)
- Error placement: a load failure appears in LOAD, an inference failure in OUTPUT while
  the page stays interactive
- `tasks.$slug`: renders the shell with disabled slots

**E2E** (`e2e/`)
- Parameterised across real task routes: the four slots exist, SELECT locks during load,
  OUTPUT has an empty state on first paint
- The mocked-API default run must not attempt a real model download — extend
  `e2e/fixtures/mockApi.ts` to stub the worker/CDN path
- Existing `webgpu/` specs still pass against the tensor route after Phase 5

**Manual**
- `just fe-dev`, visit each task route: nothing downloads until clicked; size estimate
  matches what DevTools transfers; kill the network mid-load → error in LOAD with a
  working Retry; switch models mid-load → clean teardown, no orphaned worker
- Verify on both backends: normal Chrome (WebGPU) and with WebGPU disabled (WASM fallback)

## Risks & Notes

- **Deferred load is a UX regression for small models.** Kokoro at 82M was previously
  loading while the user read the page. Mitigation: keep the estimate prominent and the
  load button primary; consider an opt-in "auto-load models under N MB" preference in
  `store/ui.ts` as a follow-up, not in this plan.
- **Phase 1 is a pure refactor and must stay one.** If a test needs changing in Phase 1,
  that is a signal the refactor changed behaviour — stop and re-check.
- **`useLiveAsr` is the genuine outlier.** It owns a `MediaStream` and a real-time loop;
  it should consume `useModelWorker` for the model half and keep its capture machine
  separate. Do not try to force the capture loop into the shared contract.
- **`training.tsx` may not fully fit.** Its "run" is a long-lived loop with pause/step, not
  a request/response. Adopt the visual bands; if the hook contract fights it, leave
  `useLinearTraining` alone and record why here.
- **Open question:** should `result` hold only the latest output, or a history? ASR
  accumulates a transcript, classification replaces the label list. Current lean: keep
  `result` as latest-only in the contract and let history be task-specific state.
- Worker message protocol unification (Phase 1) touches `tts.worker.ts`,
  `asr.worker.ts`, and `pipeline.worker.ts` together — a single commit, since the protocol
  types are shared.
