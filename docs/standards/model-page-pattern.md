# Model Page Pattern

Every task page in Model Playground is the same four-stage pipeline:

```
  ┌────────┐    ┌──────────┐    ┌───────┐    ┌────────────┐    ┌────────┐
  │ SELECT │───▶│   LOAD   │───▶│ INPUT │───▶│  GENERATE  │───▶│ OUTPUT │
  └────────┘    └──────────┘    └───────┘    └────────────┘    └────────┘
   which         [button]        give it       [button]          show what
   model?        weights         something     run the           came back
                 → memory        to chew on    model
   ─ choose ─    ─ commit ─      ─ choose ─    ─ commit ─
```

**Two of those five are buttons, and they are the only two.** SELECT and INPUT are
*choices*: they cost nothing, they can be changed freely, and they are reversible.
LOAD and GENERATE are *commitments*: one spends the user's bandwidth and the tab's
memory, the other spends their GPU and their time. Choosing never commits — that is
the one sentence this whole document is arguing for.

Four **bands** render this (SELECT · LOAD · INPUT · OUTPUT — GENERATE is the trigger
that lives in the INPUT band's transport row), which is why `ModelPage` has four slots
and the testids are `slot-1`…`slot-4`.

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
2. **Loading is explicit and consented, and the LOAD button is the only thing that does
   it.** Model weights are the user's bandwidth and the tab's memory budget. Show the
   size *before* the download starts, and start it on that press — never as a side
   effect of navigating to the page, selecting a model, or returning to a page you were
   on earlier. **A cache hit does not change this.** Cached weights make the click cheap;
   they do not make it unnecessary, and "it was free" is not the user's judgement to
   have made for them. See §5b.
3. **One state machine, one vocabulary.** Every task hook exposes the same `status`,
   `progress`, `backend`, `error`, `run`, `running`, `result`. A reviewer who has read
   one task hook has read all of them.
4. **The worker is the boundary.** Stages LOAD and RUN happen in a Web Worker; the page
   only ever sees messages. Nothing model-shaped blocks the UI thread.
5. **Degrade, don't disappear.** No WebGPU → WASM. No mic → file upload. Model failed to
   load → an error in the LOAD slot with a retry, not a blank page.
6. **Running is explicit, and the GENERATE button is the only thing that does it.**
   Choosing an input — clicking a sample, dropping a file, finishing a recording,
   placing a point on a picture — puts it in the INPUT band and stops there. It never
   starts an inference. Nor does editing a parameter beside it: a prompt, a label set,
   a template, a threshold. The consequences of getting this wrong are not subtle:
   a row of five sample images becomes five hidden run triggers, so browsing them costs
   five inferences; a 60-second audio clip is transcribed because someone wanted to read
   its reference text; and the GENERATE button sitting beside them does nothing the user
   has not already been charged for. The corollary is worth stating on its own: **the
   INPUT band's sources are not gated on a loaded model.** Picking what to run before
   deciding what to run it with is a sensible order to work in, and gating the sources
   forces a download before the user is allowed to choose.
7. **Derivation is not running.** Once a result is in hand, a control that only re-reads
   it re-derives on the main thread and asks the model nothing — `/vad`'s threshold,
   detection's confidence floor, segmentation's opacity and class toggles. §7 covers
   this; it is the one case where a control may act without a press, precisely because
   it spends nothing.

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
                       ├── {partial, id} ─▶ partial = p  ↺  (inflight unchanged)
                       ├── {result, id} ──▶ resolve · inflight -= 1 · partial = null
                       ├── {error, id}  ──▶ reject  · inflight -= 1 · partial = null
                       │                    status stays "ready"
                       └── teardown     ──▶ reject all pending · clear
```

`running` is `inflight > 0`. Track a **count, not a boolean** — a boolean is wrong the
moment two requests overlap (the first to return clears the flag while the second is
still in flight).

The `id` on an error message is the discriminator: `id != null` is a request failure
(Machine B), `id == null` is a load failure (Machine A).

### `partial` — progress inside one run

A generative decoder produces its answer over seconds; a VLM encodes the image before a
single token exists. A page that can only render the finished result therefore shows an
unlabelled multi-second pause, which is indistinguishable from a hang. So the worker may
report intermediate state against the request `id` that will eventually carry the result:

```ts
export type ModelResponse<TResult, TPartial = never> =
  | { type: "progress"; progress: ModelProgress }   // Machine A, no id
  | { type: "ready"; model: string; backend: Backend }
  | { type: "partial"; id: number; partial: TPartial }   // Machine B, in-run
  | { type: "result"; id: number; result: TResult }
  | { type: "error"; id?: number; error: string };
```

Three things it deliberately is **not**:

| | |
|---|---|
| not a `ModelStatus` | Machine A is untouched. `ready` means the weights are loaded; a run in progress does not move it, and there is no fifth state. |
| not `progress` | that variant is Machine A's download/warm-up self-loop and carries no `id`. Overloading it makes "downloading" and "generating" the same event. |
| not a running flag | `running` stays an inflight **count**. A partial neither opens nor closes a request. |

`TPartial` defaults to `never`, which is what makes the variant free for every worker
that does not stream: the arm is uninhabited, so existing exhaustive switches stay
exhaustive and no non-streaming engine or test changes.

`useModelWorker` surfaces it as `partial`, cleared when a run starts and again when its
result or error lands — and **dropped if its request has already settled**, so a late
chunk cannot repaint OUTPUT after the finished answer is on screen. A page renders
`partial ?? result`, never both.

Only the most recent run's partials are kept. That is a deliberate narrowing: `run()`
still resolves every overlapping request correctly, but one field cannot represent two
streams, and a page that genuinely interleaved two generations would have to say which
one it is drawing. `/image-text-to-text` is the reference caller.

---

## 3. The hook contract

Every task hook returns this shape. Extra task-specific fields are additive; nothing
here is optional or renamed per task.

```ts
export interface ModelTask<TInput, TOutput, TOpts = void, TPartial = never> {
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
  /**
   * Latest in-run progress from a streaming worker — partial text, an
   * encode/generate stage. Null unless a run is in flight, and cleared the moment
   * its result lands. Optional: most tasks are single-shot and never post one.
   */
  partial?: TPartial | null;

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

**`cancel()` belongs to Machine A. A long run needs its own exit, and it is `stop()`.**
`/super-resolution` is the case: one *run* is thirty inferences, one per overlapping
tile, and a user who wants out should not have to throw away the weights to get it. So
`useSuperRes` adds `stop()` — abandon the remaining tiles, stay `ready` — alongside the
`cancel()` that abandons a download. Two rules come with it:

- **Keep them distinct.** `cancel()` returns to `idle`; `stop()` leaves the model
  loaded. Overloading one name for both is how a Stop button ends up costing a 54 MB
  re-download.
- **`running` must mean the whole sequence.** The inflight count drops to zero between
  tiles, so a hook that forwards it verbatim flickers the spinner and re-enables the
  transport thirty times during a single upscale. `useSuperRes` returns
  `pipe.running || tiles != null`.

A multi-inference run also owes **progress in its own terms** (`{ done, total }` tiles),
derived from the loop rather than from a new worker message — the protocol stays one
message per inference, which is what keeps the geometry unit-testable on the main thread.

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

Note what is and is not gated in band 3. The **transport row** — the GENERATE trigger,
and `/asr`'s Start listening — is gated on `ready`; that is the commitment. The **input
surface above it** is not: the sample row, the upload button, the drop target, the mic,
and any parameter beside them all work in `idle`, because choosing what to run costs
nothing and needing a model first inverts the order people actually work in.

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
- The GENERATE trigger is `disabled={!ready || running || <no input held>}`. There is no
  "queue it up" affordance, and no path to an inference that does not go through it. The
  input *sources* above it are not gated at all (§1.6). The transport row is
  `sticky bottom-0` inside the input column, so a tall input surface can scroll without
  carrying GENERATE out of reach — sticky, not bottom-pinned: pinning leaves a chasm on
  every task whose input is short.
- OUTPUT always renders, and fills its column. An empty state that describes the coming
  result is worth more than a collapsed section, and it stops the page from jumping
  when the result lands.
- Errors render **in the slot that produced them**. A load error goes in LOAD, a decode
  error in RUN, an inference error in OUTPUT.
- The header's `aside` is for one-line answers that don't deserve a band — a backend
  chip, a device line. It is not a fifth stage.

### 4b. The INPUT band holds the input, and holding it is the point

The input is **state**, not an event that passes through on its way to the worker. Every
route keeps the decoded thing — a `RawImage`, a `Float32Array`, a list of points — and
hands the worker a copy or a downscale of it. Three things follow, and each of them was
a bug before this was true:

- **Re-running costs nothing extra.** Switch checkpoint, press GENERATE again, same
  input. No re-fetch, no re-decode, no second recording — which matters most for the
  inputs that are expensive to obtain: a 60-second clip, or a take the user recorded
  once and cannot reproduce.
- **A parameter is a reason to re-run, not a reason to re-capture.** CLAP's prompts,
  zero-shot's template, a VLM's question on `/image-text-to-text`: edit, press GENERATE,
  compare.
  Editing the prompts used to mean recording again.
- **An audio worker detaches the buffer it is given.** `useAudioPick`'s `take()` returns
  `clip.audio.slice()` for exactly this reason — hand over the stored array itself and
  the waveform goes blank and the second GENERATE has nothing to send. The vision side
  has the same trap in `toPayload`'s `copy` flag.

The shared input surfaces are [`useImagePick`](../../frontend/src/hooks/useImagePick.ts) with
[`ImageSourcePanel`](../../frontend/src/components/vision/ImageSourcePanel.tsx), and
[`useAudioPick`](../../frontend/src/hooks/useAudioPick.ts) with
[`AudioSourcePanel`](../../frontend/src/components/audio/AudioSourcePanel.tsx).
Neither pick hook takes a callback, and that is deliberate: `useImagePick` used to accept
an `onPicked`, thirteen routes used it to fire an inference on decode, and the parameter
is gone rather than merely unused so it cannot come back one route at a time.

**The result belongs to the input that produced it, and they are separate state.** A
route captures the frame or clip it actually ran on (`frame`, `source`, `scored`,
`enhanced`) *inside* its run function, and OUTPUT renders that — not the input currently
held. Otherwise picking a new clip redraws the waveform under the previous result's
timeline, and the page shows a comparison that was never computed.

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
- **the weights**, in the browser's cache, which is what makes the next load free.

**A refresh restores the decision, never the download.** The page comes back on the
model you chose, in `idle`, with the LOAD button saying "Load model (cached)" and the
copy beside it saying the bytes are already here. Then it waits.

This was not always true, and the reason it is worth a section is the argument that
made it untrue. An earlier revision also persisted *the intent to load* — a flag set
when you pressed Load — and `useModelSelection` resumed the load on mount whenever that
flag met a cache hit. The reasoning was airtight on cost: the weights are local, the
resume spends no bandwidth, and the user had already consented once. It was still wrong,
for two reasons that cost has nothing to say about:

- **A cached load is not a free load.** It still occupies hundreds of megabytes of the
  tab's memory, still holds a GPU device, and still takes seconds of a warm-up the user
  is now watching instead of doing what they came to do.
- **A page that starts working before you ask it to is a page you do not control.** The
  user arrives, has touched nothing, and the machine is busy. Whether that cost bytes is
  not the question they are asking.

So the flag is gone, and `store/models.ts` declares `partialize` to keep a stale
`autoResume` key in someone's existing `localStorage` from quietly reviving the
behaviour. What the cache probe (`model/cache.ts`) is still for is **telling the truth
about the click**: "Load model (cached) · already downloaded" is a different offer from
"Load model · 88 MB", and a user makes a different decision about each. Informing a
choice is the useful half of knowing the weights are there; making the choice for them
was not.

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
| Voice Activity Detection | model catalogue, one entry with **no weights** | ONNX graph, or nothing at all for the energy baseline | mic / file / sample clip | probability timeline + a threshold the user drags |
| Tensor Arithmetic | the operation | WGSL pipeline compile (fast, auto) | operand matrices | heatmap + numeric grid |
| Linear Training | architecture + hyperparams | dataset fetch + kernel compile | train loop | live weights + loss curve |
| Graph ML | the GNN architecture | **a GPU probe and a bundled dataset**, neither of them a download | Train / Stop, and a **depth slider that re-trains** | the graph repainted per epoch, plus accuracy against depth |
| Link Prediction | the GNN architecture | a GPU probe, and **the split** — the held-out fraction lives here because changing it rebuilds the graph and its layout | Train / Stop | the graph with predicted non-edges dashed over it, an AUC, and a **top-k slider that only re-draws** |
| Graph Classification | the GNN architecture **and the readout** (sum or mean) | a GPU probe, and the **one real download in the category** — 2 MB of dataset, cached after the first visit | Train / Stop | a gallery of held-out molecules outlined right/wrong, and an accuracy **never shown without its baseline** |
| Mask Generation | model catalogue | weight download, **then a per-image encode** | click a point on the picture | the mask, its candidates, its decode time |
| Keypoint Detection | **a pair** of checkpoints | both downloads, one aggregate bar | image / camera, threshold, people cap | skeletons over the frame |
| Video Classification | CLIP catalogue, reused | weight download | a clip, labels, a sample rate | scores over time + a filmstrip |
| Background Removal | model catalogue **+ its licence** | weight download | image / camera frame | cut-out on a checkerboard, backdrop swap, raw matte, PNG |
| Super Resolution | model catalogue | weight download | an image, **plus the tile count and time it will cost** | a draggable split against a bicubic baseline |
| Image to 3D | depth catalogue, reused | weight download **and** a GPU probe | image / camera, focal + density sliders | an orbitable point cloud, or the depth map and why not |
| `/tasks/$slug` placeholder | — | — | — | "not available yet" |

Where LOAD is fast and free (a shader compile), it may auto-run — pass `autoLoad`. The
slot still renders, so the page keeps the same four-band rhythm as its neighbours; use
[`DeviceStatus`](../../frontend/src/components/model/DeviceStatus.tsx) there, which answers
the question those pages actually raise — is there a GPU, or nothing to compute on. The
placeholder route uses the same shell with empty slots, so an unimplemented task reads
as *the same kind of page*, not a different app.

**A result the user can re-read without re-running belongs on the main thread.** `/vad` returns
per-frame speech probabilities and lets the user drag a threshold; the segments that threshold
implies are derived by a pure function
([`audio/vad/segments.ts`](../../frontend/src/audio/vad/segments.ts)) over the result already in
hand. Nothing is re-posted to the worker. The rule generalises: when OUTPUT has a knob, ask
whether the knob changes the *model's* answer or only the *reading* of it. A knob that only
re-reads must never re-run — otherwise the control lags the pointer, `running` flickers, and the
page charges the user's battery for a display preference. A knob that genuinely changes the
input (a prompt, a length, a voice) belongs in RUN, not OUTPUT.

**A page may have nothing to download at all.** `/graph` has no checkpoint — the network
is written as WGSL and trained in the tab — and its dataset is 161 KB bundled with the
app. LOAD is therefore two questions, neither of them bandwidth: is there a GPU
([`DeviceStatus`](../../frontend/src/components/model/DeviceStatus.tsx)), and has the
graph been decoded and laid out. The GPU answer comes first, because it decides which of
the two compute paths a run takes. The band still exists and still has to be pressed:
the layout behind it costs about a second, and a page that silently spent that on arrival
would be doing work the user did not ask for.

**A catalogue entry may cost nothing.** `/vad` offers an energy-based baseline alongside Silero:
`bytes: { webgpu: 0, wasm: 0 }`, no repo, no download. It still passes through SELECT → LOAD →
RUN like any model, so LOAD resolves immediately and the page works before anything is fetched.
Where a task has a credible no-model baseline, shipping it beside the model is worth more than a
paragraph of documentation about when the model is overkill.

**LOAD may not be the only wait.** `/mask-generation` has two: the download, and a
per-image *encode* that is the slow half of the first click. They are separate states
and the page shows both — collapsing them means the user clicks, waits a second, and is
told nothing, which is precisely the experience SAM's encode-once/decode-many
architecture exists to avoid. The rule generalises: **a wait the user cannot predict
needs a name in the UI.** Put the second wait in RUN, next to the input that caused it,
not in LOAD — the model is loaded, and a LOAD slot that goes busy again would say
otherwise. Give it its own testid so a spec can tell the two apart.

**A stage may cost two models.** `/pose` is the single deliberate exception to "one model
live at a time": top-down pose is a detector followed by a pose model on each person's
crop, and neither half is useful alone. What the pattern requires in exchange is that the
*user-facing* numbers stay whole — the catalogue entry names both checkpoints and quotes
their **combined** download, and LOAD shows one aggregate bar rather than two competing
ones. (That last one needed a fix: `model/progress.ts` keys its file table on **repo +
file**, because both checkpoints publish an `onnx/model_fp16.onnx` and the second was
overwriting the first's entry — the bar reached 100% halfway through and the second
download read as a stall.)

**A run whose cost the user cannot guess must be quoted before it starts.**
`/super-resolution` is many inferences, and how many depends on the picture: the RUN
slot states the output size, the tile count and a rough duration for the resolved
backend *before* the button is pressed, and Stop is offered while it runs. The general
rule is the sibling of the size-before-load guardrail in §5 — **§5 is the bandwidth a
model costs; this is the time a run costs** — and it applies wherever a run is more than
one forward pass. Where the estimate comes from a constant, keep the constant honest:
`MS_PER_TILE` was a guess an order of magnitude out until the `@slow` spec measured it.

**LOAD may be two different questions at once.** `/image-to-3d` downloads weights *and*
needs a GPU device for its render half, and the second can fail on a machine where the
first succeeds. Both live in LOAD, and the GPU answer is given **before** the download
rather than after it — better to learn there is no WebGPU before spending 50 MB. When
the device is missing the page still runs the model and shows the depth map, saying what
is unavailable and why; `detectWebGPU()` never throws, so the failure is a status, and a
page that answered it with an empty canvas would read as "the model failed".

**A licence can belong in SELECT.** Where a catalogue entry's terms constrain what the
user may do with the output — `/background-removal` offers one model that is
non-commercial only — the constraint is a property of the *choice*, so it renders beside
the picker at the moment the choice is made, in the same amber as the size guardrail.
The permissively licensed model is the default. A restriction discovered after the
download is a restriction discovered too late.

**A number the reader cannot calibrate is not a result.** `/graph-classification` is the
case that names it: PROTEINS is 663/450, so a model that ignores its input entirely scores
59.8 %, and an accuracy of 72 % printed on its own is indistinguishable from one that
learned the class prior. The page renders the **majority baseline** beside every accuracy,
states the margin in points, and says plainly when there is no margin. The general rule is
that a metric owes its null model on screen wherever one exists — chance for a balanced
classifier, the majority class for an unbalanced one, a bicubic upscale for
`/super-resolution`, an energy threshold for `/vad`. Carry it *with* the metric rather than
recomputing it in the view, or the two can be derived from different data.

**A control that changes the *data* belongs in LOAD, and re-loads.** `/link-prediction`'s
held-out fraction is the case: a different fraction is a different split, which is a
different graph and a different force-directed layout, so it cannot be a hyperparameter
passed to the next run. It sits in LOAD with the device probe, and the page says that
moving it costs a reload. The rule generalises — ask which stage's *output* the control
invalidates, and put it there.

**A control that changes the model's *shape* belongs in RUN, and re-runs.** `/graph`'s
depth slider is the case that makes the §7 rule concrete in both directions at once: on
the same page, depth (RUN) re-trains because 8 layers is a different model from 2, while
the predicted/true colour switch (OUTPUT) re-reads the run already in hand and never
re-runs. Having both on one page means the difference has to be *stated* rather than
inferred, so the page says it beside the slider.

**A control in RUN may legitimately re-run.** The §7 rule above — a knob that only
re-reads must never re-run — is about OUTPUT. `/pose`'s person-confidence slider and its
people cap sit in RUN and *do* re-run, because filtering afterwards would mean running the
pose model on people the user has already excluded, and that pass is the expensive half.
The test is not "is it a slider" but **"does this change what the model is asked?"** When
the answer is yes, say so on the page, so the difference from `/object-detection`'s
identical-looking slider reads as a decision rather than an inconsistency.

**Some models cannot run here at all, and the page should say so before the click.**
A catalogue entry may declare `backends`; `model/useBackendProbe.ts` answers what a load
would resolve to *before* anything downloads, and `ModelPicker` disables a model the
machine cannot run with the reason on its row. An undecided probe (`null`) gates nothing
— treating it as WASM greys out every WebGPU model for a frame on each page load, which
reads as "this machine cannot run it" rather than "ask again in 10 ms".

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
| `live-fps` | `ImageSourcePanel` | Measured end-to-end frame rate. Present only while the camera runs. |
| `audio-input` | `AudioSourcePanel` | The held clip: its name, duration, rate and waveform. The audio routes' "what am I about to run on?". |
| `audio-input-empty` | `AudioSourcePanel` | No clip chosen yet. The audio counterpart of `output-empty`. |
| `heavy-model-notice` | `/depth` · `/zero-shot-classification` | The opt-in a heavy model gets on top of the size line. Driven by `isHeavyDownload` in `model/size.ts` — a **size** predicate, so it is not bound to either page's catalogue. |
| `class-space` | `/segmentation` | What the selected model can possibly say, stated before the run. |
| `depth-map` · `detection-canvas` · `segmentation-canvas` | vision routes | The rendered overlay, once a result exists. |
| `encode-cost` | `/zero-shot-image-classification` | Per-tower timing, showing when the label embeddings were reused. |
| `template-verdict` | `/zero-shot-image-classification` | The top label under each prompt template. |
| `model-unsupported-<id>` | `ModelPicker` | That model needs a backend this machine did not resolve to. |
| `query-groups` | `/zero-shot-object-detection` | Detections grouped by phrase — **including the phrases that found nothing**. |
| `neighbours` · `index-size` · `embedding-facts` | `/image-features` | The ranked list, how many pictures are indexed, and the vector's norm before/after. |
| `encoding` · `encoded` | `/mask-generation` | The per-image encode, in flight and finished. Distinct from LOAD. |
| `point-canvas` · `points` | `/mask-generation` | The clickable input surface, and the points placed on it. |
| `mask-canvas` · `mask-facts` · `decode-ms` | `/mask-generation` | The mask, its coverage and IoU, and the decode time the page claims. |
| `pose-canvas` · `people` · `joints` | `/pose` | Skeletons, one entry per person, and per-joint confidence **and position**. |
| `clip-progress` · `filmstrip` · `pooled-winner` · `baseline-note` | `/video-classification` | Sampling/scoring progress, the frames the model saw, the clip-level verdict, and the limitation. |
| `pass-count` · `composed-hypothesis` · `template-problem` | `/zero-shot-classification` | What one press will cost in forward passes, the exact hypothesis the first label composes to, and the refusal when the template has no `{}`. All three are derivations over the input — none of them runs anything. |
| `ran-labels` · `ran-template` · `scoring-note` | `/zero-shot-classification` | The label set and template the answer on screen was **actually** produced with, captured inside the run, and which normalisation produced the numbers. |

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

- Nothing downloads on mount — assert the hook was called with **no `autoLoad` argument
  at all** (its default is `idle`) *and* that `load` was not called.
- `load()` fires from the LOAD slot, `retry()` from its error state, `cancel()` from the
  progress row.
- After a refresh: the stored selection is restored, and the page is still `idle` —
  **including when the weights are cached**, which is the case that used to auto-load.
  Seed a stale `autoResume` key in `localStorage` and assert it changes nothing.
- **Choosing an input runs nothing.** Pick a sample with a model `ready` and assert
  `run` was *not* called and `output-empty` is still visible; then press GENERATE and
  assert it was. This is the assertion most worth writing, because the failure it
  catches — an input that silently triggers an inference — looks like a working page.
- **The input survives its run.** Press GENERATE twice, or change a parameter and press
  again, and assert the source was consumed once (one `fetch`, one `decodeToMono`, one
  `recordMic`) while `run` was called twice.
- The input *sources* are usable in `idle`; only the GENERATE trigger is gated on
  `ready`.
- All four slots render, with `output-empty` visible, before any run.
- A load error renders in LOAD, a run error in OUTPUT, and the page stays usable.
- Any control the page says re-derives **does not** call `run` (assert the call count),
  and any control the page says re-runs **does**.
- Where a route declares `backends`, an unsupported model is offered but not selectable —
  and an undecided probe gates nothing.

## 9. Checklist — adding a task page

- [ ] Model catalogue entry: `id`, `label`, `hint`, `params`, measured `bytes` per backend
      — plus `graphs` where the repo has no `model.onnx`, and `backends` where a backend
      genuinely cannot run it
- [ ] Worker built on the shared protocol (`load` / `run` messages, id-correlated)
- [ ] Hook wraps `useModelWorker`; returns the §3 contract verbatim
- [ ] Page uses `ModelPage` + the four slots — no bespoke layout, no grid of its own
- [ ] The input surface fills its column (`flex-1`) if it is the column's main element
- [ ] `idle` default: nothing downloads until the LOAD button is pressed — not on
      arrival, not on a model change, not on a refresh with the weights cached
- [ ] Selection persisted through `useModelSelection`, with the route's own `routeKey`
- [ ] Size estimate + large-model warning shown before load
- [ ] The input is **held** (`useImagePick` / `useAudioPick` or equivalent state), and
      choosing one runs nothing — no callback fires an inference on decode
- [ ] Exactly one GENERATE trigger, gated on `ready` + an input; the input sources above
      it are not gated at all
- [ ] A parameter beside the input re-runs on the next GENERATE, never on the keystroke
- [ ] OUTPUT has an empty state, a running state, and an error state
- [ ] Errors land in the slot that produced them
- [ ] Unit tests for the hook's state machine; a route test covering §8's list
- [ ] Sidebar taxonomy entry mapped in `REAL_ROUTES`, and the taxonomy test flipped
- [ ] Added to `e2e/specs/model-page.spec.ts`'s route table and to `model-ids.spec.ts`
- [ ] A `@slow` spec that asserts a **property**, never a count

---

## Related

- [`model-visualization.md`](model-visualization.md) — how a model and its internals are drawn
- [`../explanations/webgpu-inference.md`](../explanations/webgpu-inference.md) — how inference runs
- [`../guides/adding-a-model.md`](../guides/adding-a-model.md) — kernel + registry entry
- [closed plan issues](https://github.com/bthek1/model_playground/issues?q=is%3Aissue+is%3Aclosed+label%3Aplan) — how this pattern was arrived at: the restructure, the move from one column to rail + workbench, and aggregate progress / cancel / refresh

## Reference implementations

| Piece | File |
|---|---|
| State machines | [`frontend/src/model/useModelWorker.ts`](../../frontend/src/model/useModelWorker.ts) |
| Contract types | [`frontend/src/model/types.ts`](../../frontend/src/model/types.ts) |
| Shell + slots | [`frontend/src/components/model/`](../../frontend/src/components/model/) |
| A weight-downloading page | [`routes/text-to-speech.tsx`](../../frontend/src/routes/text-to-speech.tsx) |
| The smallest complete page | [`routes/image-classification.tsx`](../../frontend/src/routes/image-classification.tsx) — an image in, a ranked list out; read this one first |
| A compile-only page | [`routes/tensor.tsx`](../../frontend/src/routes/tensor.tsx) |
| A page whose OUTPUT has a knob | [`routes/vad.tsx`](../../frontend/src/routes/vad.tsx) — the threshold re-derives, never re-runs |
| A page whose input costs nothing | [`routes/image-classification.tsx`](../../frontend/src/routes/image-classification.tsx) — picking a picture works before a model exists, and classifies the moment one does |
| A page with a second, named wait | [`routes/mask-generation.tsx`](../../frontend/src/routes/mask-generation.tsx) — LOAD is the download, `encoding` is the per-image pass |
| A page that holds two models | [`routes/pose.tsx`](../../frontend/src/routes/pose.tsx) — one combined size, one aggregate bar, controls that re-run on purpose |
| A page whose framing is a requirement | [`routes/video-classification.tsx`](../../frontend/src/routes/video-classification.tsx) — a frame-level baseline, said so in copy an E2E spec asserts |
| A page whose answer streams | [`routes/image-text-to-text.tsx`](../../frontend/src/routes/image-text-to-text.tsx) — the encode is named, the tokens arrive over seconds, and `partial` carries both |
| The empty case | [`routes/tasks.$slug.tsx`](../../frontend/src/routes/tasks.$slug.tsx) |
| The contract, asserted | [`frontend/e2e/specs/model-page.spec.ts`](../../frontend/e2e/specs/model-page.spec.ts) |
