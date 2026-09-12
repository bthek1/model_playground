# The system panel

A hideable right-hand panel that shows what the machine is doing while a model
loads and runs: GPU, memory, CPU and storage, as readouts plus two-minute
sparklines. It is collapsed by default, toggled from the navbar or with
`Alt+Shift+M`, and it samples **nothing** while it is shut.

Plan: [issue #25](https://github.com/bthek1/model_playground/issues/25).

---

## 1. The constraint that shaped it

**A web page cannot read OS-level CPU, GPU or RAM utilisation.** There is no API
for host CPU percent, none for GPU utilisation, and none for VRAM — not "not
yet", but deliberately, because all three are fingerprinting surfaces and
side-channels. Anything that looked like a task manager here would be invented.

So the panel reports **load and capacity, never utilisation**, and every card
carries one line saying which of the two its number is. Where a measurement
cannot be made at all, the card renders *the reason it cannot* rather than a
zero. That rule is in the type system, not in a convention:

```ts
type Metric<T> = { status: "ok"; value: T } | { status: "unavailable"; reason: string };
```

A memory card reading "0 MB" in Firefox looks like a working card reporting an
idle tab, and a user would believe it. `unavailable` makes that unrepresentable.
This is the same rule `/video-classification` and `/super-resolution` follow: a
surface that quietly answers a smaller question than its name promises is the
failure mode, and stating the limit next to the number is the fix.

## 2. What is actually measured

| Card | Measures | Source | Notes |
|---|---|---|---|
| **GPU** | adapter identity, limits, features; bytes **we** allocated and have not released; last WGSL compute-pass duration | [`webgpu/capabilities.ts`](../../frontend/src/webgpu/capabilities.ts), [`webgpu/allocations.ts`](../../frontend/src/webgpu/allocations.ts), [`webgpu/timing.ts`](../../frontend/src/webgpu/timing.ts) | A ledger, not a probe. No VRAM figure exists. ORT's own GPU memory is invisible. |
| **Memory** | JS heap of **this page's realm**; device-memory class | `performance.memory`, `navigator.deviceMemory` | Chrome/Edge only. The weights live in workers, whose heaps are not readable from here. |
| **CPU** | inference-busy fraction; worst frame overrun per second; long-task count; logical cores | [`telemetry/activity.ts`](../../frontend/src/telemetry/activity.ts), rAF deltas, `PerformanceObserver('longtask')`, `navigator.hardwareConcurrency` | Responsiveness and app activity, not host load. |
| **Storage** | origin usage against quota; models in the cache; bytes/s of any download in flight | `navigator.storage.estimate()`, [`model/cache.ts`](../../frontend/src/model/cache.ts), [`model/progress.ts`](../../frontend/src/model/progress.ts) | The panel's most solid number: for this app the footprint *is* cached weights. |

### Per-browser availability

| | Chrome / Edge | Firefox | Safari |
|---|---|---|---|
| GPU identity + ledger | yes | yes (needs `dom.webgpu.enabled` on Linux/macOS) | yes (26+) |
| WGSL pass timing | where `timestamp-query` is advertised | usually absent | usually absent |
| Page heap (`performance.memory`) | yes | **no** | **no** |
| Device memory class | yes | **no** | **no** |
| Long tasks | yes | **no** | **no** |
| Busy fraction, lag | yes | yes | yes |
| Storage estimate | yes | yes | yes (refused in private mode) |

Firefox and Safari therefore show fewer numbers and more reasons. That is the
correct outcome, and the reason strings are the deliverable there.

## 3. A monitor must not be the thing it measures

The sampling budget is an acceptance criterion, not a nicety.

- **Closed means off.** `useTelemetry(active)` runs no timer, no observer and no
  rAF chain unless the panel is open, and the docked panel and the narrow-screen
  Sheet are mutually exclusive renders (via `useMediaQuery`) so only one loop can
  ever exist. Hiding one with CSS would have mounted two.
- **Hidden tab means off.** A background tab throttles timers and rAF to the
  point where every sample is a lie about the machine, so the loop stops on
  `visibilitychange` and restarts when the tab returns.
- **One interval, 1 Hz.** The only per-frame work is reading
  `requestAnimationFrame`'s timestamp, which *is* the lag measurement.
- **The expensive sampler runs rarely.** Counting cached models walks every key
  in the Cache Storage bucket, so it runs on the first tick and then once every
  `STORAGE_EVERY_TICKS` (10). It only changes when a download finishes.
- **A slow tick is skipped, not queued.** If a sample is still in flight when
  the next interval fires, that tick is dropped. Queueing would leave the panel
  reporting a machine several seconds stale.
- **Samples are not application state.** They live in fixed-capacity ring
  buffers ([`telemetry/series.ts`](../../frontend/src/telemetry/series.ts))
  behind refs, and reach React through exactly one `setState` per tick. In
  Zustand they would re-render the app to move a sparkline; in TanStack Query
  they would be pretending to be server state.
- **Closing clears the history** rather than pausing it. A chart drawn across a
  ten-minute gap is a lie about continuity, and a sparkline cannot draw the gap.

Sparklines are hand-written inline SVG, per
[`model-visualization.md`](../standards/model-visualization.md) §5 — a 28px trend
line in a card does not justify a charting library, and `var(--chart-N)` works
directly in SVG, so there is no `getCSSVar()` round-trip and no `echarts` chunk.

## 4. Two things the panel cannot observe, and who tells it

**An inference is invisible from outside the hook that started it**, and so is a
download's byte total. Both are already tracked in
[`model/useModelWorker.ts`](../../frontend/src/model/useModelWorker.ts) — machine
B's in-flight *count* and the aggregate from `model/progress.ts` — so that hook
reports them to [`telemetry/activity.ts`](../../frontend/src/telemetry/activity.ts)
and the panel reads them there. It is a one-way bridge, not a second
measurement, and both reports clean up to "nothing happening" so a route
unmounted mid-run cannot leave the panel reading busy forever.

Two deliberate non-choices:

- **Not on the worker envelope.** `ModelRequest`/`ModelResponse` carries tasks;
  a telemetry variant would be implemented by six workers to serve one panel.
- **Not `PerformanceObserver('resource')` for download rate.** The Hub's
  responses are cross-origin without `Timing-Allow-Origin`, so resource timings
  report a transfer size of **zero**. Differencing the byte aggregate that the
  LOAD slot already computes is both accurate and free.

## 5. GPU bytes across realms

Every WGSL kernel runs in [`webgpu/worker.ts`](../../frontend/src/webgpu/worker.ts),
which owns its own `GPUDevice` and its own module instances. A counter kept on
the page would read zero through an entire training run; the only main-thread
allocation in the app is `PointRenderer`'s vertex buffer.

So each realm keeps a local ledger and the page aggregates them:

```
page realm                                 worker realm (one per GPU worker)
──────────                                 ─────────────────────────────────
trackBuffer / releaseBuffer  ◄── ledger ──► trackBuffer / releaseBuffer
aggregateSnapshot()                        recordPassMs()  (timestamp-query)
        ▲                                            │
        └───────── MessagePort, 1 Hz, only while ────┘
                   the panel is watching
```

- **A `MessagePort`, handed over by `createWebGPUWorker()`** — not a
  `BroadcastChannel`, which is origin-wide and would fold a *second tab's*
  allocations into this page's total.
- **The port is also the liveness signal.** A terminated worker simply stops
  publishing and is dropped after `STALE_MS`; there is no teardown message to
  miss, because `terminate()` sends none.
- **Nothing is published unless the panel is watching.** `watchAllocations(true)`
  when it opens, `false` when it closes.
- **Buffers are freed with `releaseBuffer(buffer)`, not `buffer.destroy()`.** The
  ledger remembers each buffer's size at creation, so no call site has to pass a
  byte count that could drift from the allocation.
- Aggregated `peakBytes` is a **sum of per-realm peaks**, so it is an upper bound
  when two realms peaked at different moments — the right direction to err for a
  ceiling.

`timestamp-query` is an optional WebGPU feature. `webgpu/device.ts` requests it
only when the adapter advertises it, and falls back to a bare `requestDevice()`
if that is refused: losing the device would break every kernel in the app to
save one diagnostic number. It times **our** WGSL passes only — an ONNX Runtime
session dispatches its own and is entirely opaque to us.

## 6. Layout

The panel is a **sibling of `<main>`**, never inside it. A route's own grid
([`model-page-pattern.md`](../standards/model-page-pattern.md) §4) is laid out on
viewport breakpoints; wrapping it in another grid cell would leave its DOM order
intact but its columns computed against a width that no longer exists.

Docked as an `aside` at `lg` and up; a right-side Sheet below that. Squeezing a
20rem panel out of a tablet-width workbench is how RUN and OUTPUT end up stacked
and the result ends up below the fold — the exact failure the horizontal layout
exists to prevent.

## 7. Out of scope, on purpose

- **`measureUserAgentSpecificMemory()`** is the accurate cross-realm memory
  answer and would replace the page-only heap figure. It requires **cross-origin
  isolation** (COOP/COEP), which changes how the Hugging Face CDN fetches and how
  ORT threads. That is its own plan, not a side effect of this one.
- **Host-level utilisation** — real CPU percent, real VRAM — needs something
  outside the browser (a native helper, or a backend agent reporting the dev
  machine). Also its own plan.

## 8. Testing

- **Unit** — the ring buffer; every sampler with its API present, absent and
  throwing; the busy/download bridge; the allocation ledger and its port
  protocol; `timestamp-query` readback including a failed one; the sampling
  loop's start/stop/skip behaviour under fake timers; each card in both states.
- **E2E** ([`e2e/specs/telemetry.spec.ts`](../../frontend/e2e/specs/telemetry.spec.ts))
  — the geometry jsdom has no engine for (four slots intact, output above the
  fold, no horizontal overflow, overlay on a narrow screen) and the one thing
  only a browser can prove: that closing the panel **stops** the sampling.
  `data-tick` on the panel body is what makes that observable.
- **Manual, and still outstanding:** the real-GPU path. Headless Chromium here
  acquires no device, so the `@webgpu` project self-skips — which means
  `requiredFeatures: ["timestamp-query"]`, `resolveQuerySet` and the cross-realm
  byte totals are covered by unit tests with fake devices and by nothing on real
  hardware. Verify on a machine with a GPU by running a training run on
  `/training` with the panel open: buffer bytes should rise and fall per step,
  and "Last WGSL pass" should show a sub-millisecond figure where the adapter
  advertises `timestamp-query`.
