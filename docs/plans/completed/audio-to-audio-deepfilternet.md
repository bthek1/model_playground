# Plan: Audio-to-Audio — DeepFilterNet3 speech enhancement in the browser

**Status:** Complete (2026-09-02) — with two verification gaps recorded in Phase 5:
the WebGPU provider and a manual listening check both need a machine this one is not.
**Date:** 2026-09-02

---

## Goal

Ship the **Audio to Audio** task client-side: take a noisy speech clip (upload or mic), run
**DeepFilterNet3** on it in the browser, and return enhanced audio the user can play,
compare and download. Route `/audio-to-audio`, wired from the taxonomy like every other task.

This is the one audio task with **no Transformers.js path**. It runs on
**`onnxruntime-web` directly**, and — unavoidably — on a DSP stack we write ourselves.

## Why this is a real project, not a wiring job

The published exports (`soniqo/DeepFilterNet3-ONNX`) are, in the author's words, *"the neural
graph only"*. The contract:

| Direction | Name | Shape | Meaning |
|---|---|---|---|
| in | `feat_erb` | `[1,1,T,32]` | normalised ERB-band energies |
| in | `feat_spec` | `[1,2,T,96]` | normalised complex spectrum, first 96 bins (re, im) |
| out | `erb_mask` | `[1,1,T,32]` | per-band gain |
| out | `df_coefs` | `[1,5,T,96,2]` | order-5 complex deep-filter coefficients |

Audio in / audio out is **ours**: STFT, ERB analysis, feature normalisation, mask application,
deep filtering, and overlap-add synthesis. The browser ships no real-FFT primitive either.

### What de-risks it

The export also ships **`deepfilter-auxiliary.bin`** (124 KB) — the canonical matrices, so the
band layout does not have to be re-derived. Verified layout, little-endian `float32`:

| Offset (floats) | Count | Contents |
|---|---|---|
| 0 | 481×32 | forward ERB matrix, `[bin][band]`; **columns sum to 1** (band mean) |
| 15392 | 32×481 | inverse ERB matrix, **`[band][bin]`**; 1 inside the band (broadcast) |
| 30784 | 960 | analysis/synthesis window |

> **Corrected during Phase 2.** The draft above read the inverse matrix as
> `[bin][band]` like the forward one. It is stored **band-major** — the model card
> says so, and the numbers confirm it (row sums are the band widths). Both readings
> yield 481 ones that look like a partition, so nothing errors; the bands simply
> draw their gains from the wrong bins. `parseAux` now asserts each band is a
> *contiguous* run, which is the invariant that separates the two.

Checked numerically: the window is exactly symmetric and power-complementary at 50 % overlap to
7e-8 (a Vorbis window), and the two matrices are consistent with `min_nb_erb_freqs: 2` and 32 bands.

### DSP constants (from the model's `config.json`, not guessed)

`sample_rate 48000` · `fft_size 960` · `hop_size 480` · `fft_bins 481` · `erb_bands 32` ·
`df_bins 96` · `df_order 5` · `df_lookahead 2` · `conv_lookahead 2` · `norm_tau 1.0` ·
`normalization_alpha 0.99` · `min_nb_erb_freqs 2`.

## Constraints

- **48 kHz, mono.** Unlike every other audio route (16 kHz), DFN3 is native 48 kHz. `decodeToMono`
  already takes a target rate, so this is a call-site change, not a new decoder.
- **Streaming-shaped, run offline first.** The normalisation carries per-frame state (`alpha`), and
  the deep filter has 2 frames of lookahead. Phase 3 processes a whole clip in one pass; a live mic
  path is explicitly out of scope until that is correct.
- **The model is small** (8.6 MB fp32 + 124 KB aux) — no size gate needed, unlike MusicGen.
- **Wrong DSP fails quietly.** It produces plausible audio with artefacts rather than an error.
  Every phase below therefore carries a numeric acceptance test, not a "looks fine".

## Phases

### Phase 1 — DSP primitives, verified in isolation ✅

*Done 2026-09-02. 24 unit tests, all offline — no model, no network.*

- [x] `audio/enhance/fft.ts` — radix-2 FFT, an **exact arbitrary-length DFT**, and `rfft`/`irfft`
- [x] `audio/enhance/stft.ts` — framing, Vorbis window, STFT/ISTFT with overlap-add
- [x] Unit tests: FFT vs a naive DFT (power-of-two *and* 60 = 2^2x15); `istft(stft(x)) ≈ x`

**The round-trip test earned its place immediately.** The first implementation zero-padded each
960-sample frame to 1024 to reuse the radix-2 FFT. That fails for two reasons the test caught at
once: reconstruction error of 0.22 (truncating bins 481–512 discards the 22.5–24 kHz band), and —
worse, and silently — **the bin spacing changes from 50 Hz to 46.875 Hz**, so every feature handed
to the network would have been misaligned against its ERB matrices. The model's 481 bins are
exactly `960/2 + 1`; padding is not an option.

Fixed with **Bluestein's algorithm** (chirp-z): an exact DFT of any length built from power-of-two
FFTs, with the chirp tables cached per length since n = 960 recurs every frame. A regression test
now asserts a 1 kHz tone lands in **bin 20 exactly** — under the padded implementation it landed
near 21, so that assertion is the guard against the bug returning.

### Phase 2 — The auxiliary matrices and ERB analysis ✅

- [x] `audio/enhance/aux.ts` — fetch + parse `deepfilter-auxiliary.bin` into
      `{ fwd: Float32Array, inv: Float32Array, window: Float32Array }`, with the layout asserted
- [x] Validate on load: window symmetric + power-complementary; `fwd` columns sum to 1; `inv` rows
      are contiguous unit runs partitioning all 481 bins, cross-checked against `fwd`'s `1/width`.
      **Refuses to run on a mismatch** rather than emit noise
- [x] `erbAnalysis(spec) → [32]` band means and `erbSynthesis(mask) → [481]` broadcast
- [x] The shipped window is used; `vorbisWindow()` is asserted equal to it

### Phase 3 — Features, inference, and reconstruction ✅

- [x] `audio/enhance/features.ts` — ERB energies in dB + the exponential normalisation
      (`tau 1.0`, `alpha 0.99`, ERB state initialised −60 → −90 dB, complex unit state 0.001 → 0.0001)
- [x] `audio/enhance/session.ts` — `ort.InferenceSession` over `["webgpu","wasm"]`, batched over all
      frames; `deepFilterNet.ts` owns the DSP either side of it
- [x] Apply `erb_mask` (broadcast to bins) then the order-5 deep filter over the first 96 bins,
      **from an immutable copy of the noisy spectrum**, honouring `df_lookahead: 2`
- [x] ISTFT + overlap-add back to 48 kHz mono
- [x] **Acceptance:** superseded by something stronger — see below

### Phase 4 — Worker, hook, route ✅

- [x] `enhance.worker.ts` + `enhanceEngine.ts` (testable handler; same three duties as every
      engine: one model live, warm-up, never block the main thread)
- [x] `hooks/useEnhance.ts` built on **`useModelWorker`**, returning the `ModelTask` contract
      verbatim — no task-named `enhance()` alias, unlike the older hooks
- [x] `routes/audio-to-audio.tsx` — upload / mic, A-B play (noisy vs enhanced), download WAV,
      before/after waveforms via the existing `components/audio/Waveform.tsx`
- [x] `REAL_ROUTES["audio-to-audio"] = "/audio-to-audio"`, replacing the placeholder assertion in
      `taskTaxonomy.test.ts`

### Phase 5 — Verification against the real model ⚠️ (WASM done, WebGPU unverified)

- [x] `@slow` E2E: loads the real ONNX session, enhances a noisy clip, asserts SDR improves ≥6 dB
      (`e2e/utils/enhance.ts`, run by `just fe-e2e-enhance`)
- [x] WASM path confirmed in a real browser: the JFK clip at 0 dB SNR comes out at
      **−0.01 dB → +13.43 dB** SI-SDR (the spec asserts a ≥6 dB gain)
- [ ] **WebGPU path not confirmed.** `e2e/specs/webgpu/enhance.spec.ts` is written and wired into
      `just fe-e2e-enhance`, but the development machine has no GPU device (every spec in the
      `webgpu` project skips), so it has never actually executed. Run it on a GPU box before
      trusting the WebGPU provider.
- [ ] Manual listening check not done — no audio output available here.

### Phase 6 — Docs ✅

- [x] `adding-a-model.md` §9 — "Adding a custom-ONNX task (no Transformers.js)"
- [x] `CLAUDE.md` + `.github/copilot-instructions.md` — the 48 kHz exception, the aux-file contract,
      the ORT subpath rule, and the scaling constant
- [x] `e2e-testing.md` — how the enhancement spec measures rather than looks
- [x] Marked Complete and moved to `docs/plans/completed/`

## What the DSP actually cost — read this before touching `src/audio/enhance/`

The plan's premise held: the risky part was never the graph. It was the four
constants around it, and every one of them fails **silently**.

**Guessing did not work, and could not have.** A first implementation matched the
model card and the reference source term for term, and produced audio that was
merely mediocre: on clean speech the ERB mask sat at ~0.3 and the deep-filter
coefficients at ~0.03, i.e. the network judged clean speech to be almost entirely
noise. Nothing threw. The only way through was to run the **official
implementation** (DeepFilterNet v0.5.6 + `libDF`) over the same input and diff the
intermediates stage by stage. That immediately isolated the culprit:

> **`SPEC_SCALE = 2 * hop / fft² = 1/960`.** `libDF` scales the analysis spectrum by
> its `wnorm` before computing features. The unit-norm feature divides by
> `sqrt(state)`, so it is **not level-invariant** — omit the scaling and every
> complex feature is √960 ≈ 31x too large, and the network masks the signal away.

With it in place the pipeline matched `df.enhance()` at **50–57 dB SI-SDR**
(max abs sample error ~1e-3) — the same order the model card reports for its own
reference integration (54.85 dB).

Two smaller findings:

- **`conv_lookahead` needs no handling.** The PyTorch model shifts *features*
  (`pad_feat`), not the spectrum, and the export bakes that into the graph — so
  mask/coefficient frame `t` applies to spectrum frame `t`. Scanning the shift
  empirically confirmed 0 is best; 1 and 2 cost 40+ dB of parity.
- **`PAD_FRONT` is one *hop*, not one window.** The streaming reference keeps a
  single hop of history, so frame `k` spans `[k-1, k+1) * hop`. `stft`/`istft`
  gained a `padFront` parameter for this.

**Do not "validate" this pipeline by ear or by SI-SDR against a noisy reference.**
An early check used the JFK clip as its "clean" reference and read −8.85 dB for a
clean passthrough, which looked like a catastrophic bug. It was not: the recording
genuinely is noisy, DeepFilterNet removes that noise, and the output therefore
diverges from the "reference". The reference implementation's output — which shows
the *same* −8.85 dB — is the only trustworthy oracle.

Reproducing the oracle: `pip install deepfilternet==0.5.6 torch==2.3.1
torchaudio==2.3.1` on **Python ≤3.11** (its `deepfilterlib` wheel is not built for
3.13, and building from source needs a Rust toolchain).

## Testing

Correctness here cannot rest on "it renders", because wrong DSP still produces audio.

| Layer | Test |
|---|---|
| FFT | vs. a naive O(n²) DFT on random input, to 1e-4 |
| STFT | `istft(stft(x)) ≈ x` for random and tonal input, to 1e-5 |
| Window | symmetry; power-complementary overlap-add sums to 1 |
| Aux file | parsed shapes; matrix invariants; corrupt input rejected |
| Aux file | shapes, matrix invariants (incl. contiguity), corrupt input rejected |
| Features | **byte-compared against arrays captured from `libDF`** (`__fixtures__/`) |
| Mask + deep filter | tap alignment pinned with unit impulses; noisy-copy rule pinned |
| End-to-end | **SDR improves ≥6 dB** on a real clip through the real graph (`@slow`) |
| E2E `@slow` | the real session runs on WASM; the WebGPU spec awaits a GPU machine |

## Risks

- ~~**Feature normalisation is the least-documented step.**~~ **Realised.** Matching the reference
  term by term was necessary but *not sufficient* — the missing piece was the spectrum scaling
  upstream of it, which no amount of reading the normalisation code reveals. See the section above.
- **Deep filtering is where artefacts come from.** Order-5 complex filtering with lookahead applied
  to the wrong frame produces a metallic sound rather than an error — hence the SNR acceptance test.
- **48 kHz differs from every other route.** Do not let 16 kHz assumptions leak in from the ASR code.
- ~~**`onnxruntime-web` is a new direct dependency.**~~ **Handled, with a twist.** Pinning the exact
  version `@huggingface/transformers` uses dedupes the npm package, but that is only half of it: the
  *subpath* decides which WASM binary Vite emits. Importing the bare `onnxruntime-web` entry added a
  third 26 MB `.wasm` asset; importing **`onnxruntime-web/webgpu`** — the subpath Transformers.js
  itself imports — shares one, and shrank the worker chunk from 407 KB to 120 KB. It also needed
  `optimizeDeps.include`, or Vite discovers it inside the Worker mid-session and reloads the page
  out from under a route that has just started loading.
