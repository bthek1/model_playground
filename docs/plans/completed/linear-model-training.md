# Plan: In-Browser Linear Model Training on MNIST

**Status:** In Progress — all phases implemented; `fe-build`/`fe-lint`/`fe-test`
green (including a finite-difference gradient check and toy-problem convergence).
Remaining: manual end-to-end verification on a real GPU browser (can't run
headless) that MNIST loss falls and test accuracy climbs past ~90%.
**Date:** 2026-07-04

---

## Goal

Add a **Training** page where the user trains a **linear model** (softmax /
multinomial logistic regression, 784 → 10) on **MNIST**, entirely in the
browser, and watches the training loop live — loss and accuracy curves updating
per step/epoch. This is the first *learning* (as opposed to inference) workload
in the playground.

The heavy linear algebra (the two matmuls per step) runs on the **GPU** via the
existing `runMatmul` kernel; the pointwise pieces (softmax, cross-entropy,
gradients, SGD update) run on the CPU inside the **Web Worker** so the UI thread
never blocks. Training is streamed: the worker emits progress events the UI
renders as it goes.

## Background & constraints

- Raw WebGPU only, no ML framework (repo rule). We reuse the tested `runMatmul`
  kernel rather than hand-writing a fused backprop kernel — lower risk, and it
  keeps the GPU on the critical path.
- Backend is a registry only — **no server-side training**. Data and compute are
  100% client-side.
- MNIST is fetched by the browser from a CDN (the well-known tfjs learnjs sprite
  + label files), consistent with how model weights are fetched. The dataset is
  subsampled (configurable) to keep memory and epoch time reasonable.

## The math (one training step, batch size B)

- Forward: `logits[B,10] = Xb[B,784] · W[784,10] + b[10]` — matmul (GPU) + bias.
- `probs = softmax(logits)` per row; `loss = mean cross-entropy`.
- `dLogits[B,10] = (probs − onehot(y)) / B`.
- `dW[784,10] = Xbᵀ[784,B] · dLogits[B,10]` — matmul (GPU).
- `db[10] = colsum(dLogits)`.
- SGD: `W −= lr·dW`, `b −= lr·db`.

## Phases

### Phase 1 — Dataset loader (`src/lib/mnist.ts`)
- Fetch the MNIST sprite PNG + label binary; decode pixels via a canvas into a
  normalised `Float32Array` (values in [0,1]) and `Uint8Array` int labels.
- Pure helpers `sliceDataset` / `normalizePixels` that are unit-testable without
  the network. Loader reports progress and fails gracefully when offline.

### Phase 2 — Trainer (`src/webgpu/linearModel.ts`)
- `LinearModel` (weights + bias) and `LinearTrainer` that runs epochs of
  mini-batch SGD. The matmul dependency is **injected** (`MatmulFn`) so the math
  is unit-testable with a CPU matmul and runs on GPU in production.
- Emits per-step loss and per-epoch train/test accuracy via a callback.
- `cpuMatmul` reference lives here too (used by tests and as a fallback).

### Phase 3 — Worker plumbing (streaming)
- New `train` message: worker builds a `LinearTrainer` backed by GPU `runMatmul`,
  streams `{event:"progress", metrics}` messages, then a final result. A
  `trainCancel` message flips a flag the loop checks between steps.
- `workerClient.ts`: `trainLinearInWorker(worker, config, data, onProgress)`
  returning a promise + a `cancel()` handle.

### Phase 4 — Hook + UI
- `useLinearTraining.ts`: loads MNIST once, owns the worker, exposes
  `start/stop`, dataset status, accumulated metric history, and the latest
  weight `snapshot`.
- `routes/training.tsx`: WebGPU gate, dataset status, hyper-parameter controls
  (learning rate, batch size, epochs, train/test size), Start/Stop, live loss +
  accuracy charts (lazy `EChart`), and final test accuracy. Add a nav item.

### Phase 5 — Model visualisation (weight templates)
- The worker streams a `{event:"weights"}` snapshot (a `slice()` copy of the
  live weights + bias, transferred) at every epoch boundary; `workerClient`
  surfaces it through an `onWeights` callback and the hook stores the latest.
- `components/training/ModelWeights.tsx`: renders the 10 output classes as a
  **horizontal row of 28×28 canvas templates** — each class's 784 weights
  reshaped to the input image. A diverging colour scale encodes the signed
  weight (red = positive / evidence-for, blue = negative / evidence-against),
  with alpha ∝ |weight| so near-zero weights fade to the card background and it
  reads correctly in both themes. Includes a colour-bar legend (±max |w|) and
  per-class bias. Updates live each epoch, so the templates visibly sharpen into
  digit shapes as training converges.

## Testing

- `mnist.test.ts` — `normalizePixels` scales bytes to [0,1]; `sliceDataset`
  returns the requested train/test split with correct shapes.
- `linearModel.test.ts` — with a CPU matmul: a single step reduces loss on a
  toy separable problem; training a few epochs on a tiny synthetic set reaches
  high accuracy; a finite-difference **gradient check** on `dW`/`db` matches the
  analytic gradient within tolerance.
- `just fe-build`, `just fe-lint`, `just fe-test` all green.
- Manual: open `/training`, load MNIST, run — loss falls, test accuracy climbs
  past ~90% for a linear model. (Requires a real GPU browser; can't run headless.)

- `training.test.tsx` — the model-weights card appears once a snapshot arrives,
  rendering one labelled canvas per digit class (0–9).

## Out of scope

- Non-linear models (MLP/CNN), optimizers beyond plain SGD, data augmentation,
  saving trained weights to the registry. Hooks are left where these would slot
  in later.
