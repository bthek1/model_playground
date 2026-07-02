# Plan: WebGPU Model Playground

**Status:** In Progress
**Date:** 2026-07-02

---

## Goal

Turn the Django + React template into a **model playground**: a web app that runs
LLMs, computer-vision, and custom models **in the browser on the user's GPU via
raw WebGPU** (WGSL compute shaders — no ML framework). The backend becomes a thin
**model registry** that stores the catalog and inference-run metadata; it never
runs inference.

## Background

Modern browsers expose the GPU through WebGPU. Running inference client-side means
zero server GPU cost, data never leaves the device, and instant scaling. Choosing
raw WebGPU over Transformers.js/ONNX/WebLLM maximises control and keeps the bundle
small, at the cost of writing kernels by hand. This plan scaffolds the foundation
and leaves model-specific kernels as follow-on work.

## Phases

### Phase 1 — Foundation (scaffold) ✅

- [x] Backend `registry` app: `ModelCard` + `InferenceRun` models (UUID PKs), migration
- [x] Registry API: `ModelCardViewSet` + `InferenceRunViewSet` under `/api/registry/`
- [x] Registry admin, serializers, `services.record_inference_run()`, tests
- [x] Frontend `src/webgpu/` runtime: capabilities, device, buffers, pipeline, runtime
- [x] Reference WGSL kernels: `matmul.wgsl`, `relu.wgsl`
- [x] Web Worker (`worker.ts`) + main-thread client (`workerClient.ts`)
- [x] Hooks: `useWebGPU`, `useGpuBenchmark`, `useModels`
- [x] `/playground` route: capability panel + GPU benchmark + model catalog
- [x] Docs: architecture, `webgpu-inference.md`, `adding-a-model.md`, API contracts

### Phase 2 — Custom model runner

- [ ] Weight loader: fetch + parse `weights_url` (safetensors) into GPU buffers
- [ ] Layer primitives as kernels: dense (matmul+bias), conv2d, layernorm, softmax
- [ ] A small end-to-end custom net (e.g. MNIST-class MLP/CNN) with a demo UI
- [ ] Persist run metrics via `POST /api/registry/runs/`

### Phase 3 — Computer vision

- [ ] Image preprocessing kernels (resize/normalize) + webcam/file input
- [ ] Image-classification demo model + overlay of results
- [ ] `f16` path where `shader-f16` is available

### Phase 4 — LLM

- [ ] Tokenizer (client-side) + KV-cache buffers
- [ ] Quantized matmul kernels (int8/int4) for weight-heavy layers
- [ ] Streaming token generation UI (render via `Markdown.tsx`)

### Phase 5 — Registry UX

- [ ] Model detail route, filtering by `task`, per-model run history
- [ ] Optional server-side fallback path (out of current scope)

## Testing

- **Unit (backend):** registry permissions (public vs private, auth-gated create),
  `record_inference_run`, run isolation per user. *(Phase 1 done — 8 tests.)*
- **Unit (frontend):** `capabilities.detectWebGPU()` unsupported path; per-kernel
  CPU cross-checks; `workerClient` request/response correlation.
- **Integration:** `/playground` renders capability + catalog with MSW-mocked
  registry; benchmark path exercised in a WebGPU-capable browser (manual).
- **Manual:** open `/playground` in Chrome/Edge, confirm adapter detected,
  benchmark reports GFLOP/s and "verified vs CPU", catalog lists seeded models.

## Risks & Notes

- **Browser support.** WebGPU requires a recent Chromium/Firefox/Safari; the UI
  degrades to an "unavailable" state elsewhere. A server-side fallback is
  explicitly out of scope (see Phase 5).
- **Kernel correctness.** Hand-written WGSL can silently produce wrong results;
  every kernel needs a CPU cross-check in dev.
- **Memory limits.** Large models exceed `maxStorageBufferBindingSize`; weights
  must be tiled. Track device limits from the capabilities probe.
- **Reference matmul is naive** (no tiling/shared memory) — correct but not fast;
  optimise per model in later phases.
