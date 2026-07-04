# In-Browser Inference with Raw WebGPU

This project runs models **on the user's GPU, inside the browser**, using the
WebGPU API directly — no Transformers.js, ONNX Runtime, or WebLLM. Models are
expressed as WGSL compute shaders. This document explains the pipeline; to add a
model see [`../guides/adding-a-model.md`](../guides/adding-a-model.md).

## Why raw WebGPU

- **Control.** Every kernel, buffer, and dispatch is ours — no framework
  abstractions between the code and the device.
- **Bundle size.** No multi-megabyte runtime shipped to the client.
- **Learning surface.** The playground is a place to understand GPU compute from
  the metal up.

The trade-off: you write kernels yourself. Start from the reference kernels in
`src/webgpu/shaders/` and specialise.

## The module map (`frontend/src/webgpu/`)

| File | Responsibility |
|------|----------------|
| `capabilities.ts` | `detectWebGPU()` — probe adapter, features, limits. Never throws. |
| `device.ts` | `getGPUDevice()` — memoised `GPUDevice`; re-acquires after device-lost. |
| `buffers.ts` | Create storage/uniform buffers, upload data, read results back. |
| `pipeline.ts` | Compile a WGSL string into a `GPUComputePipeline`. |
| `runtime.ts` | `runMatmul()` — the reference end-to-end kernel + benchmark. |
| `shaders/*.wgsl` | The compute kernels (imported as strings via Vite `?raw`). |
| `worker.ts` | Web Worker that owns the device and runs jobs off the main thread. |
| `workerClient.ts` | Main-thread promise API over the worker (request correlation). |

## The compute pipeline

Every kernel follows the same six steps (see `runtime.ts::runMatmul` for the
canonical implementation):

1. **Acquire the device** — `getGPUDevice()` requests an adapter and device once
   and caches the promise. If the device is lost (driver reset, tab
   backgrounded), the cache clears so the next call re-acquires.
2. **Compile the shader** — `createComputePipeline(device, wgsl)` builds a
   pipeline with `layout: "auto"`, deriving bind-group layouts from the WGSL.
3. **Upload inputs** — `createStorageBuffer` writes host `Float32Array`s into
   `STORAGE | COPY_DST` buffers; `createUniformBuffer` carries small params
   (e.g. matrix dimensions).
4. **Bind** — a bind group maps buffers to the `@binding(n)` slots the shader
   declares.
5. **Dispatch** — `dispatchWorkgroups(x, y, z)` launches a grid of workgroups.
   The grid size is `ceil(dimension / workgroup_size)` per axis; the shader
   guards against out-of-range invocations.
6. **Read back** — `readBackFloat32` copies the output buffer into a
   `MAP_READ` staging buffer and returns a detached `Float32Array`.

## Threading model

Heavy compute runs in a **dedicated Web Worker** (`worker.ts`) so the UI thread
never blocks. The worker owns its own `GPUDevice`. Communication:

```
main thread                          worker
  workerClient.call({type,id})  ──►  onmessage → runtime fn
  (input buffers transferred)        (result buffer transferred back)
  Promise resolves              ◄──  postMessage({id, ok, result})
```

Input/output `ArrayBuffer`s are **transferred**, not copied, so large tensors
move between threads with zero-copy. `workerClient.ts` correlates responses to
requests by an incrementing `id`.

> WebGPU is available in workers, but `navigator.gpu` is undefined in Node/test
> environments (happy-dom). `detectWebGPU()` returns `status: "unsupported"`
> there rather than throwing — the UI degrades gracefully.

## Browser support & requirements

`detectWebGPU()` reports one of four statuses, which the GPU Capabilities panel
turns into a plain-language message:

| Status | Meaning | Typical cause / fix |
|--------|---------|---------------------|
| `ready` | A `GPUDevice` was actually acquired — compute will run. | — |
| `no-device` | Adapter exists but `requestDevice()` failed. | Driver blocked/out of resources. |
| `no-adapter` | `navigator.gpu` exists but offered no adapter. | No compatible GPU/driver in the environment. |
| `unsupported` | `navigator.gpu` is absent entirely. | See the two gotchas below. |

Note that `ready` means a device was **acquired**, not merely that an adapter was
listed — `detectWebGPU()` calls `requestDevice()` (and discards the probe device)
so the panel never claims access it can't back up.

Two things commonly produce a false `unsupported` on a perfectly capable machine:

- **Secure context required.** Browsers only expose `navigator.gpu` on HTTPS or
  `http://localhost`/`127.0.0.1`. A plain-HTTP LAN origin (e.g.
  `http://192.168.x.x:5174`) hides the API. The Vite dev server therefore runs
  over HTTPS (`@vitejs/plugin-basic-ssl`) — see
  [`local-setup.md`](../guides/local-setup.md). The panel detects
  `window.isSecureContext === false` and says so.
- **Firefox on Linux/macOS needs a flag.** Firefox enabled WebGPU by default on
  Windows first; on Linux and macOS (even Firefox 152) it stays behind
  `dom.webgpu.enabled` in `about:config` — set it to `true` and restart. The
  panel special-cases Firefox and points users at the flag. (Firefox also blanks
  `adapter.info.vendor`/`architecture` for anti-fingerprinting, so those show as
  "unknown" — expected, not a bug.)

Baseline targets: Chrome/Edge 113+, Firefox 141+ (Windows) / flag on Linux+macOS,
Safari 26+.

## Benchmarking

`runMatmul` times encode → submit → `onSubmittedWorkDone()` → readback and
reports `gflops = 2·M·K·N / seconds`. The playground's "Compute Benchmark" card
runs a 512×512 matmul and cross-checks one output entry against a CPU reference
to catch a broken kernel or driver. This same number is what you'd `POST` to
`/api/registry/runs/` as run metadata.

## Precision & limits

- Buffers here are `f32`. `shader-f16` (when advertised in `capabilities.features`)
  halves memory and can speed up compute — opt in per kernel.
- `maxStorageBufferBindingSize` and `maxBufferSize` (shown in the capabilities
  panel) bound the largest tensor a device will accept. Large models must tile
  their weights across multiple dispatches.

## Related

- API for the catalog / run metadata: [`../standards/api-contracts.md`](../standards/api-contracts.md)
- System architecture: [`architecture.md`](architecture.md)
- Roadmap: [`../plans/webgpu-model-playground.md`](../plans/webgpu-model-playground.md)
