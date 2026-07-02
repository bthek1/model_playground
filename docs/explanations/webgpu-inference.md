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
