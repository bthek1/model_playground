# Guide: Adding a Model

A model in this project is two things: a **WGSL kernel** (how it computes, in the
browser) and a **registry entry** (its metadata, in the backend). This guide
walks through both. Background: [`../explanations/webgpu-inference.md`](../explanations/webgpu-inference.md).

## 1. Write the kernel (frontend)

Add a WGSL compute shader under `frontend/src/webgpu/shaders/`. Follow the
pattern in `matmul.wgsl`:

- Declare inputs/outputs as `@group(0) @binding(n)` storage buffers and small
  params as a `uniform`.
- Pick a `@workgroup_size(...)` and **guard against out-of-range invocations**
  with an early `return`.

```wgsl
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&input)) { return; }
  output[i] = input[i] * 2.0; // your op here
}
```

## 2. Wire it into the runtime (frontend)

Import the shader as a string and drive it with the existing helpers. A new
kernel typically adds a function to `runtime.ts`:

```ts
import myShader from "./shaders/my_kernel.wgsl?raw";
import { createStorageBuffer, createOutputBuffer, readBackFloat32 } from "./buffers";
import { createComputePipeline } from "./pipeline";
import { getGPUDevice } from "./device";

export async function runMyKernel(input: Float32Array): Promise<Float32Array> {
  const device = await getGPUDevice();
  const pipeline = createComputePipeline(device, myShader);

  const inBuf = createStorageBuffer(device, input);
  const outBuf = createOutputBuffer(device, input.byteLength);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inBuf } },
      { binding: 1, resource: { buffer: outBuf } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(input.length / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);

  const result = await readBackFloat32(device, outBuf, input.byteLength);
  [inBuf, outBuf].forEach((b) => b.destroy());
  return result;
}
```

To run it **off the main thread**, add a message type to `worker.ts` and a
matching helper in `workerClient.ts` (mirror `matmul` / `runMatmulInWorker`).

> Tip: cross-check a few outputs against a CPU reference during development, like
> `useGpuBenchmark` does — a wrong kernel often still runs and returns garbage.

## 3. Register the model (backend)

Add a catalog entry so the model appears in the playground. Either via the Django
admin (`/admin/` → Model cards) or the API:

```bash
curl -X POST http://localhost:8000/api/registry/models/ \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "slug": "my-net",
    "name": "My Net",
    "task": "vision",
    "description": "What it does.",
    "weights_url": "https://cdn.example.com/my-net.safetensors",
    "config": { "entry_point": "main", "input_shape": [1, 3, 224, 224] },
    "license": "MIT",
    "is_public": true
  }'
```

`config` is free-form JSON — put whatever the browser runtime needs to build the
pipeline and preprocess inputs (shapes, quantization, WGSL entry point, tokenizer
refs). Weights are fetched by the **browser** from `weights_url`; they are not
served by Django.

Field reference: [`../standards/api-contracts.md`](../standards/api-contracts.md).

## 4. (Optional) Report run metrics

After a run, `POST /api/registry/runs/` with `params` and `metrics`
(`latency_ms`, `tokens_per_sec`, `gflops`, adapter info) so the playground can
show history and benchmarks.

## 5. Verify

- `just fe-lint && just fe-build` — kernel imports and types compile.
- `just be-test` — registry endpoints still pass.
- Open `/playground`, confirm the model appears in the catalog and your kernel
  runs on the GPU.
