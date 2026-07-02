import {
  createOutputBuffer,
  createStorageBuffer,
  createUniformBuffer,
  readBackFloat32,
} from "./buffers";
import { getGPUDevice } from "./device";
import { createComputePipeline } from "./pipeline";
import matmulShader from "./shaders/matmul.wgsl?raw";
import type { MatmulJob, MatmulResult } from "./types";

// Matches @workgroup_size(16, 16, 1) in matmul.wgsl.
const TILE = 16;

/**
 * Run the matmul kernel on the shared GPU device. This is the reference
 * end-to-end path — upload inputs, build a bind group, dispatch, read back —
 * and doubles as a WebGPU compute benchmark (see `gflops`).
 *
 * `now` is injectable for deterministic testing.
 */
export async function runMatmul(
  job: MatmulJob,
  now: () => number = () => performance.now(),
): Promise<MatmulResult> {
  const { a, b, m, k, n } = job;
  if (a.length !== m * k) {
    throw new Error(`A must have m*k = ${m * k} elements, got ${a.length}`);
  }
  if (b.length !== k * n) {
    throw new Error(`B must have k*n = ${k * n} elements, got ${b.length}`);
  }

  const device = await getGPUDevice();
  const pipeline = createComputePipeline(device, matmulShader);

  const dimsBuffer = createUniformBuffer(device, new Uint32Array([m, k, n]));
  const aBuffer = createStorageBuffer(device, a);
  const bBuffer = createStorageBuffer(device, b);
  const cBuffer = createOutputBuffer(device, m * n * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: dimsBuffer } },
      { binding: 1, resource: { buffer: aBuffer } },
      { binding: 2, resource: { buffer: bBuffer } },
      { binding: 3, resource: { buffer: cBuffer } },
    ],
  });

  const start = now();

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(n / TILE), Math.ceil(m / TILE));
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  const data = await readBackFloat32(device, cBuffer, m * n * 4);
  const gpuTimeMs = now() - start;

  for (const buffer of [dimsBuffer, aBuffer, bBuffer, cBuffer]) {
    buffer.destroy();
  }

  const flops = 2 * m * k * n;
  const gflops = gpuTimeMs > 0 ? flops / (gpuTimeMs / 1000) / 1e9 : 0;
  return { data, gpuTimeMs, gflops };
}
