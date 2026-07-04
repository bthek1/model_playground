// Dispatch table for basic matrix/tensor arithmetic. Element-wise ops share one
// kernel (elementwise.wgsl); matmul reuses the reference runtime; transpose and
// scale each have a tiny dedicated kernel. Every op validates shapes on the CPU
// before touching the GPU so the UI gets a clear error rather than a driver one.

import {
  createOutputBuffer,
  createStorageBuffer,
  createUniformBuffer,
  readBackFloat32,
} from "./buffers";
import { getGPUDevice } from "./device";
import { createComputePipeline } from "./pipeline";
import { runMatmul } from "./runtime";
import elementwiseShader from "./shaders/elementwise.wgsl?raw";
import scaleShader from "./shaders/scale.wgsl?raw";
import transposeShader from "./shaders/transpose.wgsl?raw";
import type { TensorOp, TensorOpJob, TensorOpResult } from "./types";

// Matches the switch in elementwise.wgsl.
const OP_CODE: Record<"add" | "sub" | "mul" | "div", number> = {
  add: 0,
  sub: 1,
  mul: 2,
  div: 3,
};

const OP_SYMBOL: Record<TensorOp, string> = {
  add: "+",
  sub: "−",
  mul: "⊙",
  div: "÷",
  matmul: "·",
  transpose: "ᵀ",
  scale: "×",
};

/** Human-readable operator symbol, for labelling results in the UI. */
export function tensorOpSymbol(op: TensorOp): string {
  return OP_SYMBOL[op];
}

/** True for operations that require a second operand B. */
export function isBinaryOp(op: TensorOp): boolean {
  return op === "add" || op === "sub" || op === "mul" || op === "div" || op === "matmul";
}

/**
 * Run a single tensor-arithmetic job on the shared GPU device and read the
 * result back to the CPU. Throws (before any GPU work) if the operand shapes
 * are incompatible with the requested op.
 */
export async function runTensorOp(
  job: TensorOpJob,
  now: () => number = () => performance.now(),
): Promise<TensorOpResult> {
  switch (job.op) {
    case "add":
    case "sub":
    case "mul":
    case "div":
      return runElementwise(job, now);
    case "matmul":
      return runMatmulOp(job);
    case "transpose":
      return runTranspose(job, now);
    case "scale":
      return runScale(job, now);
    default:
      throw new Error(`Unsupported op: ${job.op as string}`);
  }
}

async function runElementwise(
  job: TensorOpJob,
  now: () => number,
): Promise<TensorOpResult> {
  const { a, b, aRows, aCols, bRows, bCols } = job;
  if (!b) throw new Error("This operation needs a second matrix B.");
  if (aRows !== bRows || aCols !== bCols) {
    throw new Error(
      `Shapes must match for element-wise ops: A is ${aRows}×${aCols}, B is ${bRows}×${bCols}.`,
    );
  }

  const len = aRows * aCols;
  const device = await getGPUDevice();
  const pipeline = createComputePipeline(device, elementwiseShader);

  const paramsBuffer = createUniformBuffer(
    device,
    new Uint32Array([OP_CODE[job.op as "add" | "sub" | "mul" | "div"], len]),
  );
  const aBuffer = createStorageBuffer(device, a);
  const bBuffer = createStorageBuffer(device, b);
  const outBuffer = createOutputBuffer(device, len * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: aBuffer } },
      { binding: 2, resource: { buffer: bBuffer } },
      { binding: 3, resource: { buffer: outBuffer } },
    ],
  });

  const start = now();
  dispatch1D(device, pipeline, bindGroup, len);
  await device.queue.onSubmittedWorkDone();
  const data = await readBackFloat32(device, outBuffer, len * 4);
  const gpuTimeMs = now() - start;

  destroy(paramsBuffer, aBuffer, bBuffer, outBuffer);
  return { data, rows: aRows, cols: aCols, gpuTimeMs };
}

async function runMatmulOp(job: TensorOpJob): Promise<TensorOpResult> {
  const { a, b, aRows, aCols, bRows, bCols } = job;
  if (!b) throw new Error("Matrix multiply needs a second matrix B.");
  if (aCols !== bRows) {
    throw new Error(
      `Inner dimensions must match for A·B: A is ${aRows}×${aCols}, B is ${bRows}×${bCols} (${aCols} ≠ ${bRows}).`,
    );
  }
  const result = await runMatmul({
    a,
    b,
    m: aRows,
    k: aCols,
    n: bCols as number,
  });
  return {
    data: result.data,
    rows: aRows,
    cols: bCols as number,
    gpuTimeMs: result.gpuTimeMs,
  };
}

async function runTranspose(
  job: TensorOpJob,
  now: () => number,
): Promise<TensorOpResult> {
  const { a, aRows, aCols } = job;
  const len = aRows * aCols;
  const device = await getGPUDevice();
  const pipeline = createComputePipeline(device, transposeShader);

  const dimsBuffer = createUniformBuffer(device, new Uint32Array([aRows, aCols]));
  const aBuffer = createStorageBuffer(device, a);
  const outBuffer = createOutputBuffer(device, len * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: dimsBuffer } },
      { binding: 1, resource: { buffer: aBuffer } },
      { binding: 2, resource: { buffer: outBuffer } },
    ],
  });

  const start = now();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(aCols / 16), Math.ceil(aRows / 16));
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  const data = await readBackFloat32(device, outBuffer, len * 4);
  const gpuTimeMs = now() - start;

  destroy(dimsBuffer, aBuffer, outBuffer);
  // Transpose flips the shape.
  return { data, rows: aCols, cols: aRows, gpuTimeMs };
}

async function runScale(
  job: TensorOpJob,
  now: () => number,
): Promise<TensorOpResult> {
  const { a, aRows, aCols } = job;
  const scalar = job.scalar ?? 0;
  const len = aRows * aCols;
  const device = await getGPUDevice();
  const pipeline = createComputePipeline(device, scaleShader);

  // Pack { len: u32, scalar: f32 } into one 8-byte uniform via two views.
  const paramsBuf = new ArrayBuffer(8);
  new Uint32Array(paramsBuf)[0] = len;
  new Float32Array(paramsBuf)[1] = scalar;
  const paramsBuffer = createUniformBuffer(device, new Uint32Array(paramsBuf));

  const aBuffer = createStorageBuffer(device, a);
  const outBuffer = createOutputBuffer(device, len * 4);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: aBuffer } },
      { binding: 2, resource: { buffer: outBuffer } },
    ],
  });

  const start = now();
  dispatch1D(device, pipeline, bindGroup, len);
  await device.queue.onSubmittedWorkDone();
  const data = await readBackFloat32(device, outBuffer, len * 4);
  const gpuTimeMs = now() - start;

  destroy(paramsBuffer, aBuffer, outBuffer);
  return { data, rows: aRows, cols: aCols, gpuTimeMs };
}

// Encode + submit a 1-D dispatch over `len` elements at @workgroup_size(64).
function dispatch1D(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  len: number,
): void {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(len / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
}

function destroy(...buffers: GPUBuffer[]): void {
  for (const buffer of buffers) buffer.destroy();
}
