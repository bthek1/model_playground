// The GPU side of the GNN: the aggregation kernel's device buffers and dispatch,
// plus a matmul that reuses the playground's existing one.
//
// The organising idea is that **the graph is uploaded once**. `rowPtr`, `colIdx`
// and the two scale vectors do not change for the lifetime of a training run —
// only the features and the weights move per epoch — so they are written to
// storage buffers at construction and left on the device. Re-uploading 10 556
// neighbour indices every epoch would cost more than the aggregation does.
//
// Everything here allocates through buffers.ts and frees with `releaseBuffer`,
// so the ledger behind the system panel's GPU-memory reading stays honest.

import { releaseBuffer } from "./allocations";
import {
  createOutputBuffer,
  createStorageBuffer,
  createUniformBuffer,
  readBackFloat32,
} from "./buffers";
import { getGPUDevice } from "./device";
import type { AggregateFn } from "./gnn";
import { createComputePipeline } from "./pipeline";
import { runMatmul } from "./runtime";
import type { MatmulFn } from "./linearModel";
import aggregateShader from "./shaders/gnn_aggregate.wgsl?raw";

/** Matches @workgroup_size(64) in gnn_aggregate.wgsl. */
const WORKGROUP = 64;

/**
 * A graph resident on the GPU, and the kernel that gathers over it.
 *
 * Create one per training run and `dispose()` it when the run ends; the caller
 * owns that lifetime because the whole point is that it outlives an epoch.
 */
export class GraphAggregator {
  private readonly outputs = new Map<number, GPUBuffer>();
  private readonly owned: GPUBuffer[] = [];
  private disposed = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly pipeline: GPUComputePipeline,
    private readonly nNodes: number,
    private readonly graphBuffers: {
      rowPtr: GPUBuffer;
      colIdx: GPUBuffer;
      alpha: GPUBuffer;
      beta: GPUBuffer;
    },
  ) {
    this.owned.push(...Object.values(graphBuffers));
  }

  static async create(
    rowPtr: Uint32Array,
    colIdx: Uint32Array,
    nNodes: number,
    alpha: Float32Array,
    beta: Float32Array,
  ): Promise<GraphAggregator> {
    if (rowPtr.length !== nNodes + 1) {
      throw new Error(`rowPtr must have nNodes+1 = ${nNodes + 1} entries`);
    }
    if (rowPtr[nNodes] !== colIdx.length) {
      throw new Error("rowPtr does not end at the edge count");
    }
    if (alpha.length !== nNodes || beta.length !== nNodes) {
      throw new Error("a scale vector must have one entry per node");
    }

    const device = await getGPUDevice();
    const pipeline = createComputePipeline(device, aggregateShader);
    return new GraphAggregator(device, pipeline, nNodes, {
      // Uploaded as u32, matching the shader's `array<u32>` bindings.
      rowPtr: createStorageBuffer(device, rowPtr),
      colIdx: createStorageBuffer(device, colIdx),
      alpha: createStorageBuffer(device, alpha),
      beta: createStorageBuffer(device, beta),
    });
  }

  /** An `AggregateFn` bound to this graph, for handing to `GnnTrainer`. */
  get aggregate(): AggregateFn {
    return (x, nFeat, transposed) => this.run(x, nFeat, transposed);
  }

  async run(
    x: Float32Array,
    nFeat: number,
    transposed: boolean,
  ): Promise<Float32Array> {
    if (this.disposed) throw new Error("this GraphAggregator has been disposed");
    if (x.length !== this.nNodes * nFeat) {
      throw new Error(
        `x must have nNodes*nFeat = ${this.nNodes * nFeat} elements, got ${x.length}`,
      );
    }
    const { device, graphBuffers: g } = this;
    const byteLength = x.byteLength;

    const dims = createUniformBuffer(
      device,
      new Uint32Array([this.nNodes, nFeat]),
    );
    const input = createStorageBuffer(device, x);
    const output = this.outputFor(byteLength);

    // Âᵀ is Â with the scales swapped, which is a different bind group and not a
    // different kernel — see the shader header.
    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dims } },
        { binding: 1, resource: { buffer: g.rowPtr } },
        { binding: 2, resource: { buffer: g.colIdx } },
        { binding: 3, resource: { buffer: transposed ? g.beta : g.alpha } },
        { binding: 4, resource: { buffer: transposed ? g.alpha : g.beta } },
        { binding: 5, resource: { buffer: input } },
        { binding: 6, resource: { buffer: output } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.nNodes / WORKGROUP));
    pass.end();
    device.queue.submit([encoder.finish()]);

    const result = await readBackFloat32(device, output, byteLength);
    releaseBuffer(dims);
    releaseBuffer(input);
    return result;
  }

  /**
   * Output buffers are cached by size. A run only ever asks for two — `nNodes ×
   * hidden` and `nNodes × nClasses` — so this is two allocations for the whole
   * training run instead of two per layer per epoch.
   */
  private outputFor(byteLength: number): GPUBuffer {
    let buffer = this.outputs.get(byteLength);
    if (!buffer) {
      buffer = createOutputBuffer(this.device, byteLength);
      this.outputs.set(byteLength, buffer);
      this.owned.push(buffer);
    }
    return buffer;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const buffer of this.owned) releaseBuffer(buffer);
    this.owned.length = 0;
    this.outputs.clear();
  }
}

/** The GPU matmul, in the shape `GnnOps` wants. */
export const gpuMatmul: MatmulFn = async (a, b, m, k, n) =>
  (await runMatmul({ a, b, m, k, n })).data;
