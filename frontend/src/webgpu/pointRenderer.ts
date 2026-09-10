// The point-cloud renderer: the raw-WebGPU half of `/image-to-3d`.
//
// This is the first place in the app where the **render** pipeline is used at
// all — `runtime.ts` and `tensorops.ts` are compute only — and it is also the
// one page where the hand-written WGSL runtime and the Transformers.js runtime
// appear together. They do not mix: inference stays in `src/vision/`, produces a
// plain `Float32Array`, and this file never knows a model exists. The route is
// where they meet, which is the arrangement
// docs/explanations/webgpu-inference.md describes.
//
// Two rules the class below exists to keep:
//
//  1. **The vertex buffer is written once per inference, not per frame.** A
//     cloud is megabytes; re-uploading it on every orbit tick would spend the
//     entire frame budget on a bus transfer of data that did not change. Only
//     the 48-byte camera uniform is rewritten while dragging.
//  2. **Nothing here throws on a machine with no GPU.** `create()` returns null
//     rather than raising, so the route can render the depth map and an
//     explanation instead of an empty canvas.

import { getGPUDevice } from "./device";
import shaderCode from "./shaders/points.wgsl?raw";

/** Where the orbit camera is, in the terms the shader wants. */
export interface CameraState {
  yaw: number;
  pitch: number;
  distance: number;
  /** Half-size of a point, in device pixels. */
  pointSize: number;
}

export const DEFAULT_CAMERA: CameraState = {
  yaw: 0,
  pitch: 0,
  distance: 3,
  pointSize: 1.5,
};

/** Floats in the camera uniform block. See the `Camera` struct in the WGSL. */
const UNIFORM_FLOATS = 12;

/** Floats per point in the vertex buffer: `[x, y, z, r, g, b]`. */
const POINT_STRIDE = 6;

export interface CloudUpload {
  data: Float32Array;
  count: number;
  center: [number, number, number];
}

export class PointRenderer {
  private vertexBuffer: GPUBuffer | null = null;
  private count = 0;
  private depthTexture: GPUTexture | null = null;
  private depthSize = { width: 0, height: 0 };
  private center: [number, number, number] = [0, 0, 0];
  private readonly uniformData = new Float32Array(UNIFORM_FLOATS);

  private constructor(
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    private readonly pipeline: GPURenderPipeline,
    private readonly uniformBuffer: GPUBuffer,
    private readonly bindGroup: GPUBindGroup,
    private readonly format: GPUTextureFormat,
  ) {}

  /**
   * Build a renderer for a canvas, or return null when WebGPU is unavailable.
   *
   * Null rather than a throw: `detectWebGPU()` never throws either, and a 3-D
   * view that takes the page down on a machine without a GPU is worse than one
   * that says so.
   */
  static async create(canvas: HTMLCanvasElement): Promise<PointRenderer | null> {
    let device: GPUDevice;
    try {
      device = await getGPUDevice();
    } catch {
      return null;
    }

    const context = canvas.getContext("webgpu");
    if (!context) return null;

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });

    const module = device.createShaderModule({ code: shaderCode });
    const pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: POINT_STRIDE * 4,
            // One point per *instance*: the six vertices of its quad are
            // generated in the shader from `vertex_index`.
            stepMode: "instance",
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" },
              { shaderLocation: 1, offset: 12, format: "float32x3" },
            ],
          },
        ],
      },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
      // Depth testing is not optional for a point cloud: without it the points
      // are drawn in buffer order, so the back of the scene paints over the
      // front and the result reads as fog rather than as geometry.
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    const uniformBuffer = device.createBuffer({
      size: UNIFORM_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });

    return new PointRenderer(
      device,
      context,
      pipeline,
      uniformBuffer,
      bindGroup,
      format,
    );
  }

  /** Upload a cloud. Called once per inference — never from the render loop. */
  upload({ data, count, center }: CloudUpload): void {
    this.vertexBuffer?.destroy();
    this.count = count;
    this.center = center;

    if (count === 0) {
      this.vertexBuffer = null;
      return;
    }

    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    this.vertexBuffer = buffer;
  }

  /** Draw one frame at the current canvas size. Cheap enough for a drag. */
  render(canvas: HTMLCanvasElement, camera: CameraState): void {
    const width = Math.max(1, canvas.width);
    const height = Math.max(1, canvas.height);

    if (
      !this.depthTexture ||
      this.depthSize.width !== width ||
      this.depthSize.height !== height
    ) {
      this.depthTexture?.destroy();
      this.depthTexture = this.device.createTexture({
        size: { width, height },
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.depthSize = { width, height };
    }

    this.uniformData[0] = camera.yaw;
    this.uniformData[1] = camera.pitch;
    this.uniformData[2] = camera.distance;
    this.uniformData[3] = width / height;
    this.uniformData[4] = this.center[0];
    this.uniformData[5] = this.center[1];
    this.uniformData[6] = this.center[2];
    // [7] is the struct's padding after a vec3.
    // Point size arrives in device pixels and leaves in clip units — clip space
    // spans 2 across the canvas, hence the factor of 2.
    this.uniformData[8] = (camera.pointSize * 2) / width;
    this.uniformData[9] = (camera.pointSize * 2) / height;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.04, g: 0.05, b: 0.08, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    if (this.vertexBuffer && this.count > 0) {
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.setVertexBuffer(0, this.vertexBuffer);
      // Six vertices (two triangles) per point, one instance per point.
      pass.draw(6, this.count);
    }

    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Release GPU memory. A cloud is megabytes — leaking one per visit adds up. */
  destroy(): void {
    this.vertexBuffer?.destroy();
    this.vertexBuffer = null;
    this.depthTexture?.destroy();
    this.depthTexture = null;
    this.uniformBuffer.destroy();
  }

  /** The canvas format in use, exposed for tests and diagnostics. */
  get canvasFormat(): GPUTextureFormat {
    return this.format;
  }
}
