// Thin helpers over the GPUBuffer lifecycle: upload host data, allocate output,
// and read results back to the CPU. Keep raw buffer wrangling here so kernels
// (runtime.ts) stay readable.

/** Create a STORAGE buffer initialised with `data`. */
export function createStorageBuffer(
  device: GPUDevice,
  data: Float32Array,
  extraUsage: GPUBufferUsageFlags = 0,
): GPUBuffer {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

/** Create an empty STORAGE buffer that can be copied back to the CPU. */
export function createOutputBuffer(
  device: GPUDevice,
  byteLength: number,
): GPUBuffer {
  return device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

/** Create a small UNIFORM buffer initialised with `data`. */
export function createUniformBuffer(
  device: GPUDevice,
  data: Uint32Array | Float32Array,
): GPUBuffer {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

/**
 * Copy a GPU buffer into a mappable staging buffer and read it as Float32Array.
 * Returns a detached copy; the staging buffer is destroyed before returning.
 */
export async function readBackFloat32(
  device: GPUDevice,
  source: GPUBuffer,
  byteLength: number,
): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const copy = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return copy;
}
