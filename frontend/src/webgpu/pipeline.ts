/**
 * Compile a WGSL string into a compute pipeline with an auto-derived layout.
 * `entryPoint` must match the `@compute fn` name in the shader.
 */
export function createComputePipeline(
  device: GPUDevice,
  code: string,
  entryPoint = "main",
): GPUComputePipeline {
  const module = device.createShaderModule({ code });
  return device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint },
  });
}
