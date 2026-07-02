import { describe, expect, it } from "vitest";

import { detectWebGPU, isWebGPUSupported } from "./capabilities";

// happy-dom does not implement WebGPU, so navigator.gpu is undefined. These
// tests pin the graceful-degradation path the UI relies on.
describe("webgpu capabilities", () => {
  it("reports WebGPU as unsupported when navigator.gpu is missing", () => {
    expect(isWebGPUSupported()).toBe(false);
  });

  it("returns an unsupported report without throwing", async () => {
    const caps = await detectWebGPU();
    expect(caps.status).toBe("unsupported");
    expect(caps.adapter).toBeNull();
    expect(caps.features).toEqual([]);
    expect(caps.limits).toEqual({});
  });
});
