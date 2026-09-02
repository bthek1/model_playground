import type { Page } from "@playwright/test";

/** Mirrors WebGPUStatus in src/webgpu/types.ts. */
export type WebGPUStatus = "unsupported" | "no-adapter" | "no-device" | "ready";

/**
 * Probe the real browser for WebGPU, using the same three gates as
 * src/webgpu/capabilities.ts: `navigator.gpu` present → an adapter is offered →
 * a GPUDevice can actually be acquired. Never throws.
 *
 * Kept as an in-page probe rather than importing detectWebGPU() because
 * page.evaluate ships a plain function string to the browser — it can't reach
 * the app's module graph.
 */
export async function probeWebGPU(page: Page): Promise<WebGPUStatus> {
  return page.evaluate(async (): Promise<WebGPUStatus> => {
    const gpu = (navigator as Navigator & { gpu?: unknown }).gpu;
    if (!gpu) return "unsupported";
    const adapter = await navigator.gpu.requestAdapter().catch(() => null);
    if (!adapter) return "no-adapter";
    try {
      const device = await adapter.requestDevice();
      device.destroy();
    } catch {
      return "no-device";
    }
    return "ready";
  });
}

/**
 * Hide WebGPU from the page before any app code runs, so the graceful-
 * degradation path can be asserted even on a machine that has a working GPU.
 */
export async function disableWebGPU(page: Page): Promise<void> {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "gpu", {
      value: undefined,
      configurable: true,
    });
  });
}
