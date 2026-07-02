import { isWebGPUSupported } from "./capabilities";

export class WebGPUUnavailableError extends Error {
  constructor(message = "WebGPU is not available in this environment") {
    super(message);
    this.name = "WebGPUUnavailableError";
  }
}

let devicePromise: Promise<GPUDevice> | null = null;

/**
 * Acquire a shared GPUDevice (adapter + device), memoised for the lifetime of
 * the context (main thread or worker). If the device is lost — driver reset,
 * tab backgrounded, etc. — the cache is cleared so the next call re-acquires.
 */
export async function getGPUDevice(): Promise<GPUDevice> {
  if (!isWebGPUSupported()) throw new WebGPUUnavailableError();

  if (!devicePromise) {
    devicePromise = (async () => {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new WebGPUUnavailableError("No WebGPU adapter available");
      }
      const device = await adapter.requestDevice();
      // Allow re-acquisition after a device-lost event.
      void device.lost.then(() => {
        devicePromise = null;
      });
      return device;
    })();
  }

  return devicePromise;
}

/** Drop the cached device (e.g. from tests or an explicit teardown). */
export function resetGPUDevice(): void {
  devicePromise = null;
}
