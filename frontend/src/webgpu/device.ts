import { isWebGPUSupported } from "./capabilities";

export class WebGPUUnavailableError extends Error {
  constructor(message = "WebGPU is not available in this environment") {
    super(message);
    this.name = "WebGPUUnavailableError";
  }
}

/**
 * Features we use if offered and live without otherwise. Only `timestamp-query`
 * so far: it buys GPU-side pass timing (see timing.ts) for the system panel,
 * and its absence costs exactly that one number.
 */
const OPTIONAL_FEATURES: GPUFeatureName[] = ["timestamp-query"];

/**
 * Request a device with whichever optional features the adapter advertises,
 * falling back to a bare device if that is refused.
 *
 * Asking for a feature the adapter lacks rejects `requestDevice` outright,
 * hence the filter. The fallback covers the case the filter cannot: an adapter
 * that advertises a feature its driver then declines to grant. Losing the
 * device would break every kernel in the app to save one diagnostic number.
 */
async function requestDeviceWith(
  adapter: GPUAdapter,
  optional: GPUFeatureName[],
): Promise<GPUDevice> {
  const requiredFeatures = optional.filter((name) =>
    Boolean(adapter.features?.has?.(name)),
  );
  if (requiredFeatures.length === 0) return adapter.requestDevice();
  try {
    return await adapter.requestDevice({ requiredFeatures });
  } catch {
    return adapter.requestDevice();
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
      const device = await requestDeviceWith(adapter, OPTIONAL_FEATURES);
      // Allow re-acquisition after a device-lost event.
      void device.lost.then(() => {
        devicePromise = null;
      });
      return device;
    })();
  }

  return devicePromise;
}

