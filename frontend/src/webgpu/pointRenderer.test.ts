import { afterEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_CAMERA, PointRenderer } from "./pointRenderer";

// Only the degradation path is unit-testable here: everything past
// `getGPUDevice()` needs a real adapter and a `webgpu` canvas context, neither of
// which happy-dom has. The drawing itself is covered by `/image-to-3d`'s `@slow`
// spec and, on a machine with a GPU, by the `webgpu` Playwright project.
//
// This file exists for the one promise the *rest of the app* depends on:
// **`create()` never throws.** `detectWebGPU()` never throws either, and the
// route's whole fallback — show the depth map, say the 3-D view needs WebGPU —
// rests on that. A renderer that threw would take the OUTPUT slot down with it
// on every machine without a GPU, which is most of them.

const getGPUDevice = vi.fn();
vi.mock("./device", () => ({
  getGPUDevice: () => getGPUDevice(),
}));

/** A canvas whose `getContext` answers however the test needs it to. */
function canvasReturning(context: unknown): HTMLCanvasElement {
  return { getContext: () => context } as unknown as HTMLCanvasElement;
}

describe("PointRenderer.create", () => {
  afterEach(() => vi.clearAllMocks());

  it("returns null instead of throwing when there is no device", async () => {
    // `getGPUDevice` raises `WebGPUUnavailableError` on a machine with no
    // adapter. The route needs an answer, not an exception.
    getGPUDevice.mockRejectedValueOnce(new Error("No WebGPU adapter available"));
    await expect(
      PointRenderer.create(canvasReturning(null)),
    ).resolves.toBeNull();
  });

  it("returns null when the canvas will not give a webgpu context", async () => {
    // A device can exist while `getContext("webgpu")` still returns null — a
    // canvas already bound to a 2-D context, or a browser that exposes the API
    // without the canvas integration.
    getGPUDevice.mockResolvedValueOnce({} as GPUDevice);
    await expect(
      PointRenderer.create(canvasReturning(null)),
    ).resolves.toBeNull();
  });

  it("does not ask for a canvas context before it has a device", async () => {
    // Ordering matters: configuring a context against a device that never
    // arrived is how this would throw rather than degrade.
    getGPUDevice.mockRejectedValueOnce(new Error("nope"));
    const getContext = vi.fn(() => null);
    await PointRenderer.create({ getContext } as unknown as HTMLCanvasElement);
    expect(getContext).not.toHaveBeenCalled();
  });
});

describe("DEFAULT_CAMERA", () => {
  it("starts square-on, at a distance the framing will replace", () => {
    // Yaw and pitch at zero means the first frame looks straight at the scene,
    // which is the only orientation that reads as "the photo, in 3-D". The
    // distance is a placeholder — `usePointCloudView` reframes from the cloud's
    // own extent as soon as one arrives.
    expect(DEFAULT_CAMERA.yaw).toBe(0);
    expect(DEFAULT_CAMERA.pitch).toBe(0);
    expect(DEFAULT_CAMERA.distance).toBeGreaterThan(0);
  });

  it("gives a point more than one pixel", () => {
    // WebGPU's `point-list` is always exactly 1 px — there is no `gl_PointSize`
    // — which is why points are drawn as instanced quads at all. A radius that
    // fell back to 1 would undo the reason for the extra geometry.
    expect(DEFAULT_CAMERA.pointSize).toBeGreaterThan(1);
  });
});
