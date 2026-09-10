import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { PointCloud } from "@/vision/pointCloud";

/** A stand-in renderer, so nothing here needs a GPUDevice or a webgpu canvas. */
const upload = vi.fn();
const render = vi.fn();
const destroy = vi.fn();
const create = vi.fn(async () => ({ upload, render, destroy }));

vi.mock("@/webgpu/pointRenderer", () => ({
  DEFAULT_CAMERA: { yaw: 0, pitch: 0, distance: 3, pointSize: 1.5 },
  PointRenderer: {
    create: (...a: unknown[]) => create(...(a as [])),
  },
}));

const { usePointCloudView } = await import("./usePointCloudView");

/** A cloud with an explicit extent, so framing is predictable. */
function cloud(span = 4, count = 100): PointCloud {
  return {
    data: new Float32Array(count * 6),
    count,
    bounds: { min: [-span / 2, -span / 2, 1], max: [span / 2, span / 2, 1 + span] },
  };
}

/** happy-dom has no layout, so a canvas needs a size for the render effect. */
function fakeCanvas(): HTMLCanvasElement {
  return {
    width: 0,
    height: 0,
    getBoundingClientRect: () => ({ width: 600, height: 400 }),
  } as unknown as HTMLCanvasElement;
}

/** Mount, attach a canvas, and wait for the renderer promise to settle. */
async function mounted(initial: PointCloud | null = null, fitKey?: unknown) {
  const view = renderHook(
    ({ c, k }: { c: PointCloud | null; k?: unknown }) =>
      usePointCloudView(c, k),
    { initialProps: { c: initial, k: fitKey } },
  );
  await act(async () => {
    view.result.current.canvasRef(fakeCanvas());
  });
  await waitFor(() => expect(view.result.current.supported).toBe(true));
  return view;
}

describe("usePointCloudView", () => {
  afterEach(() => vi.clearAllMocks());

  it("reports `null` until the renderer has answered, never `false`", () => {
    // `null` is "not decided yet" and `false` is "this machine has no GPU". A
    // route that treated the undecided state as a failure would flash the
    // fallback on every page load.
    const { result } = renderHook(() => usePointCloudView(null));
    expect(result.current.supported).toBeNull();
  });

  it("reports unsupported when the renderer cannot be built", async () => {
    // `PointRenderer.create` returns null rather than throwing — a 3-D view that
    // takes the page down on a machine without a GPU is worse than one that says
    // so — and the hook has to carry that all the way to the route.
    create.mockResolvedValueOnce(null as never);
    const { result } = renderHook(() => usePointCloudView(null));
    await act(async () => {
      result.current.canvasRef(fakeCanvas());
    });
    await waitFor(() => expect(result.current.supported).toBe(false));
  });

  it("uploads the cloud once, with its centre, and frames the camera", async () => {
    const view = await mounted(null, "a");
    expect(upload).not.toHaveBeenCalled();

    await act(async () => {
      view.rerender({ c: cloud(4), k: "b" });
    });

    expect(upload).toHaveBeenCalledTimes(1);
    // Orbiting turns around the scene, not the origin.
    expect(upload.mock.calls[0][0].center).toEqual([0, 0, 3]);
    // Framed from the cloud's own extent: the depth range varies per image, so a
    // fixed distance would put one scene in your face and the next a speck.
    expect(view.result.current.camera.distance).toBeCloseTo(4 * 1.6, 5);
  });

  it("re-uploads when the cloud changes but re-frames only on a new result", async () => {
    // **The bug this test exists for.** Both sliders re-derive a *new* cloud
    // object from the same depth map, so framing on every cloud would snap the
    // camera back to its default distance mid-drag and the view would fight the
    // user on the two controls the page is built around.
    const view = await mounted(cloud(4), "result-1");
    await waitFor(() => expect(upload).toHaveBeenCalledTimes(1));

    await act(async () => {
      view.result.current.onWheel(-500); // the user zooms in
    });
    const zoomed = view.result.current.camera.distance;
    expect(zoomed).toBeLessThan(4 * 1.6);

    // A slider moves: same depth result, a differently-strided cloud.
    await act(async () => {
      view.rerender({ c: cloud(4, 25), k: "result-1" });
    });
    expect(upload).toHaveBeenCalledTimes(2);
    expect(view.result.current.camera.distance).toBe(zoomed);

    // A genuinely new inference re-frames.
    await act(async () => {
      view.rerender({ c: cloud(10), k: "result-2" });
    });
    expect(view.result.current.camera.distance).toBeCloseTo(10 * 1.6, 5);
  });

  it("orbits on drag, and only while the pointer is down", async () => {
    const view = await mounted(cloud());
    const before = view.result.current.camera.yaw;

    // A move with no preceding press must not rotate anything.
    await act(async () => {
      window.dispatchEvent(new PointerEvent("pointermove", { clientX: 50, clientY: 0 }));
    });
    expect(view.result.current.camera.yaw).toBe(before);

    await act(async () => {
      view.result.current.onPointerDown({ clientX: 0, clientY: 0 });
      window.dispatchEvent(new PointerEvent("pointermove", { clientX: 50, clientY: 0 }));
    });
    expect(view.result.current.camera.yaw).toBeGreaterThan(before);

    // Releasing ends the drag, wherever the pointer is.
    await act(async () => {
      window.dispatchEvent(new PointerEvent("pointerup"));
    });
    const held = view.result.current.camera.yaw;
    await act(async () => {
      window.dispatchEvent(new PointerEvent("pointermove", { clientX: 500, clientY: 0 }));
    });
    expect(view.result.current.camera.yaw).toBe(held);
  });

  it("clamps pitch short of the pole, so the view never flips", async () => {
    const view = await mounted(cloud());
    await act(async () => {
      view.result.current.onPointerDown({ clientX: 0, clientY: 0 });
      for (let i = 0; i < 200; i++) {
        window.dispatchEvent(
          new PointerEvent("pointermove", { clientX: 0, clientY: (i + 1) * 20 }),
        );
      }
    });
    expect(Math.abs(view.result.current.camera.pitch)).toBeLessThanOrEqual(1.5);
  });

  it("clamps zoom out of the cloud and out of deep space", async () => {
    const view = await mounted(cloud());

    await act(async () => {
      for (let i = 0; i < 200; i++) view.result.current.onWheel(-500);
    });
    // Never inside the scene, where the camera would see nothing.
    expect(view.result.current.camera.distance).toBeGreaterThanOrEqual(0.4);

    await act(async () => {
      for (let i = 0; i < 400; i++) view.result.current.onWheel(500);
    });
    expect(view.result.current.camera.distance).toBeLessThanOrEqual(40);
  });

  it("renders on a camera change, not on a re-render", async () => {
    // The cheap effect: 48 bytes of uniform and a draw call. The *expensive* one
    // is `upload`, and the two are separate effects precisely so an orbit tick
    // does not re-send a multi-megabyte vertex buffer.
    const view = await mounted(cloud());
    await waitFor(() => expect(render).toHaveBeenCalled());
    const uploads = upload.mock.calls.length;
    const renders = render.mock.calls.length;

    await act(async () => {
      view.result.current.onWheel(-100);
    });

    expect(render.mock.calls.length).toBeGreaterThan(renders);
    expect(upload.mock.calls.length).toBe(uploads);
  });

  it("resets to the default framing on request", async () => {
    const view = await mounted(cloud(4), "r");
    await act(async () => {
      view.result.current.onPointerDown({ clientX: 0, clientY: 0 });
      window.dispatchEvent(new PointerEvent("pointermove", { clientX: 80, clientY: 40 }));
    });
    expect(view.result.current.camera.yaw).not.toBe(0);

    await act(async () => {
      view.result.current.reset();
    });
    expect(view.result.current.camera.yaw).toBe(0);
    expect(view.result.current.camera.pitch).toBe(0);
  });

  it("destroys the renderer on unmount", async () => {
    // A cloud is megabytes of GPU memory, and a session of picking images would
    // leak one per visit.
    const view = await mounted(cloud());
    view.unmount();
    await waitFor(() => expect(destroy).toHaveBeenCalled());
  });
});
