// The interactive 3-D view: a `PointRenderer` bound to a canvas, an orbit
// camera, and the rule that keeps the two cheap.
//
// **The vertex buffer is written once per cloud; the camera is written once per
// frame.** A cloud at stride 2 from a 640x480 depth map is 76,800 points — about
// 1.8 MB — and re-uploading that on every pointer move would spend the whole
// frame budget moving data that did not change. So `upload` runs when the cloud
// changes and `render` runs when the camera does, and those are different
// effects on purpose.
//
// It degrades rather than throwing: `PointRenderer.create` returns null on a
// machine with no usable GPU, `supported` goes false, and the route shows the
// depth map plus an explanation instead of a blank canvas.

import { useCallback, useEffect, useRef, useState } from "react";

import {
  DEFAULT_CAMERA,
  PointRenderer,
  type CameraState,
} from "@/webgpu/pointRenderer";
import type { PointCloud } from "@/vision/pointCloud";

export interface UsePointCloudViewResult {
  /** Attach to the `<canvas>`. A callback ref — the element appears late. */
  canvasRef: (el: HTMLCanvasElement | null) => void;
  /** False once the probe has answered and there is no usable GPU. */
  supported: boolean | null;
  camera: CameraState;
  setCamera: (next: CameraState) => void;
  /** Wire to the canvas's `onPointerDown`; the drag is tracked on the window. */
  onPointerDown: (e: { clientX: number; clientY: number }) => void;
  /** Wire to the canvas's `onWheel` to zoom. */
  onWheel: (deltaY: number) => void;
  /** Back to the default framing. */
  reset: () => void;
}

/** Keep the camera out of the cloud and out of deep space. */
const MIN_DISTANCE = 0.4;
const MAX_DISTANCE = 40;
/** Just under a right angle, so the view never flips through the pole. */
const MAX_PITCH = 1.5;

export function usePointCloudView(
  cloud: PointCloud | null,
  /**
   * Changes only when a **new inference** produced this cloud — the route passes
   * the depth result itself.
   *
   * Framing has to key off that rather than off `cloud`, because both sliders
   * re-derive a *new* cloud object from the same depth map: re-framing on every
   * cloud would snap the camera back to its default distance mid-drag, and the
   * view would fight the user on exactly the two controls the page is built
   * around.
   */
  fitKey?: unknown,
): UsePointCloudViewResult {
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null);
  const [supported, setSupported] = useState<boolean | null>(null);
  const [camera, setCamera] = useState<CameraState>(DEFAULT_CAMERA);
  const renderer = useRef<PointRenderer | null>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);

  // Create the renderer when the canvas appears, and tear it down on the way
  // out — a cloud is megabytes of GPU memory, and leaking one per visit adds up
  // across a session of picking images.
  useEffect(() => {
    if (!canvas) return;
    let live = true;
    let made: PointRenderer | null = null;

    void PointRenderer.create(canvas).then((next) => {
      made = next;
      if (!live) {
        next?.destroy();
        return;
      }
      renderer.current = next;
      setSupported(next != null);
    });

    return () => {
      live = false;
      renderer.current = null;
      made?.destroy();
    };
  }, [canvas]);

  // Upload on a *cloud* change only. This is the expensive one.
  useEffect(() => {
    const r = renderer.current;
    if (!r || !cloud) return;
    const { min, max } = cloud.bounds;
    r.upload({
      data: cloud.data,
      count: cloud.count,
      center: [
        (min[0] + max[0]) / 2,
        (min[1] + max[1]) / 2,
        (min[2] + max[2]) / 2,
      ],
    });
  }, [cloud, supported]);

  // Frame on a *new inference*, never on a re-derivation. The depth range varies
  // per image, so a fixed distance would put one scene in your face and leave
  // the next a speck — but re-framing when a slider moves would undo the user's
  // own zoom on every drag tick.
  const bounds = cloud?.bounds;
  const boundsRef = useRef(bounds);
  boundsRef.current = bounds;
  useEffect(() => {
    const b = boundsRef.current;
    if (!b) return;
    const span = Math.max(
      b.max[0] - b.min[0],
      b.max[1] - b.min[1],
      b.max[2] - b.min[2],
      0.1,
    );
    setCamera((c) => ({ ...c, distance: span * 1.6 }));
  }, [fitKey, supported]);

  // Render on a *camera* change. This one is 48 bytes and a draw call.
  useEffect(() => {
    const r = renderer.current;
    if (!r || !canvas || !cloud) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    r.render(canvas, camera);
  }, [camera, canvas, cloud, supported]);

  // Drag tracked on the window, so a pointer that leaves the canvas keeps
  // orbiting and the drag ends wherever it is released.
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const from = drag.current;
      if (!from) return;
      const dx = e.clientX - from.x;
      const dy = e.clientY - from.y;
      drag.current = { x: e.clientX, y: e.clientY };
      setCamera((c) => ({
        ...c,
        yaw: c.yaw + dx * 0.006,
        pitch: Math.max(-MAX_PITCH, Math.min(MAX_PITCH, c.pitch + dy * 0.006)),
      }));
    };
    const onUp = () => {
      drag.current = null;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, []);

  const onPointerDown = useCallback((e: { clientX: number; clientY: number }) => {
    drag.current = { x: e.clientX, y: e.clientY };
  }, []);

  const onWheel = useCallback((deltaY: number) => {
    setCamera((c) => ({
      ...c,
      distance: Math.max(
        MIN_DISTANCE,
        Math.min(MAX_DISTANCE, c.distance * (1 + deltaY * 0.0015)),
      ),
    }));
  }, []);

  const reset = useCallback(() => setCamera(DEFAULT_CAMERA), []);

  return {
    canvasRef: setCanvas,
    supported,
    camera,
    setCamera,
    onPointerDown,
    onWheel,
    reset,
  };
}
