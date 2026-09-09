// The frame pump for every live vision page.
//
// One rule, and it is the whole reason this hook exists: **do not queue frames.**
// Grab a frame only once the worker has returned the previous result. A naive
// `requestAnimationFrame` loop that posts every frame builds an unbounded
// backlog, and the overlay ends up seconds behind the picture while the page
// looks, misleadingly, like the model is slow. It is not — the queue is.
//
// The audio counterpart is `useLiveAsr`'s capture loop; this is the same shape
// with a camera instead of a microphone.

import { useCallback, useEffect, useRef, useState } from "react";

export interface UseLiveFramesOptions {
  /** The playing `<video>` to sample. */
  video: HTMLVideoElement | null;
  /** Run the loop. Flip to `false` to stop without unmounting. */
  active: boolean;
  /**
   * Handle one frame. Awaited — the next frame is not grabbed until it settles,
   * which is what keeps exactly one inference in flight.
   */
  onFrame: (video: HTMLVideoElement) => Promise<void>;
  /** Upper bound on grabs per second. The model is usually the real limit. */
  maxFps?: number;
}

export interface UseLiveFramesResult {
  /** Measured frames per second over the last few frames. Null until settled. */
  fps: number | null;
  /** True while a frame is in flight. */
  busy: boolean;
}

export function useLiveFrames({
  video,
  active,
  onFrame,
  maxFps = 30,
}: UseLiveFramesOptions): UseLiveFramesResult {
  const [fps, setFps] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);

  // Refs, not deps: a new inline `onFrame` on every render must not restart the
  // loop — restarting it mid-frame is how a "one in flight" guard springs a leak.
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;
  const inFlight = useRef(false);
  const recent = useRef<number[]>([]);

  useEffect(() => {
    if (!active || !video) {
      setFps(null);
      recent.current = [];
      return;
    }

    let raf = 0;
    let live = true;
    const minInterval = 1000 / maxFps;
    let lastStart = 0;

    const tick = () => {
      raf = requestAnimationFrame(tick);
      if (inFlight.current) return;
      const now = performance.now();
      if (now - lastStart < minInterval) return;
      lastStart = now;

      inFlight.current = true;
      setBusy(true);
      void onFrameRef
        .current(video)
        .catch(() => {
          /* a failed frame is the caller's to report; the loop keeps going */
        })
        .finally(() => {
          inFlight.current = false;
          if (!live) return;
          setBusy(false);
          const elapsed = performance.now() - now;
          const window = recent.current;
          window.push(elapsed);
          if (window.length > 10) window.shift();
          const mean = window.reduce((a, b) => a + b, 0) / window.length;
          setFps(mean > 0 ? Math.round(1000 / mean) : null);
        });
    };

    raf = requestAnimationFrame(tick);
    return () => {
      live = false;
      cancelAnimationFrame(raf);
      inFlight.current = false;
      setBusy(false);
    };
  }, [active, video, maxFps]);

  return { fps, busy };
}

/**
 * Open the camera into a `<video>` for as long as `active` is true, and close it
 * on unmount. Split out from the pump so a page can show a live preview without
 * running a model, and so the teardown has exactly one owner.
 */
export function useCamera(
  video: HTMLVideoElement | null,
  active: boolean,
  open: (video: HTMLVideoElement) => Promise<() => void>,
): { error: string | null } {
  const [error, setError] = useState<string | null>(null);

  const openRef = useRef(open);
  openRef.current = open;

  const stopRef = useRef<(() => void) | null>(null);
  const stop = useCallback(() => {
    stopRef.current?.();
    stopRef.current = null;
  }, []);

  useEffect(() => {
    if (!active || !video) {
      stop();
      return;
    }
    let cancelled = false;
    setError(null);
    void openRef
      .current(video)
      .then((close) => {
        // Unmounted while the permission prompt was up: close it immediately
        // rather than leaving the camera light on with nothing rendering it.
        if (cancelled) close();
        else stopRef.current = close;
      })
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : String(e)),
      );
    return () => {
      cancelled = true;
      stop();
    };
  }, [active, video, stop]);

  return { error };
}
