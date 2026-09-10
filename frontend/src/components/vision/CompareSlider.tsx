// A draggable before/after split between two images of the same size.
//
// This is the OUTPUT of `/super-resolution`, and it is not decoration. A 2x
// image on its own proves nothing — every upscaler produces one, and the eye has
// no reference to judge it against. Against a bicubic upscale of the same input,
// at the same size, under a handle the user drags themselves, the difference is
// either there or it isn't. That is the whole claim the page makes.
//
// Both halves are painted into canvases at source resolution and revealed with
// `clip-path`, so the two sides are the *same pixels* the download contains,
// scaled down by CSS exactly as `OverlayCanvas` does. Rasterising them at
// display size instead would resample both, and could easily make a sharpening
// model look like a blurring one.
//
// Keyboard-operable on purpose: the handle is a slider, so arrow keys move it.
// A drag-only comparison is a comparison a keyboard user cannot make.

import { useCallback, useEffect, useRef, useState } from "react";

import { OverlayCanvas } from "@/components/vision/OverlayCanvas";
import type { Pixels } from "@/vision/draw";
import { drawPixels } from "@/vision/draw";

export function CompareSlider({
  before,
  after,
  beforeLabel,
  afterLabel,
  testId,
}: {
  before: Pixels;
  after: Pixels;
  beforeLabel: string;
  afterLabel: string;
  testId?: string;
}) {
  const [split, setSplit] = useState(50);
  const frame = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const moveTo = useCallback((clientX: number) => {
    const rect = frame.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setSplit(Math.min(100, Math.max(0, pct)));
  }, []);

  // Listeners on the window, not the element: a drag that leaves the frame must
  // keep tracking, and must end wherever the pointer is released.
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      if (dragging.current) moveTo(e.clientX);
    };
    const onUp = () => {
      dragging.current = false;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [moveTo]);

  const paintBefore = useCallback(
    (ctx: CanvasRenderingContext2D) => drawPixels(ctx, before),
    [before],
  );
  const paintAfter = useCallback(
    (ctx: CanvasRenderingContext2D) => drawPixels(ctx, after),
    [after],
  );

  return (
    <div className="space-y-2">
      <div
        ref={frame}
        data-testid={testId}
        className="relative touch-none overflow-hidden rounded select-none"
        onPointerDown={(e) => {
          dragging.current = true;
          moveTo(e.clientX);
        }}
      >
        <OverlayCanvas
          width={before.width}
          height={before.height}
          draw={paintBefore}
          label={beforeLabel}
          className="block w-full"
        />

        {/* The "after" half sits on top, revealed from the left edge.
            `clip-path` rather than a width-clipped wrapper: the inner canvas is
            then still `w-full` of the same box, so both halves display at
            exactly the same size with nothing measured in JavaScript. Both stay
            mounted, so the comparison never flickers through a blank frame
            mid-drag. */}
        <div
          className="absolute inset-0"
          style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
        >
          <OverlayCanvas
            width={after.width}
            height={after.height}
            draw={paintAfter}
            label={afterLabel}
            className="block w-full"
          />
        </div>

        <div
          role="slider"
          tabIndex={0}
          aria-label={`Reveal ${afterLabel} over ${beforeLabel}`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(split)}
          data-testid="compare-handle"
          onKeyDown={(e) => {
            if (e.key === "ArrowLeft") setSplit((s) => Math.max(0, s - 5));
            if (e.key === "ArrowRight") setSplit((s) => Math.min(100, s + 5));
          }}
          className="absolute inset-y-0 w-0.5 cursor-ew-resize bg-white/90 shadow-[0_0_0_1px_rgba(0,0,0,0.35)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
          style={{ left: `${split}%` }}
        >
          <span
            aria-hidden
            className="absolute top-1/2 left-1/2 size-6 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-black/45"
          />
        </div>
      </div>

      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{afterLabel}</span>
        <span>{beforeLabel}</span>
      </div>
    </div>
  );
}
