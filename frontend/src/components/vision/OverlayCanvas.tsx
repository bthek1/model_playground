// A canvas sized to the *source* image, painted by a callback.
//
// Depth, detection and segmentation all produce a picture rather than text, and
// all three want the same thing: a bitmap at the source's own resolution, scaled
// down by CSS to fit the column. Sizing in CSS instead would resample the
// overlay along with the picture and blur a one-pixel box edge into three.
//
// The 2D context is allowed to be missing. `getContext("2d")` returns null under
// happy-dom (and on a canvas the browser has starved of memory), and an OUTPUT
// slot that throws in that case takes the whole route down with it.

import { useEffect, useRef } from "react";

import { cn } from "@/lib/utils";

/** A click on the canvas, already converted to source-image pixels. */
export interface CanvasPick {
  x: number;
  y: number;
  /** True when the click carried Alt — `/mask-generation`'s negative point. */
  alt: boolean;
}

export function OverlayCanvas({
  width,
  height,
  draw,
  label,
  className,
  testId,
  onPick,
}: {
  width: number;
  height: number;
  /** Paint one frame. Called on every change of `draw`, `width` or `height`. */
  draw: (ctx: CanvasRenderingContext2D) => void;
  /** Accessible name — a canvas is opaque to a screen reader without one. */
  label: string;
  className?: string;
  testId?: string;
  /**
   * Make the canvas clickable, with the click reported in **source-image**
   * pixels rather than CSS ones.
   *
   * The conversion belongs here rather than in a caller: the canvas is sized to
   * the source and scaled down by CSS, so a raw `offsetX` is in display pixels
   * and is wrong by the scale factor. On `/mask-generation` that error is
   * invisible — SAM returns a perfectly plausible mask of whatever happens to
   * be at the mis-mapped point.
   */
  onPick?: (pick: CanvasPick) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || width <= 0 || height <= 0) return;
    // Assigning either dimension clears the canvas, so it happens before the
    // paint and only when the value actually changed — writing the same number
    // back would silently wipe the frame we are about to keep.
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    draw(ctx);
  }, [draw, width, height]);

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={label}
      data-testid={testId}
      onClick={
        onPick &&
        ((event) => {
          const canvas = event.currentTarget;
          const rect = canvas.getBoundingClientRect();
          // happy-dom reports a zero-sized rect; fall back to 1:1 rather than
          // dividing by zero and reporting Infinity.
          const scaleX = rect.width > 0 ? width / rect.width : 1;
          const scaleY = rect.height > 0 ? height / rect.height : 1;
          onPick({
            x: (event.clientX - rect.left) * scaleX,
            y: (event.clientY - rect.top) * scaleY,
            alt: event.altKey,
          });
        })
      }
      className={cn(
        "h-auto max-w-full rounded",
        onPick && "cursor-crosshair",
        className,
      )}
    />
  );
}
