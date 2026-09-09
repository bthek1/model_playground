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

export function OverlayCanvas({
  width,
  height,
  draw,
  label,
  className,
  testId,
}: {
  width: number;
  height: number;
  /** Paint one frame. Called on every change of `draw`, `width` or `height`. */
  draw: (ctx: CanvasRenderingContext2D) => void;
  /** Accessible name — a canvas is opaque to a screen reader without one. */
  label: string;
  className?: string;
  testId?: string;
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
      className={cn("h-auto max-w-full rounded", className)}
    />
  );
}
