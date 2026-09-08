// The VAD output, drawn rather than listed — per the model-visualization
// standard (docs/standards/model-visualization.md): painted from the real
// Float32Array, theme-aware by construction (the bars take the canvas's own
// resolved `color`, so a Tailwind text class themes them in both schemes), and
// aligned to the same horizontal time axis as the <Waveform> above it.
//
// What it shows is the *model's* answer and the *user's* answer at once: the
// per-frame probability as a column chart, the threshold as a line across it,
// and the frames that survive as filled columns. Dragging the threshold is the
// whole point of the page — you can see the marginal frames sitting just under
// the line, which a list of segments hides completely.

import { useEffect, useRef } from "react";

const HEIGHT = 72; // CSS pixels

function paint(
  canvas: HTMLCanvasElement,
  probabilities: Float32Array,
  threshold: number,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return; // e.g. happy-dom in tests

  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 300;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(HEIGHT * dpr);
  ctx.scale(dpr, dpr);

  const color = getComputedStyle(canvas).color;
  ctx.clearRect(0, 0, width, HEIGHT);
  if (probabilities.length === 0) return;

  const barWidth = width / probabilities.length;
  for (let f = 0; f < probabilities.length; f++) {
    const p = probabilities[f];
    const h = Math.max(1, p * (HEIGHT - 2));
    // Above the line reads as a decision, below it as evidence. Same hue, so
    // the eye follows one curve rather than comparing two colours.
    ctx.globalAlpha = p > threshold ? 0.9 : 0.28;
    ctx.fillStyle = color;
    ctx.fillRect(f * barWidth, HEIGHT - h, Math.max(1, barWidth - 0.5), h);
  }

  ctx.globalAlpha = 0.85;
  ctx.strokeStyle = color;
  ctx.setLineDash([4, 3]);
  ctx.lineWidth = 1;
  const y = HEIGHT - threshold * (HEIGHT - 2);
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(width, y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
}

/**
 * Speech probability per frame, with the current threshold drawn across it.
 * Colour rides on `currentColor` — override with any text-colour class.
 */
export function VadTimeline({
  probabilities,
  threshold,
  className,
}: {
  probabilities: Float32Array;
  threshold: number;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => paint(canvas, probabilities, threshold);
    draw();

    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [probabilities, threshold]);

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={`Speech probability for ${probabilities.length} frames, threshold ${threshold.toFixed(2)}`}
      className={className}
      style={{ width: "100%", height: HEIGHT }}
    />
  );
}
