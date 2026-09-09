// Canvas drawing for vision outputs — boxes, single-channel maps, class masks.
// Most vision tasks produce something to draw *over* the input rather than text,
// so the OUTPUT slot is a <canvas> sized to the source image and these are the
// three things that get painted into it.
//
// Read docs/standards/model-visualization.md before adding a fourth. The signed
// diverging heatmap already exists in `components/viz/heatmap.tsx` (red = +,
// blue = −, alpha = magnitude) and is used by /training and /tensor — this file
// deliberately does not fork it. What it adds is the *sequential* case (a depth
// map has no meaningful zero) and the two overlay forms.
//
// Colours are literal Tailwind palette values rather than CSS custom properties
// on purpose: a canvas cannot read a CSS variable, and the 500-level hues used
// here carry enough contrast to stay legible on both the light and the dark
// page ground. Same trade-off, and same reasoning, as `POS`/`NEG` in
// `components/viz/heatmap.tsx`.

/** One detection, in absolute source-image pixels. */
export interface Box {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
}

export interface Detection {
  box: Box;
  label: string;
  score: number;
}

/**
 * Categorical overlay palette (Tailwind 500s, ordered for maximum separation
 * between neighbours). Ten is enough: past that, a picture with a box per class
 * is unreadable for reasons a palette cannot fix.
 */
export const OVERLAY_COLORS: readonly string[] = [
  "#06b6d4", // cyan
  "#f97316", // orange
  "#a855f7", // purple
  "#22c55e", // green
  "#ef4444", // red
  "#eab308", // yellow
  "#3b82f6", // blue
  "#ec4899", // pink
  "#14b8a6", // teal
  "#8b5cf6", // violet
];

/**
 * A stable colour for a class name. Deterministic, so a label keeps its colour
 * between runs and between frames of a live feed — a class that changes colour
 * every frame is worse than no colour at all.
 */
export function colorForLabel(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = (hash * 31 + label.charCodeAt(i)) | 0;
  }
  return OVERLAY_COLORS[Math.abs(hash) % OVERLAY_COLORS.length];
}

/**
 * Draw detection boxes with their labels.
 *
 * The boxes must be in **absolute pixels** — a Transformers.js detection
 * pipeline gives you those with `{ percentage: false }`. Getting that backwards
 * hands you 0–1 fractions, every box collapses into the top-left corner, and it
 * is the single most common bug on a first detection page.
 */
export function drawBoxes(
  ctx: CanvasRenderingContext2D,
  detections: readonly Detection[],
  { lineWidth = 2, font = "14px system-ui, sans-serif" } = {},
): void {
  ctx.save();
  ctx.lineWidth = lineWidth;
  ctx.font = font;
  ctx.textBaseline = "top";

  for (const { box, label, score } of detections) {
    const color = colorForLabel(label);
    const width = box.xmax - box.xmin;
    const height = box.ymax - box.ymin;

    ctx.strokeStyle = color;
    ctx.strokeRect(box.xmin, box.ymin, width, height);

    // The caption sits inside the box when the box is near the top edge, so it
    // is never clipped off the canvas.
    const text = `${label} ${score.toFixed(2)}`;
    const padding = 3;
    const textWidth = ctx.measureText(text).width;
    const captionHeight = lineWidth + 16;
    const above = box.ymin >= captionHeight;
    const captionY = above ? box.ymin - captionHeight : box.ymin;

    ctx.fillStyle = color;
    ctx.fillRect(box.xmin, captionY, textWidth + padding * 2, captionHeight);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, box.xmin + padding, captionY + padding);
  }

  ctx.restore();
}

/** The value range of a map, for a legend. Never a zero-width range. */
export function rangeOf(data: ArrayLike<number>): { lo: number; hi: number } {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { lo: 0, hi: 1 };
  return { lo, hi };
}

/**
 * Sequential colour ramp, dark blue → cyan → yellow → white, for `t` in [0, 1].
 * Monotonic in lightness, which is what makes a depth map readable in
 * greyscale, in both themes, and to a colour-blind reader.
 */
export const RAMP_STOPS: readonly [number, [number, number, number]][] = [
  [0.0, [12, 24, 74]],
  [0.35, [30, 110, 180]],
  [0.65, [45, 200, 190]],
  [0.85, [235, 220, 90]],
  [1.0, [255, 255, 255]],
];

/**
 * The same ramp as a CSS gradient, so a legend beside the canvas is coloured by
 * the *same* stops the pixels were. A hand-written gradient that drifts from the
 * ramp is a legend that lies.
 */
export const RAMP_CSS = `linear-gradient(to right, ${RAMP_STOPS.map(
  ([t, [r, g, b]]) => `rgb(${r} ${g} ${b}) ${Math.round(t * 100)}%`,
).join(", ")})`;

function ramp(t: number): [number, number, number] {
  const stops = RAMP_STOPS;
  const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) {
    const [t1, c1] = stops[i];
    if (x <= t1) {
      const [t0, c0] = stops[i - 1];
      const k = t1 === t0 ? 0 : (x - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * k),
        Math.round(c0[1] + (c1[1] - c0[1]) * k),
        Math.round(c0[2] + (c1[2] - c0[2]) * k),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

/**
 * Paint a single-channel map (a depth map, one segmentation logit) as a
 * colourised canvas.
 *
 * **The per-map normalisation is not cosmetic.** A relative-depth model emits an
 * inverse-depth map on an arbitrary scale — without rescaling to the values
 * actually present, the canvas comes out uniformly black or uniformly white and
 * the page looks broken rather than wrong. Pass an explicit `lo`/`hi` only when
 * you deliberately want a fixed scale across frames.
 */
export function drawHeatmap(
  ctx: CanvasRenderingContext2D,
  data: ArrayLike<number>,
  width: number,
  height: number,
  bounds?: { lo: number; hi: number },
): { lo: number; hi: number } {
  const { lo, hi } = bounds ?? rangeOf(data);
  const span = hi - lo || 1; // a constant map paints as a single flat colour
  const img = ctx.createImageData(width, height);
  const count = Math.min(data.length, width * height);

  for (let i = 0; i < count; i++) {
    const [r, g, b] = ramp((data[i] - lo) / span);
    const o = i * 4;
    img.data[o] = r;
    img.data[o + 1] = g;
    img.data[o + 2] = b;
    img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return { lo, hi };
}

/** One class mask: single-channel coverage, same dimensions as the source. */
export interface LabelledMask {
  label: string;
  /** Single-channel mask data, row-major. 0–255 or 0–1, both are handled. */
  data: ArrayLike<number>;
  width: number;
  height: number;
}

/**
 * Composite class masks into one canvas, each in its label's colour.
 *
 * One canvas, not N — a semantic segmentation model with 150 ADE classes hands
 * back one mask per class present, and rendering them as separate images is the
 * failure mode this helper exists to prevent. Later masks paint over earlier
 * ones, so pass them in ascending order of importance.
 */
export function drawMasks(
  ctx: CanvasRenderingContext2D,
  masks: readonly LabelledMask[],
  { width, height, alpha = 0.55 }: { width: number; height: number; alpha?: number },
): void {
  if (masks.length === 0) return;
  // Composited through a scratch canvas rather than `putImageData` straight onto
  // `ctx`: `putImageData` *replaces* pixels, alpha included, so writing the
  // overlay directly would erase the source image underneath instead of tinting
  // it. `drawImage` is the call that actually blends.
  const scratch = document.createElement("canvas");
  scratch.width = width;
  scratch.height = height;
  const sctx = scratch.getContext("2d");
  if (!sctx) return;
  const img = sctx.createImageData(width, height);
  const a = Math.round(Math.min(1, Math.max(0, alpha)) * 255);

  for (const mask of masks) {
    const color = colorForLabel(mask.label);
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);
    const count = Math.min(mask.data.length, width * height);

    for (let i = 0; i < count; i++) {
      // Tolerate both 0–1 probabilities and 0–255 bytes: anything past the
      // midpoint of its own scale counts as covered.
      const v = mask.data[i];
      if (v <= (v > 1 ? 127 : 0.5)) continue;
      const o = i * 4;
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = a;
    }
  }
  sctx.putImageData(img, 0, 0);
  ctx.drawImage(scratch, 0, 0);
}

// --- Source pixels, and the coordinate space they live in --------------------

/** A `RawImage`-shaped pixel buffer: interleaved samples plus its dimensions. */
export interface Pixels {
  data: ArrayLike<number>;
  width: number;
  height: number;
  /** 1 = grey, 3 = RGB, 4 = RGBA. */
  channels: number;
}

/**
 * Paint a `RawImage`'s own pixels into a canvas.
 *
 * Every overlay in this file draws *over* the picture the model saw, so the
 * picture has to get into the canvas first. Going through the `RawImage` rather
 * than the `<img>` on screen is deliberate: it is the same buffer the pipeline
 * was handed, it needs no load event, and it works identically for a webcam
 * frame — which has no `<img>` at all.
 */
export function drawPixels(ctx: CanvasRenderingContext2D, src: Pixels): void {
  const { width, height, channels: c } = src;
  const img = ctx.createImageData(width, height);
  const count = Math.min(Math.floor(src.data.length / c), width * height);

  for (let i = 0; i < count; i++) {
    const s = i * c;
    const o = i * 4;
    if (c === 1) {
      const v = src.data[s];
      img.data[o] = v;
      img.data[o + 1] = v;
      img.data[o + 2] = v;
      img.data[o + 3] = 255;
    } else {
      img.data[o] = src.data[s];
      img.data[o + 1] = src.data[s + 1];
      img.data[o + 2] = src.data[s + 2];
      img.data[o + 3] = c === 4 ? src.data[s + 3] : 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/**
 * Move detections from the resolution the model ran at to the resolution the
 * canvas is drawn at.
 *
 * `downscale()` caps the *source* before inference (resolution is the throttle —
 * a detector at 1280x720 costs ~4x the same detector at 640x480), so the boxes
 * come back in the downscaled frame's pixels while the canvas shows the original.
 * Without this every box is drawn a fixed fraction too small and too far
 * top-left, which looks like a mediocre detector rather than a bug in our
 * arithmetic — the same class of silent failure as getting `percentage` backwards.
 */
export function scaleDetections(
  detections: readonly Detection[],
  scale: number,
): Detection[] {
  if (scale === 1) return [...detections];
  return detections.map((d) => ({
    ...d,
    box: {
      xmin: d.box.xmin * scale,
      ymin: d.box.ymin * scale,
      xmax: d.box.xmax * scale,
      ymax: d.box.ymax * scale,
    },
  }));
}
