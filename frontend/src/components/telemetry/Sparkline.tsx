// A 28px trend line, hand-written in SVG.
//
// model-visualization.md §5 is explicit that a sparkline in a card is inline
// SVG and not a charting library: ECharts is for the interactive, dense and
// animated, and pulling it in here would put a lazy chunk boundary and a canvas
// per card inside the one surface whose whole design constraint is costing
// nothing. `var(--chart-N)` also works directly in SVG, so there is no
// `getCSSVar()` round-trip and the line recolors with the theme for free.
//
// Scaling is the part that had to be got right. Two different questions get
// asked of these lines, and one scale cannot answer both:
//
//   * **Trend** (default): how is this moving? Auto-ranged to the series, so a
//     heap creeping up by 2 MB is visible. Normalising a near-constant series to
//     its own maximum — the first attempt here — pinned the line to the top of
//     the box and filled it solid, which read as "pegged" when the real answer
//     was 0.13 ms of frame lag.
//   * **Absolute** (`min`/`max` given): where does this sit on a known scale?
//     An idle busy-fraction belongs on the floor of a 0–1 axis, not in the
//     middle of an auto-range.

import { maxOf } from "@/telemetry/series";

const WIDTH = 100;
const HEIGHT = 28;
/** Head- and foot-room so an auto-ranged extreme isn't drawn on the border. */
const PAD = 3;

export interface SparklineProps {
  values: number[];
  /** Fixed floor. Given with `max`, switches from trend to absolute scaling. */
  min?: number;
  /** Fixed ceiling. */
  max?: number;
  /** Describes the trend for a screen reader — the shape alone says nothing. */
  label: string;
  /** A `--chart-N` token. */
  token?: string;
}

function minOf(values: number[]): number {
  let min = 0;
  let seen = false;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (!seen || v < min) {
      min = v;
      seen = true;
    }
  }
  return min;
}

export function Sparkline({
  values,
  min,
  max,
  label,
  token = "--chart-1",
}: SparklineProps) {
  // One point is not a trend, and a flat line drawn from it would imply a
  // steady state we haven't observed yet.
  if (values.length < 2) {
    return (
      <div
        role="img"
        aria-label={`${label}: not enough samples yet`}
        className="h-7 w-full border-b border-dashed border-border"
      />
    );
  }

  const floor = min ?? minOf(values);
  const ceiling = max ?? maxOf(values);
  const span = ceiling - floor;

  // A series that never varies has no shape to draw. On a fixed scale its
  // height still means something (idle sits at the bottom of a 0–1 axis); on an
  // auto-range it doesn't, so it is centred — except at a constant zero, where
  // the floor is the reading a viewer expects.
  const flatY =
    span > Number.EPSILON
      ? null
      : min != null && max != null
        ? HEIGHT - PAD
        : ceiling === 0
          ? HEIGHT - PAD
          : HEIGHT / 2;

  const points = values.map((value, index) => {
    const x = (index / (values.length - 1)) * WIDTH;
    if (flatY != null) return `${x.toFixed(2)},${flatY.toFixed(2)}`;
    const clamped = Math.min(Math.max(value, floor), ceiling);
    const unit = (clamped - floor) / span;
    const y = HEIGHT - PAD - unit * (HEIGHT - PAD * 2);
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      // The x-axis is "samples", not a measured quantity, so stretching is
      // correct here; `vector-effect` keeps the stroke from stretching with it.
      preserveAspectRatio="none"
      className="h-7 w-full overflow-visible"
      role="img"
      aria-label={label}
    >
      {/* A line, with no area fill under it. The fill was the first version
          and it turned every chart into a solid block at this height — a 28px
          band reads as a bar chart rather than as a trend. */}
      <polyline
        points={points.join(" ")}
        fill="none"
        stroke={`var(${token})`}
        strokeWidth={1.5}
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}
