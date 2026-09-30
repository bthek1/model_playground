// A time series, and what the page is willing to say about it.
//
// The parsing rule that matters here is a refusal: **gaps and irregular spacing
// are reported, never interpolated.** A series with missing periods silently
// filled gives a seasonal naive forecast that is confidently off by a phase —
// every value lands one step from where it belongs, the chart looks plausible,
// and the error is attributed to the method rather than to the fill. Saying
// "this series has four gaps" and letting the user decide is both honest and
// the more useful answer.

export interface SeriesGap {
  /** Index in `values` *after* which the gap occurs. */
  index: number;
  /** How many periods appear to be missing. */
  missing: number;
}

export interface Series {
  name: string;
  values: Float32Array;
  /** Epoch milliseconds per point, or null when the input was a bare column. */
  times: Float64Array | null;
  /** The modal gap between consecutive timestamps, in ms. Null without times. */
  stepMs: number | null;
  /** What that step is in words — "daily", "monthly". Null when unrecognised. */
  frequency: string | null;
  /** A natural season length for that frequency (12 for monthly), else null. */
  suggestedSeason: number | null;
  gaps: SeriesGap[];
  /** True when the spacing is not a consistent multiple of `stepMs`. */
  irregular: boolean;
}

const DAY = 86_400_000;

/** Named steps, longest first, with the season a period of that length implies. */
const FREQUENCIES: { label: string; ms: number; season: number | null; tolerance: number }[] = [
  { label: "yearly", ms: 365 * DAY, season: null, tolerance: 5 * DAY },
  { label: "quarterly", ms: 91 * DAY, season: 4, tolerance: 5 * DAY },
  { label: "monthly", ms: 30 * DAY, season: 12, tolerance: 3 * DAY },
  { label: "weekly", ms: 7 * DAY, season: 52, tolerance: DAY / 2 },
  { label: "daily", ms: DAY, season: 7, tolerance: DAY / 4 },
  { label: "hourly", ms: 3_600_000, season: 24, tolerance: 60_000 },
];

/**
 * Parse a pasted block of text.
 *
 * Accepts a bare column of numbers, or two columns of `date,value` (comma, tab
 * or whitespace separated). Reusing `tabular/csv.ts` for the two-column case
 * would mean building a whole columnar dataset to read two columns of a paste,
 * so this reads the text directly — but the *file* path goes through that
 * parser, which is where quoting and ragged rows actually arise.
 */
export function parseSeries(text: string, name = "series"): Series {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length === 0) throw new Error("Paste a column of numbers, or date,value pairs.");

  // A header is a first line whose value column is not a number.
  const first = splitRow(lines[0]);
  const headerLikely = first.length > 0 && !isNumeric(first[first.length - 1]);
  const rows = headerLikely ? lines.slice(1) : lines;
  if (rows.length < 3) {
    throw new Error("Need at least three points to forecast anything.");
  }

  const values: number[] = [];
  const times: number[] = [];
  let anyTime = false;

  for (let i = 0; i < rows.length; i++) {
    const parts = splitRow(rows[i]);
    const raw = parts[parts.length - 1];
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      throw new Error(`Line ${i + (headerLikely ? 2 : 1)}: "${raw}" is not a number.`);
    }
    values.push(value);
    if (parts.length >= 2) {
      const t = Date.parse(parts[0]);
      if (Number.isFinite(t)) {
        times.push(t);
        anyTime = true;
      } else {
        times.push(NaN);
      }
    } else {
      times.push(NaN);
    }
  }

  return describe(
    name,
    Float32Array.from(values),
    anyTime && times.every((t) => Number.isFinite(t)) ? Float64Array.from(times) : null,
  );
}

function splitRow(line: string): string[] {
  return line.split(/[,\t;]|\s{2,}|\s(?=\S*$)/).map((p) => p.trim()).filter((p) => p.length > 0);
}

function isNumeric(s: string): boolean {
  return s.length > 0 && Number.isFinite(Number(s));
}

/**
 * Attach the frequency, gaps and regularity report to a bare pair of arrays.
 *
 * Exported because the bundled samples build their arrays directly and must go
 * through the same description — a sample that skipped this would be the one
 * series on the page whose gaps were never checked.
 */
export function describe(
  name: string,
  values: Float32Array,
  times: Float64Array | null,
): Series {
  if (!times || times.length < 2) {
    return {
      name,
      values,
      times,
      stepMs: null,
      frequency: null,
      suggestedSeason: null,
      gaps: [],
      irregular: false,
    };
  }

  const diffs: number[] = [];
  for (let i = 1; i < times.length; i++) diffs.push(times[i] - times[i - 1]);
  const step = modalStep(diffs);

  const match = FREQUENCIES.find((f) => Math.abs(step - f.ms) <= f.tolerance);
  const gaps: SeriesGap[] = [];
  let irregular = false;
  for (let i = 0; i < diffs.length; i++) {
    const ratio = diffs[i] / step;
    const rounded = Math.round(ratio);
    if (Math.abs(ratio - rounded) > 0.25) {
      irregular = true;
    } else if (rounded > 1) {
      gaps.push({ index: i, missing: rounded - 1 });
    } else if (rounded < 1) {
      irregular = true;
    }
  }

  return {
    name,
    values,
    times,
    stepMs: step,
    frequency: match?.label ?? null,
    suggestedSeason: match?.season ?? null,
    gaps,
    irregular,
  };
}

/**
 * The most common gap between consecutive points.
 *
 * The *mode*, not the mean and not the median. The mean is dragged by one long
 * gap far enough to make every ordinary interval look irregular — but so is the
 * median on a short series: three monthly points with one month missing have
 * two diffs, `[1 month, 2 months]`, whose upper median is the gap itself, and
 * then the page reports the *regular* interval as the anomaly. The mode is the
 * step that actually recurs, which is what "the implied frequency" means.
 */
function modalStep(diffs: number[]): number {
  const buckets = new Map<number, { total: number; count: number }>();
  for (const d of diffs) {
    if (d <= 0) continue;
    // Bucketed to three significant figures, so timestamps that wobble by a
    // second (or by a daylight-saving hour) still land together.
    const key = Number(d.toPrecision(3));
    const bucket = buckets.get(key);
    if (bucket) {
      bucket.total += d;
      bucket.count++;
    } else {
      buckets.set(key, { total: d, count: 1 });
    }
  }
  if (buckets.size === 0) return diffs[0] ?? 1;
  let best = 0;
  let bestCount = -1;
  for (const [key, { total, count }] of buckets) {
    // Ties go to the shorter step: a gap is a multiple of the real interval, so
    // where both occur equally often the smaller one is the frequency.
    if (count > bestCount || (count === bestCount && key < best)) {
      best = total / count;
      bestCount = count;
    }
  }
  return best;
}

/** A label for point `i` — a date where there is one, else its index. */
export function pointLabel(series: Series, i: number): string {
  if (!series.times) return String(i + 1);
  const d = new Date(series.times[i]);
  if (series.stepMs != null && series.stepMs >= 28 * DAY) {
    return d.toISOString().slice(0, 7);
  }
  return d.toISOString().slice(0, 10);
}

/** Extend a series' own time axis by `h` steps, for plotting the forecast. */
export function futureTimes(series: Series, h: number): Float64Array | null {
  if (!series.times || series.stepMs == null || series.times.length === 0) return null;
  const last = series.times[series.times.length - 1];
  const out = new Float64Array(h);
  for (let i = 0; i < h; i++) out[i] = last + (i + 1) * series.stepMs;
  return out;
}
