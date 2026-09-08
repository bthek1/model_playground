// Frame probabilities → speech segments. Pure, synchronous, and deliberately
// separate from the model: the page lets the user drag the threshold, and every
// drag re-derives segments from the *same* probabilities. Re-running a 2 MB
// recurrent model on every pointer move would be absurd, and would also make the
// timeline flicker while the worker caught up.

import { FRAME_SAMPLES } from "./vad";

export interface Segment {
  /** Seconds from the start of the clip. */
  start: number;
  end: number;
}

export interface SegmentOptions {
  /** Speech above this probability. */
  threshold?: number;
  /** Gaps shorter than this are bridged rather than splitting a segment. */
  minSilenceMs?: number;
  /** Runs shorter than this are dropped as blips. */
  minSpeechMs?: number;
  frameSamples?: number;
  sampleRate?: number;
}

export const DEFAULT_THRESHOLD = 0.5;

/**
 * Group frames above `threshold` into segments, bridging short gaps and
 * dropping short runs.
 *
 * The hangover is why this isn't just `probs.map(p => p > t)`: speech dips below
 * any threshold at a stop consonant, so a raw mask cuts "activity detection"
 * into four segments. Upstream silero uses a hysteresis band
 * (`neg_threshold = threshold - 0.15`) for the same reason; bridging by duration
 * is easier to explain on a page where the user is dragging the threshold and
 * watching the boundaries move.
 */
export function toSegments(
  probabilities: ArrayLike<number>,
  {
    threshold = DEFAULT_THRESHOLD,
    minSilenceMs = 100,
    minSpeechMs = 250,
    frameSamples = FRAME_SAMPLES,
    sampleRate = 16000,
  }: SegmentOptions = {},
): Segment[] {
  const frameMs = (frameSamples / sampleRate) * 1000;
  const bridge = Math.round(minSilenceMs / frameMs);
  const minFrames = Math.max(1, Math.round(minSpeechMs / frameMs));

  // Pass 1 — raw runs of above-threshold frames, as [firstFrame, lastFrame].
  const runs: [number, number][] = [];
  let start = -1;
  for (let f = 0; f < probabilities.length; f++) {
    const speech = probabilities[f] > threshold;
    if (speech && start < 0) start = f;
    // A run also ends at the last frame, which is why this can't be a plain
    // `else` — speech running to the end of the clip is the common case for a
    // mic take that was cut off mid-word.
    if (start >= 0 && (!speech || f === probabilities.length - 1)) {
      runs.push([start, speech ? f : f - 1]);
      start = -1;
    }
  }

  // Pass 2 — bridge gaps, then drop what is still too short.
  const merged: [number, number][] = [];
  for (const run of runs) {
    const last = merged[merged.length - 1];
    if (last && run[0] - last[1] - 1 <= bridge) last[1] = run[1];
    else merged.push([...run]);
  }

  const seconds = frameSamples / sampleRate;
  return merged
    .filter(([from, to]) => to - from + 1 >= minFrames)
    .map(([from, to]) => ({ start: from * seconds, end: (to + 1) * seconds }));
}

/** Share of the clip above `threshold`, 0–1. Drives the "62% speech" summary. */
export function speechFraction(
  probabilities: ArrayLike<number>,
  threshold = DEFAULT_THRESHOLD,
): number {
  if (probabilities.length === 0) return 0;
  let speech = 0;
  for (let f = 0; f < probabilities.length; f++) {
    if (probabilities[f] > threshold) speech++;
  }
  return speech / probabilities.length;
}

/** Total speech time in seconds across `segments`. */
export function speechSeconds(segments: readonly Segment[]): number {
  return segments.reduce((total, s) => total + (s.end - s.start), 0);
}
