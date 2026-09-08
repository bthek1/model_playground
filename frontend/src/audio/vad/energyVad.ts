// The zero-download baseline: short-time energy, no model at all.
//
// It ships beside Silero for two reasons. It answers "does this task actually
// need a neural network?" honestly — on a clean close-mic recording the answer
// is often no — and it gives the page something to run before any weights are
// fetched, which is the only route in the app where the empty state can do real
// work.
//
// It is a baseline, not a model. It keys off level alone, so a fan, a road or a
// second talker all read as speech; that failure is the point of having Silero
// next to it in the picker.

import { FRAME_SAMPLES } from "./vad";

/** Frames quieter than this below the clip's loud level are certainly silence. */
const FLOOR_DB = -60;

/** Root-mean-square level of a frame, in dBFS (−Infinity for digital silence). */
function frameDb(audio: Float32Array, at: number, length: number): number {
  let sum = 0;
  const end = Math.min(at + length, audio.length);
  for (let i = at; i < end; i++) sum += audio[i] * audio[i];
  const rms = Math.sqrt(sum / Math.max(1, end - at));
  return rms > 0 ? 20 * Math.log10(rms) : FLOOR_DB;
}

/**
 * Speech probability per frame from level alone.
 *
 * The scale is **relative to this clip**, not absolute: a quiet recording and a
 * loud one should both produce a usable timeline, and no fixed dBFS threshold
 * does that. The floor is the 10th percentile frame (the room tone) and the top
 * is the 95th (the loudest speech, ignoring clicks); everything between maps
 * linearly onto 0–1. A clip with no dynamic range at all — pure silence, or a
 * constant tone — collapses to zero rather than dividing by nothing.
 */
export function energyProbabilities(
  audio: Float32Array,
  frameSamples = FRAME_SAMPLES,
): Float32Array {
  const frames = Math.ceil(audio.length / frameSamples);
  const db = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    db[f] = Math.max(FLOOR_DB, frameDb(audio, f * frameSamples, frameSamples));
  }

  const sorted = Float32Array.from(db).sort();
  const floor = sorted[Math.floor(sorted.length * 0.1)] ?? FLOOR_DB;
  const peak = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] ?? 0;
  const span = peak - floor;

  const out = new Float32Array(frames);
  // 12 dB of range is about the gap between room tone and a spoken word. Below
  // that the clip has no speech/silence structure to find, and stretching what
  // little range there is would turn noise into confident detections.
  if (span < 12) return out;

  for (let f = 0; f < frames; f++) {
    out[f] = Math.min(1, Math.max(0, (db[f] - floor) / span));
  }
  return out;
}
