// The live-ASR gate: "did anyone say anything since the last tick?"
//
// `useLiveAsr` re-transcribes the tail of the take every time MediaRecorder
// emits a chunk. Without a gate that is a full Whisper pass per 1.5 s of
// silence, and Whisper is not neutral about silence — it is prone to
// transcribing room tone as "Thank you." or "you". This module decides, per
// tick, whether the audio that arrived since the last scored tick holds speech.
//
// It is the energy baseline's arithmetic (`energyVad.ts`), not Silero, and that
// is a measured decision rather than a shortcut (docs/roadmaps/audio.md §3.6,
// scripts/measure-live-asr-gate.mjs). On the question a *gate* asks — any speech
// in this 1.5 s, yes or no — it skipped 25 of the 28 silent ticks Silero
// skipped across two rooms, never skipped a tick Silero heard speech on, and
// heard all 29 ticks of a 44 s talker who never paused. Silero would recover the
// other three wasted passes for a second model live beside the ASR one, its own
// LOAD and its own worker.
//
// Its known limit: a sound with no quiet part for the whole window (a held note,
// a whistle) is its own floor and reads as steady noise. That is not speech, and
// real continuous speech — syllable gaps included — never does it.
//
// Two rules make it safe to put in front of a transcriber:
//
//   1. **It fails open.** Every uncertainty resolves to "speech": a take too
//      short to have a noise-floor estimate, a room too loud to find structure
//      in, a scoring error in the caller. A skipped word is worse than a wasted
//      pass, so the gate can only ever skip too little.
//   2. **The new slice is judged against the window, never alone.** The energy
//      VAD's scale is relative to the clip it is given, and a 1.5 s slice of
//      nothing but a sustained vowel — or nothing but silence — has no dynamic
//      range, which `energyProbabilities` reads as zero speech. Here the noise
//      floor comes from the whole window the transcriber will see, so a slice
//      is loud or quiet *relative to this room*.

import { FRAME_SAMPLES } from "./vad";

/** Frames quieter than this are silence whatever the room sounds like. */
export const ABS_SILENCE_DB = -55;
/**
 * How far above the room's floor a frame must be to count as activity. The same
 * 12 dB `energyVad.ts` uses as the gap between room tone and a spoken word.
 */
export const GATE_MARGIN_DB = 12;
/** Active time a slice needs before it counts as speech: three 32 ms frames. */
export const MIN_ACTIVE_MS = 96;
/**
 * Below this much audio there is no floor estimate to trust, so the gate falls
 * back to the absolute threshold alone: only near-digital silence is skipped.
 * Over a fan that means the first tick of a take is transcribed (measured) —
 * one wasted pass, against guessing a floor from 1.5 s that may be all speech.
 */
export const FLOOR_MIN_SECONDS = 3;

export interface GateDecision {
  /** True → transcribe. Always true when the gate cannot decide. */
  speech: boolean;
  /** Milliseconds of the new slice above the activity threshold. */
  activeMs: number;
  /** The level a frame had to beat, in dBFS. */
  thresholdDb: number;
  /** The room's floor (10th-percentile frame of the window), or null if not yet estimable. */
  floorDb: number | null;
}

function frameDb(audio: Float32Array, at: number, length: number): number {
  let sum = 0;
  const end = Math.min(at + length, audio.length);
  for (let i = at; i < end; i++) sum += audio[i] * audio[i];
  const rms = Math.sqrt(sum / Math.max(1, end - at));
  return rms > 0 ? 20 * Math.log10(rms) : -Infinity;
}

/**
 * Does `window[from…]` hold speech?
 *
 * `window` is the audio the transcriber would be handed (the tail of the take);
 * `from` is the sample index inside it where the unscored audio begins. Frames
 * are aligned to `from`, so the verdict does not depend on where the window
 * happened to start.
 */
export function sliceHasSpeech(
  window: Float32Array,
  from: number,
  sampleRate = 16000,
): GateDecision {
  const start = Math.max(0, Math.min(from, window.length));

  let floorDb: number | null = null;
  if (window.length >= FLOOR_MIN_SECONDS * sampleRate) {
    const frames = Math.floor(window.length / FRAME_SAMPLES);
    const db = new Float32Array(frames);
    for (let f = 0; f < frames; f++) {
      db[f] = Math.max(-120, frameDb(window, f * FRAME_SAMPLES, FRAME_SAMPLES));
    }
    db.sort();
    floorDb = db[Math.floor(frames * 0.1)];
  }

  const thresholdDb =
    floorDb == null ? ABS_SILENCE_DB : Math.max(ABS_SILENCE_DB, floorDb + GATE_MARGIN_DB);

  let active = 0;
  for (let at = start; at < window.length; at += FRAME_SAMPLES) {
    if (frameDb(window, at, FRAME_SAMPLES) > thresholdDb) active++;
  }
  const activeMs = (active * FRAME_SAMPLES * 1000) / sampleRate;

  return { speech: activeMs >= MIN_ACTIVE_MS, activeMs, thresholdDb, floorDb };
}
