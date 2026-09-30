import { describe, expect, it } from "vitest";

import {
  ABS_SILENCE_DB,
  FLOOR_MIN_SECONDS,
  GATE_MARGIN_DB,
  sliceHasSpeech,
} from "./liveGate";

const RATE = 16000;

/** A 220 Hz tone at `db` dBFS RMS. */
function tone(seconds: number, db: number): Float32Array {
  const amp = Math.SQRT2 * 10 ** (db / 20);
  const out = new Float32Array(Math.round(seconds * RATE));
  for (let i = 0; i < out.length; i++) out[i] = amp * Math.sin((2 * Math.PI * 220 * i) / RATE);
  return out;
}

/** Deterministic white noise at `db` dBFS RMS. */
function noise(seconds: number, db: number, seed = 1): Float32Array {
  const out = new Float32Array(Math.round(seconds * RATE));
  let s = seed;
  // Uniform on ±0.5 has RMS 1/√12.
  const amp = 10 ** (db / 20) * Math.sqrt(12);
  for (let i = 0; i < out.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = amp * (s / 2 ** 32 - 0.5);
  }
  return out;
}

function concat(...parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

function mix(a: Float32Array, b: Float32Array): Float32Array {
  const out = a.slice();
  for (let i = 0; i < out.length; i++) out[i] += b[i];
  return out;
}

describe("sliceHasSpeech", () => {
  it("skips digital silence", () => {
    const d = sliceHasSpeech(new Float32Array(1.5 * RATE), 0);
    expect(d.speech).toBe(false);
    expect(d.activeMs).toBe(0);
  });

  it("hears a word on a short take, before a noise floor exists", () => {
    const d = sliceHasSpeech(tone(1.5, -20), 0);
    expect(d.floorDb).toBeNull();
    expect(d.thresholdDb).toBe(ABS_SILENCE_DB);
    expect(d.speech).toBe(true);
  });

  // Fail-open, measured: over a fan the first tick of a take has no floor yet,
  // so the gate transcribes it. That costs one pass; guessing a floor from 1.5 s
  // that might be all speech could cost the first word.
  it("fails open in a loud room until it has heard enough to estimate the floor", () => {
    const d = sliceHasSpeech(noise(1.5, -40), 0);
    expect(d.floorDb).toBeNull();
    expect(d.speech).toBe(true);
  });

  it("skips a noise-only slice once the floor is known, and hears speech over it", () => {
    const room = noise(FLOOR_MIN_SECONDS + 1.5, -40);
    const quiet = sliceHasSpeech(room, FLOOR_MIN_SECONDS * RATE);
    expect(quiet.floorDb).toBeGreaterThan(-45);
    expect(quiet.floorDb).toBeLessThan(-35);
    expect(quiet.thresholdDb).toBeCloseTo(quiet.floorDb! + GATE_MARGIN_DB, 5);
    expect(quiet.speech).toBe(false);

    const talking = mix(room, concat(new Float32Array(FLOOR_MIN_SECONDS * RATE), tone(1.5, -20)));
    expect(sliceHasSpeech(talking, FLOOR_MIN_SECONDS * RATE).speech).toBe(true);
  });

  // Rule 2 of the module: the energy VAD scales to the clip it is given, so a
  // slice with no dynamic range of its own — one sustained vowel — reads as zero
  // speech in isolation. Judged against the window's floor, it is plainly loud.
  it("judges the slice against the window, so a sustained vowel is not silence", () => {
    const window = concat(new Float32Array(3 * RATE), tone(1.5, -20));
    expect(sliceHasSpeech(window, 3 * RATE).speech).toBe(true);
  });

  it("ignores a click shorter than three frames", () => {
    const window = concat(new Float32Array(3 * RATE), tone(0.05, -10), new Float32Array(RATE));
    expect(sliceHasSpeech(window, 3 * RATE).speech).toBe(false);
  });

  // The known limit, pinned so it is a decision rather than a surprise: a sound
  // with no quiet part at all for the whole window is its own floor, and reads
  // as a steady noise. Measured, real continuous speech never does this — a
  // talker with no pause longer than JFK's, for 44 s over a fan, was heard on
  // 29 ticks of 29 (scripts/measure-live-asr-gate.mjs). A held note or a
  // whistle is skipped, and it is not speech.
  it("reads an unbroken tone filling the window as a steady noise, not speech", () => {
    expect(sliceHasSpeech(tone(4.5, -20), 3 * RATE).speech).toBe(false);
  });

  it("does not look before `from`", () => {
    const window = concat(tone(3, -20), new Float32Array(1.5 * RATE));
    expect(sliceHasSpeech(window, 0).speech).toBe(true);
    expect(sliceHasSpeech(window, 3 * RATE).speech).toBe(false);
  });

  it("treats an empty slice as nothing new", () => {
    const window = tone(2, -20);
    expect(sliceHasSpeech(window, window.length).speech).toBe(false);
    expect(sliceHasSpeech(window, window.length + 100).speech).toBe(false);
  });
});
