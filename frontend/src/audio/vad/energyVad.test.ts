import { describe, expect, it } from "vitest";

import { energyProbabilities } from "./energyVad";
import { speechFraction, toSegments } from "./segments";
import { FRAME_SAMPLES } from "./vad";

function noise(frames: number, amplitude: number): Float32Array {
  const out = new Float32Array(frames * FRAME_SAMPLES);
  for (let i = 0; i < out.length; i++) {
    // Deterministic pseudo-noise — a seeded ramp through sin, not Math.random,
    // so a failure is reproducible.
    out[i] = Math.sin(i * 12.9898) * amplitude;
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

describe("energyProbabilities", () => {
  it("returns one probability per frame, matching the model path", () => {
    const probs = energyProbabilities(new Float32Array(FRAME_SAMPLES * 4 + 7));
    expect(probs).toHaveLength(5);
  });

  it("scores loud frames above quiet ones", () => {
    const clip = concat(noise(10, 0.001), noise(10, 0.3), noise(10, 0.001));
    const probs = energyProbabilities(clip);
    expect(probs[15]).toBeGreaterThan(0.8);
    expect(probs[2]).toBeLessThan(0.2);
  });

  it("produces segments over the loud stretch", () => {
    const clip = concat(noise(20, 0.001), noise(20, 0.3), noise(20, 0.001));
    const segments = toSegments(energyProbabilities(clip), {
      frameSamples: FRAME_SAMPLES,
      sampleRate: 16000,
    });
    expect(segments).toHaveLength(1);
    // The loud stretch is frames 20–39, i.e. 0.64 s to 1.28 s.
    expect(segments[0].start).toBeCloseTo(0.64, 1);
    expect(segments[0].end).toBeCloseTo(1.28, 1);
  });

  it("finds nothing in a clip with no dynamic range", () => {
    // A constant tone has energy everywhere. Stretching its tiny spread would
    // manufacture confident detections out of nothing.
    expect(speechFraction(energyProbabilities(noise(30, 0.2)))).toBe(0);
  });

  it("finds nothing in digital silence, without dividing by zero", () => {
    const probs = energyProbabilities(new Float32Array(FRAME_SAMPLES * 10));
    expect(Array.from(probs).every((p) => p === 0)).toBe(true);
  });

  it("is level-relative: the same clip scaled down scores the same", () => {
    const clip = concat(noise(10, 0.001), noise(10, 0.3));
    const quiet = Float32Array.from(clip, (v) => v * 0.05);
    const loud = energyProbabilities(clip);
    const soft = energyProbabilities(quiet);
    for (let f = 0; f < loud.length; f++) {
      expect(soft[f]).toBeCloseTo(loud[f], 4);
    }
  });
});
