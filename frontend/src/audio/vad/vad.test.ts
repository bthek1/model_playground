import { describe, expect, it, vi } from "vitest";

import {
  CONTEXT_SAMPLES,
  FRAME_SAMPLES,
  STATE_SIZE,
  WINDOW_SAMPLES,
  frameProbabilities,
  type FrameInfer,
} from "./vad";

/** A fake session that records every window and state it was handed. */
function recorder(probability: (frame: number) => number = () => 0) {
  const windows: Float32Array[] = [];
  const states: Float32Array[] = [];
  const infer = vi.fn<FrameInfer>(async (window, state) => {
    // The loop reuses one scratch window, so copy before the next frame
    // overwrites it — a test that kept the reference would assert on the last
    // frame N times and pass regardless of what the loop did.
    windows.push(window.slice());
    states.push(state.slice());
    const next = new Float32Array(STATE_SIZE);
    next.fill(windows.length); // a state the next call can be checked against
    return { probability: probability(windows.length - 1), state: next };
  });
  return { infer, windows, states };
}

describe("frameProbabilities", () => {
  it("scores one frame per 32 ms, zero-padding a partial tail", async () => {
    const { infer } = recorder();
    // Two whole frames plus 100 samples.
    const probs = await frameProbabilities(
      new Float32Array(FRAME_SAMPLES * 2 + 100),
      infer,
    );
    expect(probs).toHaveLength(3);
    expect(infer).toHaveBeenCalledTimes(3);
  });

  it("feeds the model 576 samples — 64 of context plus the 512-sample frame", async () => {
    // This is the contract that fails *silently* when it is wrong: a bare 512
    // window runs fine and returns plausible numbers that never fire.
    const { infer, windows } = recorder();
    const audio = new Float32Array(FRAME_SAMPLES * 3);
    audio.fill(1); // frame 0
    audio.fill(2, FRAME_SAMPLES); // frame 1
    audio.fill(3, FRAME_SAMPLES * 2); // frame 2
    await frameProbabilities(audio, infer);

    expect(windows).toHaveLength(3);
    for (const w of windows) expect(w).toHaveLength(WINDOW_SAMPLES);

    // First frame: silence for context, because there is no history yet.
    expect(Array.from(windows[0].subarray(0, CONTEXT_SAMPLES))).toEqual(
      Array(CONTEXT_SAMPLES).fill(0),
    );
    expect(windows[0][CONTEXT_SAMPLES]).toBe(1);

    // Later frames: the context is the *previous* frame's last 64 samples.
    expect(windows[1][0]).toBe(1);
    expect(windows[1][CONTEXT_SAMPLES - 1]).toBe(1);
    expect(windows[1][CONTEXT_SAMPLES]).toBe(2);
    expect(windows[2][0]).toBe(2);
    expect(windows[2][CONTEXT_SAMPLES]).toBe(3);
  });

  it("carries the model's state from each frame into the next", async () => {
    const { infer, states } = recorder();
    await frameProbabilities(new Float32Array(FRAME_SAMPLES * 3), infer);

    // Frame 0 starts from zeros; every later frame gets what the previous call
    // returned. Losing this makes the model stateless and its output noise.
    expect(states[0].every((v) => v === 0)).toBe(true);
    expect(states[1][0]).toBe(1);
    expect(states[2][0]).toBe(2);
  });

  it("resets state and context per clip, so a take is independent", async () => {
    const first = recorder();
    const audio = new Float32Array(FRAME_SAMPLES * 2);
    audio.fill(0.5);
    await frameProbabilities(audio, first.infer);

    const second = recorder();
    await frameProbabilities(audio, second.infer);

    expect(second.states[0].every((v) => v === 0)).toBe(true);
    expect(Array.from(second.windows[0].subarray(0, CONTEXT_SAMPLES))).toEqual(
      Array(CONTEXT_SAMPLES).fill(0),
    );
    // Same audio in, same windows out — no leakage from the earlier take.
    expect(Array.from(second.windows[0])).toEqual(Array.from(first.windows[0]));
  });

  it("zero-fills a short final frame rather than repeating the previous one", async () => {
    const { infer, windows } = recorder();
    const audio = new Float32Array(FRAME_SAMPLES + 10);
    audio.fill(1);
    await frameProbabilities(audio, infer);

    const tail = windows[1];
    expect(tail[CONTEXT_SAMPLES + 9]).toBe(1);
    expect(tail[CONTEXT_SAMPLES + 10]).toBe(0);
    expect(tail[WINDOW_SAMPLES - 1]).toBe(0);
  });

  it("returns the probability of each frame, in order", async () => {
    const { infer } = recorder((f) => (f + 1) / 10);
    const probs = await frameProbabilities(
      new Float32Array(FRAME_SAMPLES * 3),
      infer,
    );
    expect(Array.from(probs).map((p) => +p.toFixed(2))).toEqual([0.1, 0.2, 0.3]);
  });

  it("handles an empty clip without calling the model", async () => {
    const { infer } = recorder();
    const probs = await frameProbabilities(new Float32Array(0), infer);
    expect(probs).toHaveLength(0);
    expect(infer).not.toHaveBeenCalled();
  });
});
