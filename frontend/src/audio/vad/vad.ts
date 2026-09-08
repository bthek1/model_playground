// The Silero-VAD frame loop — the one genuinely new thing in this route.
//
// Every other audio task hands a whole clip to a model and gets a whole answer
// back. VAD is *recurrent*: the graph scores one 32 ms frame at a time and hands
// back a state tensor that must be fed into the next call. Two consequences, and
// both are silent failures if you get them wrong:
//
//   1. **The window is 576 samples, not 512.** Silero v5 expects the 64 samples
//      preceding the frame to be prepended as context (this mirrors
//      `OnnxWrapper.__call__` in the upstream `silero-vad` package). The graph's
//      input dimensions are dynamic, so feeding a bare 512 samples runs happily
//      and returns *plausible-looking numbers that never fire*. Measured on
//      jfk.wav: with the context the model reads 0.005 on the lead-in silence
//      and 0.83–0.99 on speech; without it, a strong tone scores 0.05 and the
//      page looks like a broken model rather than a broken frame loop.
//   2. **State is per take.** `state` starts as zeros and `context` as silence
//      for every clip. Carrying either across takes leaks the previous clip's
//      tail into the first frames of the next one.
//
// The loop is deliberately not the ONNX session's business: `session.ts` supplies
// a single-frame `FrameInfer` and this module owns the windowing, so the whole
// contract above is unit-testable with a fake infer and no model download.

/** Samples scored per step. 512 @ 16 kHz = 32 ms. Fixed by the model. */
export const FRAME_SAMPLES = 512;

/** Samples of history prepended to each frame. Fixed by the model. */
export const CONTEXT_SAMPLES = 64;

/** What the graph actually receives: `[context | frame]`. */
export const WINDOW_SAMPLES = CONTEXT_SAMPLES + FRAME_SAMPLES;

/** Silero v5 carries one `[2, 1, 128]` state tensor — v4's separate h/c are gone. */
export const STATE_SIZE = 2 * 1 * 128;

/** Score one window, returning the speech probability and the next state. */
export type FrameInfer = (
  window: Float32Array,
  state: Float32Array<ArrayBufferLike>,
) => Promise<{ probability: number; state: Float32Array<ArrayBufferLike> }>;

/**
 * Speech probability per 32 ms frame, for a whole clip.
 *
 * A trailing partial frame is zero-padded rather than dropped — upstream drops
 * it, but the page draws this array *over* the waveform, and a timeline that
 * stops up to 31 ms short of the clip reads as a bug in the last segment's end
 * time. One extra frame of silence costs nothing.
 */
export async function frameProbabilities(
  audio: Float32Array,
  infer: FrameInfer,
): Promise<Float32Array> {
  const frames = Math.ceil(audio.length / FRAME_SAMPLES);
  const probabilities = new Float32Array(frames);
  if (frames === 0) return probabilities;

  // Explicitly `ArrayBufferLike`: the state comes back out of an ONNX tensor,
  // whose data may be backed by a SharedArrayBuffer when ORT runs threaded.
  let state: Float32Array<ArrayBufferLike> = new Float32Array(STATE_SIZE);
  // One scratch window, refilled per frame: `[0, 64)` is the previous frame's
  // tail, `[64, 576)` the current frame.
  const window = new Float32Array(WINDOW_SAMPLES);

  for (let f = 0; f < frames; f++) {
    const at = f * FRAME_SAMPLES;
    const frame = audio.subarray(at, Math.min(at + FRAME_SAMPLES, audio.length));

    // Shift the previous frame's last 64 samples down into the context slot,
    // then lay the new frame after it (zero-filling a short final frame).
    window.copyWithin(0, FRAME_SAMPLES, WINDOW_SAMPLES);
    window.fill(0, CONTEXT_SAMPLES);
    window.set(frame, CONTEXT_SAMPLES);

    const out = await infer(window, state);
    probabilities[f] = out.probability;
    state = out.state;
  }

  return probabilities;
}
