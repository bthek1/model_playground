// Frame selection for `/video-text-to-text`.
//
// **A video model in a tab is a frame sampler plus an image model**, and this
// file is the sampler half. Pure, for the same reason `vision/tile.ts` is: the
// geometry is the part that can be wrong while everything still looks like it
// works, and it must be testable without a `<video>` element.
//
// It is deliberately *not* `vision/video.ts`'s `frameTimes`, and the difference
// is the whole design. That function samples at a fixed **rate** (fps) with a
// hard cap, which is right for `/video-classification`: every frame is an
// independent CLIP pass, the chart is a time series, and a longer clip should
// produce more points. Here the frames go into **one prompt** — each carries its
// own image tokens through the encoder and they are all in context together — so
// the cost is the *count*, and the count is what the user sets. Eight frames
// must mean eight frames whether the clip is six seconds or six minutes.
//
// The decode itself is still `vision/video.ts`'s `sampleVideo`. There is one
// frame-extraction path in this repo and this page did not get a second one; it
// grew a `times` option instead.

/** What the slider offers. Four is enough to see motion; the cap is the honest part. */
export const MIN_FRAMES = 1;
export const DEFAULT_FRAMES = 4;

/**
 * The hard stop, and it is not a round number either.
 *
 * Each frame at 512px is one tile and `processor_config.json` sets
 * `image_seq_len: 64`, so N frames cost 64N image tokens **before** the question
 * is appended, and every one of them is attended over for every generated token.
 * Eight is 512 image tokens through a 256M decoder — already the slowest run in
 * the app. The model's own `preprocessor_config.json` allows 64
 * (`video_sampling.max_frames`); that is a number for a server.
 */
export const MAX_FRAMES = 8;

/**
 * Longest side of a sampled frame. **512, and it is the model's own number** —
 * `video_sampling.video_size.longest_edge` in `preprocessor_config.json`, which
 * happens to equal `max_image_size.longest_edge`.
 *
 * It is quoted from the video config rather than inherited from
 * `MAX_INFERENCE_SIDE` on purpose: they agree today, and if the checkpoint ever
 * disagreed with itself this page would need the video number. Inheriting a
 * sibling's hyperparameter is a mistake this repo has already made once.
 *
 * Getting it wrong is not a slow page, it is a hung one. Above 512 the processor
 * splits each frame into a grid of tiles **plus** a global view — a 640x360 frame
 * becomes five tiles — so the tile problem multiplies by the frame count rather
 * than adding to it.
 */
export const MAX_FRAME_SIDE = 512;

/**
 * `count` timestamps spread evenly over a clip of `duration` seconds.
 *
 * Sampled at the **centre of each of `count` equal slices** rather than at the
 * endpoints: `0` is very often a black frame or a fade, and `duration` is past
 * the last decodable frame on a good many encodes. Both failures look like the
 * model ignoring the video.
 *
 * Returns fewer than `count` only when there is nothing to return — a clip with
 * no duration. A one-frame request lands in the middle of the clip, which is the
 * single most representative frame available without looking at any of them.
 */
export function uniformFrameTimes(duration: number, count: number): number[] {
  if (!(duration > 0)) return [];
  const n = Math.max(1, Math.min(Math.floor(count), MAX_FRAMES));
  const slice = duration / n;
  const times: number[] = [];
  for (let i = 0; i < n; i++) times.push((i + 0.5) * slice);
  return times;
}

/**
 * The frames in the order they will be sent.
 *
 * The reverse is the page's experiment, not a detail of it: feeding the frames
 * backwards and watching the answer *not* change is how a small video VLM shows
 * that it is describing a picture rather than reading a sequence. So it is a
 * real reordering of the list that reaches the model — it costs an inference,
 * and the page says so — never a relabelling of a result already in hand.
 */
export function orderFrames<T>(frames: readonly T[], reversed: boolean): T[] {
  // A copy either way: the caller keeps its own array for the filmstrip, and
  // `Array.prototype.reverse` is in place.
  return reversed ? [...frames].reverse() : [...frames];
}
