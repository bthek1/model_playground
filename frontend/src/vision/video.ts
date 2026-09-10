// Frame sampling for `/video-classification`.
//
// **Seeking, not playback.** A 30-second clip must not take 30 seconds to
// sample: setting `currentTime` and waiting for `seeked` decodes only the frames
// asked for, and a 4 fps pass over half a minute is a hundred-odd seeks rather
// than a real-time play-through. It is also the only way to sample a clip
// faster than 1x at all.
//
// Two caps, both stated in the UI. The *rate* (`fps`) is how densely the clip is
// looked at; the *count* (`maxFrames`) is the hard stop, because this is the page
// most likely to be handed a ten-minute video and every frame is a CLIP vision
// pass. Without the second cap a long clip is not slow, it is a hung tab.
//
// The pure half — which times to sample — is split out and unit-tested;
// everything that needs a real `<video>` is not, and the `@slow` spec covers it.

import { RawImage } from "@huggingface/transformers";

import { fitWithin } from "./image";

/** Frames per second to sample. More than 4 buys nothing a caption can read. */
export const DEFAULT_SAMPLE_FPS = 2;

/** The hard stop. 120 frames at 2 fps is a minute of video. */
export const MAX_FRAMES = 120;

/** Longest side of a sampled frame — resolution is the throttle, as ever. */
export const MAX_FRAME_SIDE = 448;

/**
 * The timestamps to sample, in seconds.
 *
 * Offset by half a step rather than starting at zero: the first frame of a clip
 * is very often a black frame or a fade, and a baseline whose first data point
 * is "an image of darkness" reads as a model failure rather than an editing
 * convention.
 *
 * Pure, so the cap and the arithmetic are testable without a browser.
 */
export function frameTimes(
  duration: number,
  fps: number,
  maxFrames: number,
): number[] {
  if (!(duration > 0) || !(fps > 0) || maxFrames < 1) return [];
  const wanted = Math.min(Math.ceil(duration * fps), Math.floor(maxFrames));
  const step = 1 / fps;
  const times: number[] = [];
  for (let i = 0; i < wanted; i++) {
    const t = (i + 0.5) * step;
    if (t >= duration) break;
    times.push(t);
  }
  // A clip shorter than one step still deserves one frame.
  if (times.length === 0) times.push(duration / 2);
  return times;
}

/** True when the frame count was cut short by the cap rather than the clip. */
export function wasCapped(
  duration: number,
  fps: number,
  maxFrames: number,
): boolean {
  return Math.ceil(duration * fps) > maxFrames;
}

export interface SampledFrame {
  time: number;
  image: RawImage;
}

export interface SampleOptions {
  fps?: number;
  maxFrames?: number;
  maxSide?: number;
  /** Called as each frame is decoded, for the progress line. */
  onFrame?: (frame: SampledFrame, index: number, total: number) => void;
  /** Return true to stop early — the page's Cancel. */
  cancelled?: () => boolean;
}

/**
 * Decode a clip's frames at a fixed rate.
 *
 * `src` is either an object URL for a `File` the user picked or a bundled sample
 * URL. `crossOrigin` is set **before** `src`, because a cross-origin video
 * assigned first is fetched without CORS and then taints the canvas — and a
 * tainted canvas fails at `getImageData` with a security error, several steps
 * away from the cause.
 */
export async function sampleVideo(
  src: string,
  {
    fps = DEFAULT_SAMPLE_FPS,
    maxFrames = MAX_FRAMES,
    maxSide = MAX_FRAME_SIDE,
    onFrame,
    cancelled,
  }: SampleOptions = {},
): Promise<{ frames: SampledFrame[]; duration: number; capped: boolean }> {
  const video = document.createElement("video");
  video.crossOrigin = "anonymous"; // before `src` — see above
  video.preload = "auto";
  video.muted = true;
  video.playsInline = true;
  video.src = src;

  try {
    await once(video, "loadedmetadata");
    const duration = video.duration;
    if (!Number.isFinite(duration) || duration <= 0) {
      throw new Error("Could not read the clip's duration");
    }

    const times = frameTimes(duration, fps, maxFrames);
    const target = fitWithin(video.videoWidth, video.videoHeight, maxSide);
    const canvas = document.createElement("canvas");
    canvas.width = target.width;
    canvas.height = target.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Could not get a 2D context to sample frames");

    const frames: SampledFrame[] = [];
    for (let i = 0; i < times.length; i++) {
      if (cancelled?.()) break;
      video.currentTime = times[i];
      await once(video, "seeked");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frame = { time: times[i], image: RawImage.fromCanvas(canvas) };
      frames.push(frame);
      onFrame?.(frame, i, times.length);
    }

    return { frames, duration, capped: wasCapped(duration, fps, maxFrames) };
  } finally {
    // Release the decoder and the buffered data. A `<video>` left with a `src`
    // holds its whole decoded pipeline for as long as the element is reachable.
    video.removeAttribute("src");
    video.load();
  }
}

/** A frame as a data URL, for the filmstrip. */
export function thumbnail(image: RawImage): string {
  try {
    return image.toCanvas().toDataURL("image/jpeg", 0.7);
  } catch {
    // A canvas the browser has starved of memory, or a test environment with no
    // canvas at all. A filmstrip is a convenience; a route that throws is not.
    return "";
  }
}

/** Resolve on the next `event`, or reject if the element errors first. */
function once(video: HTMLVideoElement, event: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const done = () => {
      video.removeEventListener(event, done);
      video.removeEventListener("error", fail);
      resolve();
    };
    const fail = () => {
      video.removeEventListener(event, done);
      video.removeEventListener("error", fail);
      reject(new Error("Could not decode that video"));
    };
    video.addEventListener(event, done, { once: true });
    video.addEventListener("error", fail, { once: true });
  });
}

// --- The bundled clips --------------------------------------------------------

export interface VideoSample {
  id: string;
  label: string;
  url: string;
  hint: string;
  /** Labels that make this clip interesting to score. Bare nouns — templated. */
  labels: readonly string[];
}

const BASE =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main";

/**
 * Short clips, so the page works with no upload.
 *
 * Each carries the label set it is worth scoring against, because a zero-shot
 * score is relative to the labels given — a clip with no plausible distractor in
 * the list demonstrates nothing.
 */
export const VIDEO_SAMPLES: VideoSample[] = [
  {
    id: "interview",
    label: "Interview",
    url: `${BASE}/interview.mp4`,
    hint: "Two people talking to camera — barely moves, so a frame-level model does well",
    labels: ["an interview", "a football match", "a car chase"],
  },
  {
    id: "courtroom",
    label: "Courtroom",
    url: `${BASE}/courtroom.mp4`,
    hint: "A wide indoor scene — the distinctive thing is the room, not the motion",
    labels: ["a courtroom", "a kitchen", "a beach"],
  },
];
