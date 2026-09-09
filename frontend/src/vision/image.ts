// Browser image I/O — the client-side equivalent of the notebooks' `PIL.Image`
// plus `cv2.VideoCapture`. The vision counterpart of `audio/io.ts`, and the
// module every vision route imports.
//
// `RawImage` is Transformers.js's own image type, accepted anywhere a pipeline
// takes an image, so decoding is its job rather than ours.
//
// **Do not resize or normalise for the model here.** Every pipeline calls
// `AutoProcessor`, which reads `preprocessor_config.json` from the model's own
// repo and applies that model's resize, crop, rescale and normalise. That file
// *is* the published input contract. Reimplementing it by hand is the single
// most common way to end up with a model that runs, produces plausible-looking
// output, and is quietly wrong — which is exactly how the two DSP bugs in
// `audio/enhance/` shipped past a green test suite, and that route at least had
// the excuse that its model publishes no processor at all.
//
// `downscale` below is the one deliberate exception, and it is not preprocessing:
// it caps the *source* resolution before the processor ever sees the frame,
// because a detector at 1280x720 costs roughly 4x the same detector at 640x480
// (docs/roadmaps/vision.md §5 — resolution is the throttle, not the model).

import { RawImage } from "@huggingface/transformers";

/** Decode a file the user picked or dropped. */
export async function fromFile(file: File): Promise<RawImage> {
  return RawImage.fromBlob(file);
}

/** Fetch and decode an image by URL — the bundled samples come through here. */
export async function fromUrl(url: string): Promise<RawImage> {
  return RawImage.fromURL(url);
}

/**
 * Grab the current frame out of a live `<video>`, for the camera demos. The
 * browser analogue of a single `cv2.VideoCapture.read()`.
 */
export function fromVideo(video: HTMLVideoElement): RawImage {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get a 2D context to capture the frame");
  ctx.drawImage(video, 0, 0);
  return RawImage.fromCanvas(canvas);
}

export interface CameraOptions {
  width?: number;
  height?: number;
  /** `"user"` is the selfie camera, `"environment"` the rear one on a phone. */
  facingMode?: "user" | "environment";
}

/**
 * Open the camera into `video` and start playing. Returns a **stop function** —
 * call it on unmount, always. A leaked `MediaStream` leaves the webcam light on
 * after the user has navigated away, which is the one bug in this file a user
 * notices from across the room.
 */
export async function openCamera(
  video: HTMLVideoElement,
  { width = 640, height = 480, facingMode = "user" }: CameraOptions = {},
): Promise<() => void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: width }, height: { ideal: height }, facingMode },
  });
  video.srcObject = stream;
  await video.play();
  return () => {
    stream.getTracks().forEach((track) => track.stop());
    if (video.srcObject === stream) video.srcObject = null;
  };
}

/**
 * Target dimensions that fit `width`x`height` inside `maxSide` without changing
 * the aspect ratio. Returns the input untouched when it already fits — pure, so
 * the arithmetic is testable without a canvas.
 */
export function fitWithin(
  width: number,
  height: number,
  maxSide: number,
): { width: number; height: number } {
  const longest = Math.max(width, height);
  if (longest <= maxSide) return { width, height };
  const scale = maxSide / longest;
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

/**
 * Cap an image's longest side at `maxSide`. A no-op (same object, no canvas
 * round-trip) when it already fits.
 */
export async function downscale(
  image: RawImage,
  maxSide: number,
): Promise<RawImage> {
  const target = fitWithin(image.width, image.height, maxSide);
  if (target.width === image.width && target.height === image.height) {
    return image;
  }
  return image.resize(target.width, target.height);
}

// --- Worker transport --------------------------------------------------------
//
// A `RawImage` is a class instance, so it does not survive `postMessage`: the
// structured clone arrives as a plain object with no methods, and the pipeline
// rejects it. Send the pixels plus the three numbers that describe them, and
// rebuild the instance on the other side. Settled here, once, so no route has to
// discover it.

export interface ImagePayload {
  data: Uint8ClampedArray | Uint8Array;
  width: number;
  height: number;
  channels: 1 | 2 | 3 | 4;
}

/**
 * Flatten a `RawImage` for `postMessage`.
 *
 * `copy` defaults to **true**, and that default is the point: the page is
 * usually still showing the very image it just sent, and transferring the
 * underlying buffer detaches it, leaving a blank preview. Pass `copy: false`
 * only for a frame the main thread is finished with — a webcam grab, say — and
 * transfer it with {@link transferablesOf}.
 */
export function toPayload(
  image: RawImage,
  { copy = true }: { copy?: boolean } = {},
): ImagePayload {
  return {
    data: copy ? image.data.slice() : image.data,
    width: image.width,
    height: image.height,
    channels: image.channels,
  };
}

/** Rebuild a `RawImage` inside the worker. The inverse of {@link toPayload}. */
export function fromPayload(payload: ImagePayload): RawImage {
  return new RawImage(
    payload.data,
    payload.width,
    payload.height,
    payload.channels,
  );
}

/** The transfer list for a payload, for the caller that wants a zero-copy post. */
export function transferablesOf(payload: ImagePayload): Transferable[] {
  return [payload.data.buffer as ArrayBuffer];
}
