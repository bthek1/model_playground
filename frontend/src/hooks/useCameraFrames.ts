// Camera → model, for every live vision page. Three pieces that always travel
// together, and one of them is easy to forget:
//
//   `useCamera`     opens the stream and, above all, **closes it** — a leaked
//                   MediaStream leaves the webcam light on after the user has
//                   navigated away, which is the one bug here a user notices
//                   from across the room.
//   `useLiveFrames` grabs the next frame only once the previous result is back.
//                   Never a queue: a rAF loop that posts every frame drifts
//                   seconds behind the picture and looks like a slow model.
//   `downscale`     caps the source resolution before the processor sees it.
//                   Not preprocessing — resolution is the throttle, and a
//                   detector at 1280x720 costs roughly 4x the same detector at
//                   640x480.
//
// The `<video>` is tracked as state rather than a ref object because `useCamera`
// has to re-run when the element appears: a ref's `.current` is null on the
// render that mounts it, and nothing re-renders when it fills in.

import { useCallback, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import { useCamera, useLiveFrames } from "@/hooks/useLiveFrames";
import { downscale, fromVideo, openCamera } from "@/vision/image";

export interface UseCameraFramesResult {
  /** Put this on the `<video>`. */
  videoRef: (el: HTMLVideoElement | null) => void;
  /** A denied permission or a missing device. Belongs to the RUN slot. */
  error: string | null;
  /** Measured end-to-end frames per second — the model's real rate, not the rAF rate. */
  fps: number | null;
  busy: boolean;
}

export function useCameraFrames({
  active,
  maxSide,
  onFrame,
}: {
  active: boolean;
  maxSide: number;
  /** Handle one downscaled frame. Awaited — that is what keeps one in flight. */
  onFrame: (frame: RawImage) => Promise<void>;
}): UseCameraFramesResult {
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const { error } = useCamera(video, active, (el) => openCamera(el));

  const handle = useCallback(
    async (el: HTMLVideoElement) => {
      // The first few animation frames land before the stream has dimensions;
      // capturing then yields a 0x0 canvas and a pipeline error per frame.
      if (!el.videoWidth || !el.videoHeight) return;
      await onFrame(await downscale(fromVideo(el), maxSide));
    },
    [onFrame, maxSide],
  );

  const { fps, busy } = useLiveFrames({ video, active, onFrame: handle });

  return { videoRef: setVideo, error, fps, busy };
}
