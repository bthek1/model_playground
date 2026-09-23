// Picking a clip, for `/video-text-to-text` — the third of the pick hooks, after
// `useImagePick` and `useAudioPick`, and it owes the same two rules.
//
// **Exactly one object URL alive at a time, and none after unmount.** A preview
// URL that outlives its `<video>` pins the whole decoded pipeline for the tab's
// lifetime, and a clip is a much bigger buffer to leave pinned than a still.
//
// **Picking is not running.** Choosing a clip, changing the frame count and
// flipping the frame order are all INPUT: they change what the next GENERATE
// will send, and they spend nothing (model-page-pattern.md §1.6).
//
// What it adds over the other two is a **cache**, because the decode here is not
// a one-shot: sampling frames is a seek per frame, and pressing GENERATE twice
// on the same clip at the same frame count must cost one decode and two
// inferences. The cache is keyed on exactly the two things that change which
// frames come out — the source and the count — so changing either is a new
// decode and changing neither is not.
//
// It does not own the frame *order*. Reversing is a property of the run, not of
// the decode (`multimodal/frames.ts`), and a hook that cached reversed frames
// would decode the same clip twice to produce the same pictures.

import { useCallback, useEffect, useRef, useState } from "react";

import { MAX_FRAME_SIDE, uniformFrameTimes } from "@/multimodal/frames";
import { sampleVideo, type SampledFrame, type VideoSample } from "@/vision/video";

export interface PickedVideo {
  /** Object URL for an upload, or the bundled sample's own URL. */
  url: string;
  /** True when `url` is ours to revoke. */
  owned: boolean;
  name: string;
}

export interface SampleProgress {
  done: number;
  total: number;
}

export interface UseVideoPickResult {
  picked: PickedVideo | null;
  /** Frame extraction in flight, for the line under the preview. Null when idle. */
  sampling: SampleProgress | null;
  /** A failed decode or fetch. Belongs to the RUN slot, never to OUTPUT. */
  error: string | null;
  clearError: () => void;
  pickFile: (file: File | undefined) => void;
  pickSample: (sample: VideoSample) => void;
  /**
   * `count` frames spread evenly over the held clip, decoding only if this
   * exact (clip, count) pair has not been decoded already.
   *
   * Async because the first call for a pair really does seek the video. The
   * frames are handed back as they are — `toPayload` copies before a worker can
   * detach anything, so the caller's array stays usable for the filmstrip.
   */
  take: (count: number) => Promise<SampledFrame[]>;
}

export function useVideoPick(): UseVideoPickResult {
  const [picked, setPicked] = useState<PickedVideo | null>(null);
  const [sampling, setSampling] = useState<SampleProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const ownedUrl = useRef<string | null>(null);
  useEffect(
    () => () => {
      if (ownedUrl.current) URL.revokeObjectURL(ownedUrl.current);
    },
    [],
  );

  // One entry, not a map: two clips' worth of decoded frames is tens of MB of
  // pixels, and the page only ever asks about the clip it is showing.
  const cache = useRef<{ key: string; frames: SampledFrame[] } | null>(null);

  const adopt = useCallback((next: PickedVideo | null) => {
    if (ownedUrl.current) URL.revokeObjectURL(ownedUrl.current);
    ownedUrl.current = next?.owned ? next.url : null;
    cache.current = null;
    setError(null);
    setPicked(next);
  }, []);

  const pickFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      adopt({ url: URL.createObjectURL(file), owned: true, name: file.name });
    },
    [adopt],
  );

  const pickSample = useCallback(
    (sample: VideoSample) =>
      adopt({ url: sample.url, owned: false, name: sample.label }),
    [adopt],
  );

  const take = useCallback(
    async (count: number): Promise<SampledFrame[]> => {
      const source = picked;
      if (!source) return [];
      const key = `${source.url}|${count}`;
      if (cache.current?.key === key) return cache.current.frames;

      setError(null);
      setSampling({ done: 0, total: count });
      try {
        // The times come from `multimodal/frames.ts` and the decode from
        // `vision/video.ts` — one frame-extraction path in this repo, with an
        // option rather than a copy. A function, because the spacing depends on
        // the clip's duration and only `sampleVideo` has read it by then.
        const { frames } = await sampleVideo(source.url, {
          maxSide: MAX_FRAME_SIDE,
          times: (duration) => uniformFrameTimes(duration, count),
          onFrame: (_frame, index, total) =>
            setSampling({ done: index + 1, total }),
        });
        cache.current = { key, frames };
        return frames;
      } catch (e) {
        // A clip that will not decode is an **input** failure: it belongs beside
        // the file that caused it in RUN, not in OUTPUT where the model's own
        // failures go. Returning an empty list rather than throwing is what lets
        // the caller bail without a try/catch of its own.
        setError(e instanceof Error ? e.message : String(e));
        return [];
      } finally {
        setSampling(null);
      }
    },
    [picked],
  );

  const clearError = useCallback(() => setError(null), []);

  return {
    picked,
    sampling,
    error,
    clearError,
    pickFile,
    pickSample,
    take,
  };
}
