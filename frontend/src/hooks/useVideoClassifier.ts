// Video classification, as a frame-level baseline.
//
// It reuses `/zero-shot-image-classification`'s worker wholesale — the CLIP
// catalogue, the prompt template, and above all the **text-embedding cache**.
// That cache is the reason this page is usable: the label embeddings are
// constant across every frame of a clip, so they are computed once per label
// edit rather than once per frame. On a 60-frame clip that is the difference
// between scoring in seconds and scoring in a minute, and it is "encode once,
// decode many" applied a second time on the same page.
//
// Three things this hook owns:
//
//  1. **One request in flight at a time**, in a plain `for` loop rather than a
//     fan-out. Two overlapping calls into one ONNX session is not a guarantee
//     worth relying on, and a fan-out would queue anyway while making the
//     per-frame progress meaningless.
//  2. **Cancellation mid-clip.** This is the only page whose run is minutes
//     long, so abandoning it has to be possible — and has to leave no pending
//     request behind.
//  3. **The scores stay unpooled.** Pooling is a pure derivation in `pool.ts`,
//     so the window slider re-derives the chart without re-scoring the clip.

import { useCallback, useMemo, useRef, useState } from "react";

import { useZeroShotImage } from "@/hooks/useZeroShotImage";
import {
  sampleVideo,
  thumbnail,
  type SampleOptions,
} from "@/vision/video";

export interface FrameScores {
  /** Seconds into the clip. */
  time: number;
  /** One score per label, in the label list's own order. */
  scores: number[];
  /** A data URL for the filmstrip, so a spike is traceable to a frame. */
  thumb: string;
}

export interface ClipScores {
  /** The labels these scores belong to, captured when the run started. */
  labels: string[];
  frames: FrameScores[];
  /** Clip length in seconds. */
  duration: number;
  /** True when the frame cap cut the clip short. */
  capped: boolean;
}

export interface ClipProgress {
  phase: "sampling" | "scoring";
  done: number;
  total: number;
}

export interface UseVideoClassifierResult {
  status: ReturnType<typeof useZeroShotImage>["status"];
  idle: boolean;
  loading: boolean;
  ready: boolean;
  progress: ReturnType<typeof useZeroShotImage>["progress"];
  loadProgress: ReturnType<typeof useZeroShotImage>["loadProgress"];
  loadedInMs: number | null;
  backend: string | null;
  running: boolean;
  error: string | null;
  result: ClipScores | null;
  /** Non-null while a clip is being sampled or scored. */
  clipProgress: ClipProgress | null;
  /** Sample `src` and score every frame against `labels`. */
  run: (
    src: string,
    labels: readonly string[],
    template: string,
    opts?: Pick<SampleOptions, "fps" | "maxFrames">,
  ) => Promise<ClipScores>;
  /** Abandon a run mid-clip. */
  stop: () => void;
  load: () => void;
  retry: (overrides?: Record<string, unknown>) => void;
  cancel: () => void;
}

export function useVideoClassifier(
  model?: string,
  autoLoad = false,
): UseVideoClassifierResult {
  const clip = useZeroShotImage(model, autoLoad);
  const { run: score } = clip;

  const [result, setResult] = useState<ClipScores | null>(null);
  const [clipProgress, setClipProgress] = useState<ClipProgress | null>(null);
  const cancelled = useRef(false);

  const stop = useCallback(() => {
    cancelled.current = true;
  }, []);

  const run = useCallback(
    async (
      src: string,
      labels: readonly string[],
      template: string,
      opts?: Pick<SampleOptions, "fps" | "maxFrames">,
    ): Promise<ClipScores> => {
      const clean = labels.map((l) => l.trim()).filter(Boolean);
      if (clean.length === 0) throw new Error("Add at least one label to score");

      cancelled.current = false;
      setResult(null);
      setClipProgress({ phase: "sampling", done: 0, total: 0 });

      try {
        const { frames, duration, capped } = await sampleVideo(src, {
          ...opts,
          cancelled: () => cancelled.current,
          onFrame: (_frame, index, total) =>
            setClipProgress({ phase: "sampling", done: index + 1, total }),
        });

        const scored: FrameScores[] = [];
        for (let i = 0; i < frames.length; i++) {
          if (cancelled.current) break;
          setClipProgress({
            phase: "scoring",
            done: i,
            total: frames.length,
          });
          // One template, always: the side-by-side prompt comparison belongs to
          // the still-image page, and doubling the per-frame cost for a
          // comparison nobody can read across sixty frames buys nothing.
          const [answer] = await score(frames[i].image, clean, [template]);
          scored.push({
            time: frames[i].time,
            scores: answer.scores.map((s) => s.score),
            thumb: thumbnail(frames[i].image),
          });
        }

        const out: ClipScores = {
          labels: clean,
          frames: scored,
          duration,
          capped,
        };
        setResult(out);
        return out;
      } finally {
        setClipProgress(null);
      }
    },
    [score],
  );

  const running = clip.running || clipProgress !== null;

  return useMemo(
    () => ({
      status: clip.status,
      idle: clip.idle,
      loading: clip.loading,
      ready: clip.ready,
      progress: clip.progress,
      loadProgress: clip.loadProgress,
      loadedInMs: clip.loadedInMs,
      backend: clip.backend,
      running,
      error: clip.error,
      result,
      clipProgress,
      run,
      stop,
      load: clip.load,
      retry: clip.retry,
      cancel: clip.cancel,
    }),
    [clip, running, result, clipProgress, run, stop],
  );
}
