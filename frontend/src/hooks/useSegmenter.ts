// In-browser semantic segmentation. A thin wrapper over `useVisionPipeline`.
//
// The pipeline returns **one single-channel mask per class present** —
// `[{ label, score, mask }, …]` — not one indexed label map. A `RawImage` does
// not survive `postMessage` as a class, so each mask arrives as the plain object
// underneath it: `{ data, width, height, channels }`. That is exactly the shape
// `drawMasks` takes, which is why nothing is converted here.
//
// Toggling a class off and dragging the opacity slider are **pure derivations**
// over the masks already in hand. Neither re-runs the model, which is the whole
// reason the masks are kept rather than a pre-composited canvas.

import { useCallback, useMemo, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import {
  useVisionPipeline,
  type UseVisionPipelineResult,
} from "@/hooks/useVisionPipeline";
import type { LabelledMask } from "@/vision/draw";
import { DEFAULT_SEGMENTER, SEGMENTER_MODELS } from "@/vision/segmentation";

export interface SegmentMask {
  label: string;
  /** Mean confidence for the class. Null on models that don't report one. */
  score: number | null;
  mask: {
    data: Uint8ClampedArray | Uint8Array | number[];
    width: number;
    height: number;
    channels?: number;
  };
}

export interface UseSegmenterResult
  extends Omit<UseVisionPipelineResult, "run"> {
  result: SegmentMask[] | null;
  run: (
    image: RawImage,
    opts?: { consume?: boolean },
  ) => Promise<SegmentMask[]>;
}

export function useSegmenter(
  model: string = DEFAULT_SEGMENTER,
  autoLoad = false,
): UseSegmenterResult {
  const meta = useMemo(
    () => SEGMENTER_MODELS.find((m) => m.id === model) ?? SEGMENTER_MODELS[0],
    [model],
  );
  const pipe = useVisionPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<SegmentMask[] | null>(null);

  const run = useCallback(
    async (
      image: RawImage,
      opts?: { consume?: boolean },
    ): Promise<SegmentMask[]> => {
      const out = (await post(image, [], opts)) as SegmentMask[];
      setResult(out);
      return out;
    },
    [post],
  );

  return { ...pipe, result, run };
}

/**
 * How much of the picture each class covers, as a fraction. Sorted descending,
 * so the legend leads with the classes that actually matter — a 150-class model
 * routinely returns a dozen, most of them a rounding error.
 */
export function coverageOf(masks: readonly SegmentMask[]): {
  label: string;
  coverage: number;
}[] {
  return masks
    .map(({ label, mask }) => {
      const total = mask.width * mask.height || 1;
      let covered = 0;
      for (let i = 0; i < mask.data.length; i++) {
        const v = mask.data[i];
        if (v > (v > 1 ? 127 : 0.5)) covered++;
      }
      return { label, coverage: covered / total };
    })
    .sort((a, b) => b.coverage - a.coverage);
}

/**
 * The masks to composite, in the order `drawMasks` wants them.
 *
 * Later masks paint over earlier ones, so this returns **largest coverage
 * first**: the sky goes down before the lamp post, and a class covering 2% of
 * the frame is the one still visible at the end. Reversed, every small class
 * disappears under the background it sits in front of — and because the masks
 * are mostly disjoint, it would look almost right.
 */
export function visibleMasks(
  masks: readonly SegmentMask[] | null,
  hidden: ReadonlySet<string>,
): LabelledMask[] {
  if (!masks) return [];
  const order = coverageOf(masks);
  // Rank 0 is the largest class, and ascending rank is exactly paint order.
  const rank = new Map(order.map((o, i) => [o.label, i]));
  return masks
    .filter((m) => !hidden.has(m.label))
    .slice()
    .sort((a, b) => (rank.get(a.label) ?? 0) - (rank.get(b.label) ?? 0))
    .map((m) => ({
      label: m.label,
      data: m.mask.data,
      width: m.mask.width,
      height: m.mask.height,
    }));
}
