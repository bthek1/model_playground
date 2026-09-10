// In-browser background removal. A thin wrapper over `useVisionPipeline` — it
// picks the catalogue entry, types the result, and adds nothing else. Worker
// lifecycle, the pending table and the two state machines belong to
// `model/useModelWorker.ts` and are not re-derived here.
//
// The pipeline returns **one RGBA image**, not a mask and an image: the matte is
// already in the alpha channel (`BackgroundRemovalPipeline` is the segmentation
// pipeline plus `putAlpha`). So there is nothing to pair up and nothing to
// re-align — everything the page draws, `vision/matte.ts` derives from this one
// buffer.

import { useCallback, useMemo, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import {
  useVisionPipeline,
  type UseVisionPipelineResult,
} from "@/hooks/useVisionPipeline";
import {
  DEFAULT_MATTE_MODEL,
  MATTE_MODELS,
  type MatteModel,
} from "@/vision/backgroundRemoval";
import type { RgbaImage } from "@/vision/matte";

export interface UseBackgroundRemovalResult
  extends Omit<UseVisionPipelineResult, "run"> {
  /** The cut-out: source pixels with the soft matte in the alpha channel. */
  result: RgbaImage | null;
  run: (image: RawImage, opts?: { consume?: boolean }) => Promise<RgbaImage>;
  /** The selected catalogue entry, so the route can read its licence. */
  meta: MatteModel;
}

export function useBackgroundRemoval(
  model: string = DEFAULT_MATTE_MODEL,
  autoLoad = false,
): UseBackgroundRemovalResult {
  const meta = useMemo(
    () => MATTE_MODELS.find((m) => m.id === model) ?? MATTE_MODELS[0],
    [model],
  );
  const pipe = useVisionPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<RgbaImage | null>(null);

  const run = useCallback(
    async (
      image: RawImage,
      opts?: { consume?: boolean },
    ): Promise<RgbaImage> => {
      const out = (await post(image, [], opts)) as RgbaImage;
      setResult(out);
      return out;
    },
    [post],
  );

  return { ...pipe, result, run, meta };
}
