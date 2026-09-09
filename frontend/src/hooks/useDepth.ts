// In-browser depth estimation. A thin wrapper over `useVisionPipeline`: it picks
// the catalogue entry, types the result, and adds nothing else — worker
// lifecycle, the pending table and the two state machines belong to
// `model/useModelWorker.ts` and are not re-derived here.
//
// The pipeline hands back **both** representations, and both are kept:
//
//   `predicted_depth` — the raw tensor. Arbitrary scale, so it is the only one
//                       you can take a range off, and the only one worth
//                       asserting on in a test.
//   `depth`           — an already-normalised single-channel image, for a page
//                       that just wants a picture.
//
// The page chooses. `/depth` draws from the tensor through `drawHeatmap`, which
// normalises per frame — without that a relative-depth map renders uniformly
// black or white and the page looks broken rather than wrong.

import { useCallback, useMemo, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import {
  useVisionPipeline,
  type UseVisionPipelineResult,
} from "@/hooks/useVisionPipeline";
import { DEFAULT_DEPTH_MODEL, DEPTH_MODELS } from "@/vision/depth";

/** The tensor half of the pipeline's output, as it crosses the worker boundary. */
export interface DepthTensor {
  data: Float32Array | number[];
  /** `[1, h, w]` or `[h, w]` — the pipeline is not consistent across models. */
  dims: number[];
}

export interface DepthResult {
  /** Raw relative depth. Larger is *nearer* for the Depth Anything family. */
  predicted_depth: DepthTensor;
  /** The same map, normalised to 0–255 by the pipeline. */
  depth?: { data: Uint8ClampedArray | number[]; width: number; height: number };
}

export interface UseDepthResult extends Omit<UseVisionPipelineResult, "run"> {
  result: DepthResult | null;
  run: (image: RawImage, opts?: { consume?: boolean }) => Promise<DepthResult>;
}

/**
 * Width and height of a depth map, from the tensor's own dims. `[1, h, w]` and
 * `[h, w]` both occur, and reading them the wrong way round transposes the map
 * into diagonal streaks rather than failing.
 */
export function depthDims(tensor: DepthTensor): {
  width: number;
  height: number;
} {
  const dims = tensor.dims ?? [];
  const [height, width] = dims.slice(-2);
  return { width: width ?? 0, height: height ?? 0 };
}

export function useDepth(
  model: string = DEFAULT_DEPTH_MODEL,
  autoLoad = false,
): UseDepthResult {
  const meta = useMemo(
    () => DEPTH_MODELS.find((m) => m.id === model) ?? DEPTH_MODELS[0],
    [model],
  );
  const pipe = useVisionPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<DepthResult | null>(null);

  const run = useCallback(
    async (
      image: RawImage,
      opts?: { consume?: boolean },
    ): Promise<DepthResult> => {
      const out = (await post(image, [], opts)) as DepthResult;
      setResult(out);
      return out;
    },
    [post],
  );

  return { ...pipe, result, run };
}
