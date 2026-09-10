// In-browser super-resolution. A thin wrapper over `useVisionPipeline` plus the
// one thing this task genuinely needs that no other vision route does: it drives
// **many inferences per run**.
//
// A photo is cut into overlapping tiles (`vision/tile.ts`) and each tile is a
// separate `run` through the shared worker. That stays main-thread orchestration
// on purpose — the worker protocol keeps one message per inference, and adding a
// "run these thirty patches" message would put the tiling geometry inside the
// worker where it cannot be unit-tested. The loop is sequential rather than
// parallel because it already saturates the backend; posting thirty at once
// would only enlarge the peak memory the tiling exists to avoid.
//
// Two consequences for the shared contract, both additive:
//
//   `tiles`  — `{ done, total }` while a run is in flight. Derived from the loop
//              rather than from a new worker message, so it costs no protocol.
//   `stop()` — abandons the remaining tiles. Distinct from `cancel()`, which
//              belongs to the load state machine and abandons the *download*.
//              A run that is thirty inferences long needs a way out that does
//              not throw away the weights.

import { useCallback, useMemo, useRef, useState } from "react";
import type { RawImage } from "@huggingface/transformers";

import {
  useVisionPipeline,
  type UseVisionPipelineResult,
} from "@/hooks/useVisionPipeline";
import type { PixelBuffer, Pixels } from "@/vision/draw";
import { fromPayload, toPayload } from "@/vision/image";
import {
  DEFAULT_SUPER_RES_MODEL,
  SUPER_RES_MODELS,
  type SuperResModel,
} from "@/vision/superRes";
import {
  blendTile,
  createTileCanvas,
  cropTile,
  finishCanvas,
  planTiles,
} from "@/vision/tile";

/** Progress through one run's tiles. Null when nothing is running. */
export interface TileProgress {
  done: number;
  total: number;
}

/** Raised by `run` when `stop()` was called part-way through. */
export class RunCancelled extends Error {
  constructor() {
    super("Upscale cancelled");
    this.name = "RunCancelled";
  }
}

export interface UseSuperResResult extends Omit<UseVisionPipelineResult, "run"> {
  result: PixelBuffer | null;
  /** The source the current `result` was produced from, for the comparison. */
  source: Pixels | null;
  run: (image: RawImage) => Promise<PixelBuffer>;
  /** Abandon the tiles still queued. Leaves the model loaded. */
  stop: () => void;
  tiles: TileProgress | null;
  meta: SuperResModel;
}

export function useSuperRes(
  model: string = DEFAULT_SUPER_RES_MODEL,
  autoLoad = false,
): UseSuperResResult {
  const meta = useMemo(
    () => SUPER_RES_MODELS.find((m) => m.id === model) ?? SUPER_RES_MODELS[0],
    [model],
  );
  const pipe = useVisionPipeline(meta.task, meta.id, autoLoad, meta.dtypes);
  const { run: post } = pipe;

  const [result, setResult] = useState<PixelBuffer | null>(null);
  const [source, setSource] = useState<Pixels | null>(null);
  const [tiles, setTiles] = useState<TileProgress | null>(null);

  // A ref, not state: the loop reads it between tiles and must see the newest
  // value without waiting for a re-render. A state flag would let one more tile
  // through after Stop, which on WASM is a couple of seconds of nothing
  // happening after the user has asked for it to stop.
  const cancelled = useRef(false);

  const stop = useCallback(() => {
    cancelled.current = true;
  }, []);

  const run = useCallback(
    async (image: RawImage): Promise<PixelBuffer> => {
      cancelled.current = false;

      const src: Pixels = toPayload(image);
      const scale = meta.scale;
      const plan = planTiles(src.width, src.height);
      const canvas = createTileCanvas(
        src.width * scale,
        src.height * scale,
        src.channels,
      );

      setSource(src);
      setResult(null);
      setTiles({ done: 0, total: plan.tiles.length });

      try {
        for (const [i, tile] of plan.tiles.entries()) {
          if (cancelled.current) throw new RunCancelled();

          const patch = cropTile(src, tile);
          // `consume: true`: the patch is a buffer this loop just cut and will
          // never look at again, so transferring it saves a copy per tile —
          // which at thirty tiles is thirty copies of a 192x192 image.
          const out = (await post(fromPayload(patch), [], {
            consume: true,
          })) as Pixels;

          blendTile(canvas, out, tile, {
            scale,
            overlap: plan.overlap,
            source: src,
          });
          setTiles({ done: i + 1, total: plan.tiles.length });
        }

        const finished = finishCanvas(canvas);
        setResult(finished);
        return finished;
      } finally {
        setTiles(null);
      }
    },
    [post, meta.scale],
  );

  return {
    ...pipe,
    result,
    source,
    run,
    stop,
    tiles,
    meta,
    // **`running` means the whole tile sequence, not one inference.** The
    // pipeline's own count drops to zero between tiles, so taking it at face
    // value would flicker the spinner and re-enable the transport thirty times
    // during a single upscale.
    running: pipe.running || tiles != null,
  };
}
