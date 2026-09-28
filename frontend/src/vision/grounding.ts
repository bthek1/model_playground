// `/robotics` — grounding an instruction in what the camera sees.
//
// **This module adds no checkpoint.** Both halves are catalogue entries the app
// already ships and already measures: OWLv2 from `/zero-shot-object-detection`
// finds the thing the user named, and Depth Anything V2 Small from `/depth` says
// how far away it is. The pair is composed here *from those entries*, never
// restated — so the ids, the dtypes and above all the measured `bytes` cannot
// drift from the pages that own them. `grounding.test.ts` asserts the combined
// size is the sum of the two sources for the same reason: the combined figure is
// also `combineProgress`'s denominator, and a stale constant there is a progress
// bar that is wrong with nothing failing.
//
//   OWLv2 base/16 ensemble      fp16 307,904,711 · q8 155,312,754
//   Depth Anything V2 Small     fp16  49,642,442 · q8  27,258,801
//   combined                    fp16 357,547,153 · q8 182,571,555
//
// Past `LARGE_MODEL_BYTES` (200 MiB) on WebGPU, so the picker's warning fires;
// under `HEAVY_MODEL_BYTES` (500 MB) on both, so there is no second opt-in.
// Both directions are pinned by a test.
//
// **One entry for the pair, not two selectors** — `/pose`'s user-facing rule:
// the pair is what the user chooses, and a size quoted per half is a size that
// can disagree with itself. But not `/pose`'s *shape*: pose loads its pair in
// one worker; here each half rides its own `useVisionPipeline`, so each gets its
// own worker. That is `/text-ranking`'s shape, and `hooks/useGrounding.ts` says
// what follows from it.
//
// **The picker is a list of `RoboticsEntry`, discriminated on `kind`,** so a
// second kind of entry (a behaviour-cloning policy, trained in the tab, with no
// download at all) can sit beside this one in the same SELECT band. Only the
// `grounding` kind exists today; the seam is the union and nothing else.

import type { Backend } from "@/model/backend";
import type { MeasuredBytes } from "@/model/size";

import { DEPTH_MODELS, type DepthModel } from "./depth";
import {
  ZERO_SHOT_DETECTOR_MODELS,
  type ZeroShotDetectorModel,
} from "./zeroShotDetection";

/** A detector + depth pair: one LOAD, two workers, one combined size. */
export interface GroundingEntry {
  kind: "grounding";
  /** Composite id — the pair is what the user picks, not either half. */
  id: string;
  label: string;
  hint: string;
  /** Combined parameter count in millions, for the shared `ModelPicker`. */
  params: number;
  /** **Combined** measured download, summed from the two source entries. */
  bytes: MeasuredBytes;
  /** The `/zero-shot-object-detection` entry, by reference. */
  detector: ZeroShotDetectorModel;
  /** The `/depth` entry, by reference. Its `metric` flag sets the direction. */
  depth: DepthModel;
  /** What the detector wants typed — carried up so the route can say so. */
  queries: ZeroShotDetectorModel["queries"];
  /**
   * Where the *pair* can run: the intersection of what the two halves declare,
   * so a restriction written on either source entry reaches this picker
   * without being restated here. Undefined when neither half declares one.
   */
  backends?: readonly Backend[];
}

/**
 * Everything the `/robotics` picker can offer. A union on `kind`, with one
 * member today; a later entry that downloads nothing joins it here.
 */
export type RoboticsEntry = GroundingEntry;

function sumBytes(a?: MeasuredBytes, b?: MeasuredBytes): MeasuredBytes {
  return {
    webgpu: (a?.webgpu ?? 0) + (b?.webgpu ?? 0),
    wasm: (a?.wasm ?? 0) + (b?.wasm ?? 0),
  };
}

/** Both halves must run on a backend for the pair to. */
function bothRun(
  a?: readonly Backend[],
  b?: readonly Backend[],
): readonly Backend[] | undefined {
  if (!a && !b) return undefined;
  if (!a) return b;
  if (!b) return a;
  return a.filter((x) => b.includes(x));
}

function find<T extends { id: string }>(list: readonly T[], id: string): T {
  const hit = list.find((m) => m.id === id);
  // Loud at import time rather than a pair silently built from `undefined`:
  // renaming either source entry must break this module, not the page.
  if (!hit) throw new Error(`grounding: no catalogue entry for ${id}`);
  return hit;
}

/** Compose a pair from the two shipped entries. Exported for the test. */
export function groundingPair(
  detector: ZeroShotDetectorModel,
  depth: DepthModel,
  over: { id: string; label: string; hint: string },
): GroundingEntry {
  return {
    kind: "grounding",
    ...over,
    params: detector.params + depth.params,
    bytes: sumBytes(detector.bytes, depth.bytes),
    detector,
    depth,
    queries: detector.queries,
    backends: bothRun(detector.backends, depth.backends),
  };
}

export const GROUNDING_DETECTOR_ID = "Xenova/owlv2-base-patch16-ensemble";
export const GROUNDING_DEPTH_ID = "onnx-community/depth-anything-v2-small";

export const GROUNDING_PAIR: GroundingEntry = groundingPair(
  find(ZERO_SHOT_DETECTOR_MODELS, GROUNDING_DETECTOR_ID),
  find(DEPTH_MODELS, GROUNDING_DEPTH_ID),
  {
    id: "owlv2-base+depth-anything-v2-small",
    label: "OWLv2 base + Depth Anything V2 Small",
    hint: "Find what you named, then say how near it is. Both models already ship on their own pages — nothing new to download.",
  },
);

export const ROBOTICS_ENTRIES: RoboticsEntry[] = [GROUNDING_PAIR];

export const DEFAULT_ROBOTICS_ENTRY = ROBOTICS_ENTRIES[0].id;

/** The grounding entries alone — what `useGrounding` resolves an id against. */
export const GROUNDING_ENTRIES: GroundingEntry[] = ROBOTICS_ENTRIES.filter(
  (e): e is GroundingEntry => e.kind === "grounding",
);

/**
 * Starting phrases, chosen for the bundled city-street sample, which has cars
 * and bicycles at clearly different distances. Short noun phrases, because
 * OWLv2 scores each one as a class name rather than reading it as a sentence.
 */
export const DEFAULT_GROUNDING_QUERIES = ["a car", "a bicycle"];

/**
 * A phrase longer than this is probably an instruction, not a noun phrase —
 * "pick up the red block nearest the camera" — and OWLv2 will score it as one
 * enormous class name. The route says so beside the phrase rather than letting
 * the query quietly underperform.
 */
export const MAX_PHRASE_WORDS = 4;

/** Phrases that read like sentences. Pure; the route renders the warning. */
export function sentenceLike(queries: readonly string[]): string[] {
  return queries.filter(
    (q) => q.trim().split(/\s+/).filter(Boolean).length > MAX_PHRASE_WORDS,
  );
}
