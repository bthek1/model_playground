// Florence-2's grounded answers, converted to the app's `Detection` shape.
//
// A separate module from the worker, and pure, because the coordinate handling
// here is the part most likely to be quietly wrong and the worker cannot be
// unit-tested without the runtime.
//
// `post_process_generation` already maps the model's discretised `<loc_…>` bins
// back to pixels, so what arrives is `{ labels: string[], bboxes: number[][] }`
// with each box `[x1, y1, x2, y2]` in **source-image pixels** — which is what
// `drawBoxes` wants. There is no `percentage` flag on this path and nothing to
// scale back, unlike `/object-detection`.

import type { Detection } from "../draw";

/** What `post_process_generation` returns for a box-producing task. */
export interface GroundedAnswer {
  labels?: string[];
  bboxes?: number[][];
}

/**
 * Turn `{ labels, bboxes }` into detections.
 *
 * Tolerant on purpose: a generative model can emit a box with a missing label
 * (the regex duplicates the previous one, and there may not be one), and a
 * truncated generation can end mid-box. Dropping a malformed box is right;
 * throwing would lose the eleven good ones with it.
 */
export function toDetections(answer: unknown): Detection[] {
  const { labels = [], bboxes = [] } = (answer ?? {}) as GroundedAnswer;
  const out: Detection[] = [];
  for (let i = 0; i < bboxes.length; i++) {
    const box = bboxes[i];
    if (!Array.isArray(box) || box.length < 4) continue;
    if (box.some((n) => !Number.isFinite(n))) continue;
    out.push({
      label: labels[i]?.trim() || "object",
      // Florence-2 emits a location, not a confidence, so there is no score to
      // report. 1 is honest in a way 0 would not be: every box it returns is
      // one it committed to, and a 0 would be filtered out by anything
      // downstream that thresholds.
      score: 1,
      box: { xmin: box[0], ymin: box[1], xmax: box[2], ymax: box[3] },
    });
  }
  return out;
}
