// The coordinate round-trip, which is the whole correctness surface of `/pose`.
//
// Top-down pose is two models: a detector finds people, then the pose model runs
// on **each person's crop**. That means every keypoint comes back in the crop's
// own pixel space and has to be put back into the source image's. Get it wrong
// and the skeleton floats beside the person — plausible, drawn confidently, and
// invisible to any count-based test. So this file is pure, and asserted against
// hand-computed values.
//
// The specific trap is in `VitPoseImageProcessor.post_process_pose_estimation`:
//
//     const xScale = bbox.at(-2) / width;      // box *width*  / heatmap width
//     const keypoint = [(xScale * xWeightedSum) / sum, …];
//
// It scales the heatmap peak by the box's **size** and never adds the box's
// **origin**. Pass the whole image as the box (as the single-person example
// does) and the origin is (0, 0), so the omission is invisible. Pass a crop and
// every joint is offset by the crop's top-left corner — which is exactly the
// multi-person case this route is built for.

import type { Detection } from "../draw";
import type { Keypoint, Person } from "./skeleton";

/** A crop rectangle in source pixels, COCO-style: `[x, y, width, height]`. */
export type CocoBox = [number, number, number, number];

/**
 * The people worth running the pose model on.
 *
 * Filtered to the **person** class before cropping: running a pose model on a
 * car crop is wasted work and a confusing overlay, and on a street scene most
 * detections are not people. Sorted by confidence and capped, because each
 * extra person is another forward pass.
 */
export function personBoxes(
  detections: readonly Detection[] | null,
  { threshold, maxPeople }: { threshold: number; maxPeople: number },
): Detection[] {
  return (detections ?? [])
    .filter((d) => d.label === "person" && d.score >= threshold)
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(0, maxPeople));
}

/**
 * A detection's box as an integer crop rectangle, clamped to the image.
 *
 * **Clamped, not wrapped.** A detector routinely returns a box that runs off the
 * edge — a person half out of frame is still a person — and `RawImage.crop`
 * with a negative origin or an over-wide extent produces a garbage buffer
 * rather than an error. A degenerate box (zero width after clamping) is
 * rejected by returning `null`, so the caller drops that person instead of
 * feeding the pose model an empty image.
 */
export function cropBox(
  box: Detection["box"],
  imageWidth: number,
  imageHeight: number,
): CocoBox | null {
  const x0 = clamp(Math.floor(box.xmin), 0, imageWidth);
  const y0 = clamp(Math.floor(box.ymin), 0, imageHeight);
  const x1 = clamp(Math.ceil(box.xmax), 0, imageWidth);
  const y1 = clamp(Math.ceil(box.ymax), 0, imageHeight);
  const width = x1 - x0;
  const height = y1 - y0;
  if (width < 1 || height < 1) return null;
  return [x0, y0, width, height];
}

/**
 * Move one person's keypoints from crop-local pixels into source pixels.
 *
 * `post_process_pose_estimation` has already scaled the heatmap peak by the
 * box's width and height; what it has *not* done is add the box's origin. This
 * is that addition, and it is the bug the `@slow` spec's anatomical assertion
 * exists to catch.
 */
export function toSourceKeypoints(
  keypoints: readonly (readonly [number, number])[],
  scores: readonly number[],
  labels: readonly number[],
  box: CocoBox,
): Keypoint[] {
  const [ox, oy] = box;
  return keypoints.map((point, i) => ({
    x: point[0] + ox,
    y: point[1] + oy,
    score: scores[i] ?? 0,
    index: labels[i] ?? i,
  }));
}

/**
 * Scale a person — box and joints together — from the resolution the models ran
 * at back to the resolution the canvas is drawn at.
 *
 * The same trip `scaleDetections` makes for `/object-detection`, and it has to
 * cover the keypoints too: scaling the box but not the joints is a skeleton that
 * shrinks away from its own outline as the source gets larger.
 */
export function scalePeople(people: readonly Person[], scale: number): Person[] {
  if (scale === 1) return [...people];
  return people.map((person) => ({
    score: person.score,
    box: {
      xmin: person.box.xmin * scale,
      ymin: person.box.ymin * scale,
      xmax: person.box.xmax * scale,
      ymax: person.box.ymax * scale,
    },
    keypoints: person.keypoints.map((k) => ({
      ...k,
      x: k.x * scale,
      y: k.y * scale,
    })),
  }));
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, value));
}
