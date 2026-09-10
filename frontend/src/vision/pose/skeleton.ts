// The COCO 17-keypoint skeleton: the joint names, the edges between them, and
// the painter that draws one person.
//
// Pure and free of the runtime, so the edge list is testable and the route does
// not carry seventeen magic indices. Kept out of `vision/draw.ts` on purpose:
// that file holds the three *generic* overlay forms every vision task shares
// (boxes, a single-channel map, class masks), and a skeleton is neither generic
// nor meaningful without this specific joint ordering.

/** COCO's 17 keypoints, in the order the model emits them. Index is the label. */
export const COCO_KEYPOINTS = [
  "nose",
  "left eye",
  "right eye",
  "left ear",
  "right ear",
  "left shoulder",
  "right shoulder",
  "left elbow",
  "right elbow",
  "left wrist",
  "right wrist",
  "left hip",
  "right hip",
  "left knee",
  "right knee",
  "left ankle",
  "right ankle",
] as const;

export type KeypointName = (typeof COCO_KEYPOINTS)[number];

/**
 * The limbs, as index pairs into {@link COCO_KEYPOINTS}.
 *
 * The standard COCO skeleton, minus the face-to-shoulder edges that make a
 * low-confidence ear look like a broken neck. Order matters only for painting.
 */
export const COCO_EDGES: readonly (readonly [number, number])[] = [
  // face
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  // torso
  [5, 6],
  [5, 11],
  [6, 12],
  [11, 12],
  // arms
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  // legs
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

/** One joint, in source-image pixels, with the model's confidence in it. */
export interface Keypoint {
  x: number;
  y: number;
  score: number;
  /** Index into {@link COCO_KEYPOINTS}. */
  index: number;
}

/** One detected person: their box, their joints, and the detector's score. */
export interface Person {
  /** The detector's box, in source pixels. */
  box: { xmin: number; ymin: number; xmax: number; ymax: number };
  /** Detector confidence that this is a person. */
  score: number;
  keypoints: Keypoint[];
}

/**
 * Below this, a joint is drawn faintly rather than confidently.
 *
 * A heatmap always has a maximum somewhere, so **every** keypoint comes back
 * with a position whether or not the joint is in the picture: an occluded ankle
 * is not absent, it is a confident-looking guess. Dimming by confidence is the
 * difference between showing uncertainty and drawing a lie.
 */
export const JOINT_CONFIDENCE = 0.3;

/** Opacity for a joint or limb, from its confidence. Never fully invisible. */
export function jointAlpha(score: number): number {
  if (score >= JOINT_CONFIDENCE) return 1;
  // Linear down to a floor: a joint at 0 is still faintly there, because
  // "the model put it here and does not believe it" is information.
  return 0.15 + 0.85 * Math.max(0, score / JOINT_CONFIDENCE);
}

/**
 * Paint one person's skeleton over a canvas already holding the source pixels.
 *
 * A limb is drawn at the *lower* of its two joints' confidences: an edge is only
 * as believable as its weakest end, and averaging would let one solid joint
 * carry a guessed one into looking certain.
 */
export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  person: Person,
  color: string,
  { lineWidth = 3, radius = 4 } = {},
): void {
  const points = person.keypoints;
  if (points.length === 0) return;
  const byIndex = new Map(points.map((p) => [p.index, p]));

  ctx.save();
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.strokeStyle = color;

  for (const [a, b] of COCO_EDGES) {
    const pa = byIndex.get(a);
    const pb = byIndex.get(b);
    if (!pa || !pb) continue;
    ctx.globalAlpha = jointAlpha(Math.min(pa.score, pb.score));
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }

  ctx.fillStyle = color;
  for (const p of points) {
    ctx.globalAlpha = jointAlpha(p.score);
    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}
