// World (metres) to canvas (CSS pixels) for the reaching task's workspace.
// Pure, because happy-dom has no 2D context and because the one mistake this
// mapping invites — forgetting that canvas y grows downward — draws "above the
// obstacle" below it, which on this page reverses the meaning of every path.

import type { Point } from "@/rl/envs/reacher2d";

export interface WorldBounds {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

/**
 * The drawn region: the task above, and the arm's base and elbow below it — in
 * this inverse-kinematics branch the elbow sits *under* the base at the start,
 * and clipping it made the arm read as a stray line.
 */
export const WORKSPACE: WorldBounds = { x0: -1.8, x1: 1.8, y0: -1.1, y1: 1.9 };

export interface WorkspaceMap {
  /** Pixels per metre — equal on both axes, so a circle stays a circle. */
  scale: number;
  toPixels: (p: Point) => Point;
}

export function workspaceMap(width: number, height: number, bounds = WORKSPACE): WorkspaceMap {
  const scale = Math.min(width / (bounds.x1 - bounds.x0), height / (bounds.y1 - bounds.y0));
  const offX = (width - scale * (bounds.x1 - bounds.x0)) / 2;
  const offY = (height - scale * (bounds.y1 - bounds.y0)) / 2;
  return {
    scale,
    toPixels: (p) => ({
      x: offX + (p.x - bounds.x0) * scale,
      // World y grows upward; canvas y grows downward.
      y: offY + (bounds.y1 - p.y) * scale,
    }),
  };
}
