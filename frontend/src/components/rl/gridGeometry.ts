// Where each cell of the grid lands on the canvas, and what arrow it gets.
//
// Pure, and in its own module, because happy-dom gives a canvas no 2D context:
// `GridCanvas`'s painting cannot be asserted in a unit test, but this can —
// `GraphCanvas.layoutToPixels` is the precedent, and pulling it out there
// immediately surfaced a drawing centred and then shifted again.

/** Padding inside the canvas, in CSS pixels. */
export const GRID_PAD = 4;

export interface GridGeometry {
  rows: number;
  cols: number;
  /** Side of one square cell, in CSS pixels. */
  cell: number;
  originX: number;
  originY: number;
}

/** The largest square cells that fit the box, centred in it. */
export function gridGeometry(
  rows: number,
  cols: number,
  width: number,
  height: number,
): GridGeometry {
  const cell = Math.max(
    0,
    Math.min((width - 2 * GRID_PAD) / cols, (height - 2 * GRID_PAD) / rows),
  );
  return {
    rows,
    cols,
    cell,
    originX: (width - cell * cols) / 2,
    originY: (height - cell * rows) / 2,
  };
}

/** The top-left corner of a state's cell. */
export function cellOrigin(state: number, g: GridGeometry): { x: number; y: number } {
  const row = Math.floor(state / g.cols);
  const col = state % g.cols;
  return { x: g.originX + col * g.cell, y: g.originY + row * g.cell };
}

/** The cell under a point, or null outside the grid. */
export function stateAt(px: number, py: number, g: GridGeometry): number | null {
  if (g.cell <= 0) return null;
  const col = Math.floor((px - g.originX) / g.cell);
  const row = Math.floor((py - g.originY) / g.cell);
  if (col < 0 || row < 0 || col >= g.cols || row >= g.rows) return null;
  return row * g.cols + col;
}

export interface CellArrow {
  /** The greedy action — first index of the maximum, as the learner picks it. */
  action: number;
  value: number;
  /** Every action is worth the same: the arrow is only the tie-break rule. */
  tie: boolean;
}

export function cellArrow(q: Float32Array, state: number, nActions = 4): CellArrow {
  const base = state * nActions;
  let action = 0;
  let value = q[base];
  let tie = true;
  for (let a = 1; a < nActions; a++) {
    if (q[base + a] !== q[base]) tie = false;
    if (q[base + a] > value) {
      value = q[base + a];
      action = a;
    }
  }
  return { action, value, tie };
}

/** Unit direction for each action: LEFT, DOWN, RIGHT, UP. y grows downward. */
export const ACTION_VECTORS: readonly (readonly [number, number])[] = [
  [-1, 0],
  [0, 1],
  [1, 0],
  [0, -1],
];
