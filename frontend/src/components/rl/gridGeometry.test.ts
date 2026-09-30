import { describe, expect, it } from "vitest";

import {
  ACTION_VECTORS,
  cellArrow,
  cellOrigin,
  GRID_PAD,
  gridGeometry,
  stateAt,
} from "./gridGeometry";

describe("gridGeometry", () => {
  it("fits square cells and centres the grid in a wide box", () => {
    const g = gridGeometry(4, 4, 400, 200);
    expect(g.cell).toBe((200 - 2 * GRID_PAD) / 4);
    // Equal margins left and right — no second offset on top of the centring.
    const right = 400 - (g.originX + 4 * g.cell);
    expect(g.originX).toBeCloseTo(right);
    expect(g.originY).toBeCloseTo(GRID_PAD);
  });

  it("puts the corner cells at the corners, on both maps", () => {
    for (const n of [4, 8]) {
      const g = gridGeometry(n, n, 300, 300);
      const first = cellOrigin(0, g);
      const last = cellOrigin(n * n - 1, g);
      expect(first).toEqual({ x: GRID_PAD, y: GRID_PAD });
      expect(last.x + g.cell).toBeCloseTo(300 - GRID_PAD);
      expect(last.y + g.cell).toBeCloseTo(300 - GRID_PAD);
      // The top-right and bottom-left cells: row-major, not column-major.
      expect(cellOrigin(n - 1, g).y).toBe(GRID_PAD);
      expect(cellOrigin(n * (n - 1), g).x).toBe(GRID_PAD);
    }
  });

  it("maps a point back to the cell it is in, and refuses one outside", () => {
    const g = gridGeometry(4, 4, 208, 208);
    for (const s of [0, 3, 12, 15, 6]) {
      const { x, y } = cellOrigin(s, g);
      expect(stateAt(x + 1, y + 1, g)).toBe(s);
      expect(stateAt(x + g.cell - 1, y + g.cell - 1, g)).toBe(s);
    }
    expect(stateAt(1, 1, g)).toBeNull();
    expect(stateAt(207, 100, g)).toBeNull();
    expect(stateAt(10, 10, gridGeometry(4, 4, 0, 0))).toBeNull();
  });
});

describe("cellArrow", () => {
  it("points at the first maximum, as the learner's greedy action does", () => {
    const q = new Float32Array([0, 0.2, 0.5, 0.5]);
    expect(cellArrow(q, 0)).toEqual({ action: 2, value: 0.5, tie: false });
  });

  it("marks an untouched cell as a tie, pointing where np.argmax would", () => {
    expect(cellArrow(new Float32Array(4), 0)).toEqual({ action: 0, value: 0, tie: true });
  });

  it("draws LEFT, DOWN, RIGHT, UP in screen space, where y grows downward", () => {
    expect(ACTION_VECTORS).toEqual([
      [-1, 0],
      [0, 1],
      [1, 0],
      [0, -1],
    ]);
  });
});
