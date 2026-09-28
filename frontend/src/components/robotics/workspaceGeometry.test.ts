import { describe, expect, it } from "vitest";

import { WORKSPACE, workspaceMap } from "./workspaceGeometry";

describe("workspaceMap", () => {
  it("draws higher world points higher on the canvas", () => {
    const m = workspaceMap(360, 200);
    expect(m.toPixels({ x: 0, y: 1.5 }).y).toBeLessThan(m.toPixels({ x: 0, y: 0.5 }).y);
  });

  it("maps the bounds' corners to the drawn box's corners, centred", () => {
    const m = workspaceMap(600, 200);
    const topLeft = m.toPixels({ x: WORKSPACE.x0, y: WORKSPACE.y1 });
    const bottomRight = m.toPixels({ x: WORKSPACE.x1, y: WORKSPACE.y0 });
    expect(topLeft.y).toBeCloseTo(0);
    expect(bottomRight.y).toBeCloseTo(200);
    // Equal margins left and right in a wide box.
    expect(topLeft.x).toBeCloseTo(600 - bottomRight.x);
  });

  it("uses one scale on both axes, so the obstacle stays round", () => {
    const m = workspaceMap(500, 300);
    const a = m.toPixels({ x: 0, y: 1 });
    const b = m.toPixels({ x: 1, y: 1 });
    const c = m.toPixels({ x: 0, y: 0 });
    expect(b.x - a.x).toBeCloseTo(c.y - a.y);
  });
});
