import { describe, expect, it } from "vitest";

import { demonstrations, pairs, sidesFor, waypoint } from "./demos";
import { makeScene } from "./envs/reacher2d";

describe("the scripted demonstrator", () => {
  // Checked before any learning: if the paths were not genuinely bimodal, the
  // page would demonstrate nothing and still produce a clean-looking result.

  it("keeps an 'above' path above the obstacle's centre line and a 'below' path below it", () => {
    const scene = makeScene(0);
    for (const d of demonstrations(scene, "both", 20, 1)) {
      // Where the hand passes the obstacle, it is on its chosen side.
      const passing = d.path.filter((p) => Math.abs(p.x - scene.obstacle.x) < scene.radius);
      expect(passing.length).toBeGreaterThan(0);
      for (const p of passing) {
        if (d.side === "above") expect(p.y).toBeGreaterThan(scene.obstacle.y);
        else expect(p.y).toBeLessThan(scene.obstacle.y);
      }
    }
  });

  it("is a competent expert: every demonstration reaches the target, at every offset", () => {
    // The first version turned for the target 0.15 m early and cut the corner
    // over the obstacle — 3 of 20 collided. An expert that fails muddies the
    // experiment it exists for.
    for (const offset of [-0.2, 0, 0.15, 0.25]) {
      const demos = demonstrations(makeScene(offset), "both", 30, 3);
      expect(demos.every((d) => d.reached), `offset ${offset}`).toBe(true);
    }
  });

  it("returns the mix it was asked for", () => {
    expect(sidesFor("one", 4)).toEqual(["above", "above", "above", "above"]);
    expect(sidesFor("both", 4)).toEqual(["above", "below", "above", "below"]);
    const one = demonstrations(makeScene(0), "one", 6, 2);
    expect(one.every((d) => d.side === "above")).toBe(true);
  });

  it("gives the same set for the same seed", () => {
    const a = pairs(demonstrations(makeScene(0), "both", 8, 4));
    const b = pairs(demonstrations(makeScene(0), "both", 8, 4));
    expect(Array.from(a.y)).toEqual(Array.from(b.y));
    const c = pairs(demonstrations(makeScene(0), "both", 8, 5));
    expect(Array.from(c.y)).not.toEqual(Array.from(a.y));
  });

  it("puts the two waypoints on opposite sides, clear of the obstacle", () => {
    const scene = makeScene(0.1);
    const up = waypoint(scene, "above");
    const down = waypoint(scene, "below");
    expect(up.y).toBeGreaterThan(scene.obstacle.y);
    expect(down.y).toBeLessThan(scene.obstacle.y);
    for (const w of [up, down]) {
      expect(Math.hypot(w.x - scene.obstacle.x, w.y - scene.obstacle.y)).toBeGreaterThan(scene.radius);
    }
  });
});
