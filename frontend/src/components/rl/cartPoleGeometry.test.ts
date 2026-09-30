import { describe, expect, it } from "vitest";

import { THETA_THRESHOLD, X_THRESHOLD } from "@/rl/envs/cartPole";

import { cartPoleGeometry } from "./cartPoleGeometry";

describe("cartPoleGeometry", () => {
  it("centres a cart at x = 0 with the pole straight up", () => {
    const g = cartPoleGeometry(0, 0, 600, 300);
    expect(g.cartX).toBe(300);
    expect(g.tip.x).toBeCloseTo(g.pivot.x);
    expect(g.tip.y).toBeLessThan(g.pivot.y);
  });

  it("draws a positive angle leaning right and a negative one leaning left", () => {
    expect(cartPoleGeometry(0, THETA_THRESHOLD, 600, 300).tip.x).toBeGreaterThan(300);
    expect(cartPoleGeometry(0, -THETA_THRESHOLD, 600, 300).tip.x).toBeLessThan(300);
  });

  it("puts the cart at the track's ends at ±2.4 m, inside the canvas", () => {
    const right = cartPoleGeometry(X_THRESHOLD, 0, 600, 300);
    const left = cartPoleGeometry(-X_THRESHOLD, 0, 600, 300);
    expect(right.cartX).toBeCloseTo(right.trackRight);
    expect(left.cartX).toBeCloseTo(left.trackLeft);
    expect(right.cartX + right.cartWidth / 2).toBeLessThanOrEqual(600);
    expect(left.cartX - left.cartWidth / 2).toBeGreaterThanOrEqual(0);
  });

  it("draws the whole pole — twice the half-length the physics uses", () => {
    const g = cartPoleGeometry(0, 0, 600, 300);
    expect(g.pivot.y - g.tip.y).toBeCloseTo(1.0 * g.scale);
  });
});
