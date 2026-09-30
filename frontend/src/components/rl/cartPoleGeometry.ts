// Where the cart and the pole land on the canvas. Pure, for the reason
// `gridGeometry.ts` is: happy-dom gives a canvas no 2D context, so this is the
// part of the drawing a unit test can reach — and the part a sign error would
// hide in (a pole drawn leaning the wrong way looks exactly like a pole).

import { HALF_LENGTH, X_THRESHOLD } from "@/rl/envs/cartPole";

export interface CartPoleGeometry {
  /** Pixels per metre. */
  scale: number;
  /** The track's y, and its ends in pixels (±2.4 m). */
  trackY: number;
  trackLeft: number;
  trackRight: number;
  cartX: number;
  cartWidth: number;
  cartHeight: number;
  /** The pole's pivot (top of the cart) and its tip. */
  pivot: { x: number; y: number };
  tip: { x: number; y: number };
}

export function cartPoleGeometry(
  x: number,
  theta: number,
  width: number,
  height: number,
): CartPoleGeometry {
  // The track spans ±2.4 m plus room for half a cart at either end, so a cart
  // parked at the limit is still wholly on screen (a 10% margin was a pixel
  // short — the test found it).
  const scale = width / (2 * X_THRESHOLD * 1.15);
  const centre = width / 2;
  const trackY = height * 0.72;
  const cartWidth = 0.5 * scale;
  const cartHeight = 0.3 * scale;
  const cartX = centre + x * scale;
  const pivot = { x: cartX, y: trackY - cartHeight };
  // The *whole* pole is twice HALF_LENGTH. θ is measured from vertical, and a
  // positive θ leans right — screen y grows downward, hence the minus.
  const poleLength = 2 * HALF_LENGTH * scale;
  const tip = {
    x: pivot.x + poleLength * Math.sin(theta),
    y: pivot.y - poleLength * Math.cos(theta),
  };
  return {
    scale,
    trackY,
    trackLeft: centre - X_THRESHOLD * scale,
    trackRight: centre + X_THRESHOLD * scale,
    cartX,
    cartWidth,
    cartHeight,
    pivot,
    tip,
  };
}
