import { describe, expect, it } from "vitest";

import {
  forwardKinematics,
  handVelocity,
  inverseKinematics,
  jointVelocity,
  L1,
  L2,
  makeScene,
  Reacher2D,
} from "./reacher2d";

describe("Reacher2D", () => {
  it("agrees with hand-computed forward kinematics at two known poses", () => {
    // Straight out along x: both links on the axis.
    expect(forwardKinematics(0, 0).hand.x).toBeCloseTo(L1 + L2);
    expect(forwardKinematics(0, 0).hand.y).toBeCloseTo(0);
    // Shoulder up 90°, elbow bent back 90°: elbow at (0, 1), hand at (1, 1).
    const p = forwardKinematics(Math.PI / 2, -Math.PI / 2);
    expect(p.elbow.x).toBeCloseTo(0);
    expect(p.elbow.y).toBeCloseTo(1);
    expect(p.hand.x).toBeCloseTo(1);
    expect(p.hand.y).toBeCloseTo(1);
  });

  it("inverts its own kinematics, and refuses a point the arm cannot reach", () => {
    for (const target of [{ x: -1.2, y: 0.8 }, { x: 0.3, y: 1.5 }, { x: 1.2, y: 0.8 }]) {
      const q = inverseKinematics(target)!;
      const { hand } = forwardKinematics(q[0], q[1]);
      expect(hand.x).toBeCloseTo(target.x, 9);
      expect(hand.y).toBeCloseTo(target.y, 9);
    }
    // The reachable workspace is what the joint lengths imply: |p| ≤ L1 + L2.
    expect(inverseKinematics({ x: 2.01, y: 0 })).toBeNull();
    expect(inverseKinematics({ x: 1.99, y: 0 })).not.toBeNull();
  });

  it("maps a hand velocity to joint velocities and back", () => {
    const v = { x: 0.4, y: -0.3 };
    const dq = jointVelocity(0.9, -1.2, v)!;
    const back = handVelocity(0.9, -1.2, dq[0], dq[1]);
    expect(back.x).toBeCloseTo(v.x, 9);
    expect(back.y).toBeCloseTo(v.y, 9);
  });

  it("terminates on contact with the obstacle", () => {
    const scene = makeScene(0);
    const env = new Reacher2D(scene);
    env.reset(1, 0);
    // Drive the hand straight at the obstacle's centre.
    for (let t = 0; t < 200; t++) {
      const h = env.hand;
      const dq = jointVelocity(env.q[0], env.q[1], { x: scene.obstacle.x - h.x, y: scene.obstacle.y - h.y })!;
      const r = env.step(Float32Array.from(dq));
      if (r.terminated) break;
    }
    expect(env.outcome).toBe("collided");
    expect(() => env.step(Float32Array.from([0, 0]))).toThrow(/finished episode/);
  });

  it("is deterministic at a seed", () => {
    const a = new Reacher2D(makeScene(0)).reset(5);
    const b = new Reacher2D(makeScene(0)).reset(5);
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  it("calls a run that never arrives stalled, not collided", () => {
    const env = new Reacher2D(makeScene(0), 5);
    env.reset(1, 0);
    for (let t = 0; t < 5; t++) env.step(Float32Array.from([0, 0]));
    expect(env.outcome).toBe("stalled");
  });
});
