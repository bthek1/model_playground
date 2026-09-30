// The three RL canvases, rendered. happy-dom gives a canvas no 2D context and
// no ResizeObserver, so the painting is unreachable here — the geometry is
// tested in gridGeometry / cartPoleGeometry / workspaceGeometry. What *is*
// reachable, and worth pinning, is that each mounts without throwing on those
// guarded early returns and that its accessible name carries the state it
// draws: a screen reader gets the angle, the outcome, the map.

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Workspace } from "@/components/robotics/Workspace";
import { demonstrations } from "@/rl/demos";
import { makeScene } from "@/rl/envs/reacher2d";

import { CartPoleCanvas } from "./CartPoleCanvas";
import { GridCanvas } from "./GridCanvas";

describe("RL canvases", () => {
  it("GridCanvas names the map it draws", () => {
    render(<GridCanvas map="8x8" q={new Float32Array(256)} agent={0} />);
    expect(screen.getByTestId("grid-canvas")).toBeInTheDocument();
    expect(screen.getByRole("img")).toHaveAccessibleName(/8x8 grid/);
  });

  it("CartPoleCanvas reads the cart's position and the pole's angle in degrees", () => {
    render(<CartPoleCanvas state={Float32Array.from([0.5, 0, Math.PI / 18, 0])} />);
    expect(screen.getByRole("img")).toHaveAccessibleName(/0\.50 m.*10\.0 degrees/);
    expect(screen.getByText(/θ 10\.0°/)).toBeInTheDocument();
  });

  it("Workspace says how many demonstrations it shows, and the rollout's outcome", () => {
    const scene = makeScene(0);
    const demos = demonstrations(scene, "both", 4, 1);
    render(
      <Workspace
        scene={scene}
        demos={demos}
        rollout={{ path: [scene.start, scene.obstacle], outcome: "collided" }}
        field={[]}
        testId="ws"
      />,
    );
    expect(screen.getByTestId("ws")).toBeInTheDocument();
    expect(screen.getByRole("img")).toHaveAccessibleName(/4 demonstrations.*collided/);
  });
});
