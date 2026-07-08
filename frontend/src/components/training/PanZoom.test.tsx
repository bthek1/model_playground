import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PanZoom } from "./PanZoom";

// happy-dom has no layout (offsetWidth/clientWidth are 0), so `fit()` no-ops and
// the transform stays at its initial value — these tests assert the surface
// mounts, renders its children and zoom controls, and survives interaction.
describe("PanZoom", () => {
  it("renders its children and zoom controls", () => {
    render(
      <PanZoom>
        <div>diagram</div>
      </PanZoom>,
    );
    expect(screen.getByText("diagram")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /zoom in/i })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /zoom out/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /fit to view/i }),
    ).toBeInTheDocument();
    // Zoom level indicator starts at 100%.
    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("does not throw on a pan drag gesture", () => {
    render(
      <PanZoom>
        <div>diagram</div>
      </PanZoom>,
    );
    const surface = screen.getByText("diagram").parentElement!.parentElement!;
    expect(() => {
      fireEvent.pointerDown(surface, { button: 0, clientX: 10, clientY: 10 });
      fireEvent.pointerMove(surface, { clientX: 40, clientY: 55 });
      fireEvent.pointerUp(surface, { clientX: 40, clientY: 55 });
    }).not.toThrow();
  });

  it("zoom-in button is clickable without throwing", () => {
    render(
      <PanZoom>
        <div>diagram</div>
      </PanZoom>,
    );
    expect(() =>
      fireEvent.click(screen.getByRole("button", { name: /zoom in/i })),
    ).not.toThrow();
  });
});
