import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { Pixels } from "@/vision/draw";
import { CompareSlider } from "./CompareSlider";

function pixels(width = 4, height = 4): Pixels {
  return {
    data: new Uint8ClampedArray(width * height * 3),
    width,
    height,
    channels: 3,
  };
}

describe("CompareSlider", () => {
  it("renders both halves, each with its own accessible name", () => {
    // A canvas is opaque to a screen reader without one, and here the two names
    // are the entire content of the comparison.
    render(
      <CompareSlider
        before={pixels()}
        after={pixels()}
        beforeLabel="Bicubic"
        afterLabel="Swin2SR"
        testId="sr-compare"
      />,
    );
    expect(screen.getByRole("img", { name: "Bicubic" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Swin2SR" })).toBeInTheDocument();
  });

  it("is operable from the keyboard, not by drag alone", () => {
    // A drag-only comparison is a comparison a keyboard user cannot make. The
    // handle is a slider, so the arrow keys have to move it.
    render(
      <CompareSlider
        before={pixels()}
        after={pixels()}
        beforeLabel="Bicubic"
        afterLabel="Swin2SR"
      />,
    );
    const handle = screen.getByRole("slider");
    expect(handle).toHaveAttribute("aria-valuenow", "50");

    fireEvent.keyDown(handle, { key: "ArrowRight" });
    expect(handle).toHaveAttribute("aria-valuenow", "55");

    fireEvent.keyDown(handle, { key: "ArrowLeft" });
    fireEvent.keyDown(handle, { key: "ArrowLeft" });
    expect(handle).toHaveAttribute("aria-valuenow", "45");
  });

  it("clamps the split to the frame at both ends", () => {
    render(
      <CompareSlider
        before={pixels()}
        after={pixels()}
        beforeLabel="Bicubic"
        afterLabel="Swin2SR"
      />,
    );
    const handle = screen.getByRole("slider");
    for (let i = 0; i < 30; i++) fireEvent.keyDown(handle, { key: "ArrowLeft" });
    expect(handle).toHaveAttribute("aria-valuenow", "0");

    for (let i = 0; i < 60; i++) fireEvent.keyDown(handle, { key: "ArrowRight" });
    expect(handle).toHaveAttribute("aria-valuenow", "100");
  });
});
