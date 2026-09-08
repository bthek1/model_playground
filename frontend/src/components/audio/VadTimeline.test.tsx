// happy-dom has no 2-D canvas context, so nothing here asserts pixels — the
// drawing is exercised for real in `e2e/specs/audio-models.spec.ts`. What is
// worth pinning in a unit test is the contract the rest of the app depends on:
// the element survives a null context, and it announces itself, because the
// route test and any screen reader both find it by its accessible name.

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { VadTimeline } from "./VadTimeline";

const probs = Float32Array.from([0.01, 0.4, 0.9, 0.95, 0.2]);

describe("VadTimeline", () => {
  it("renders without a 2-D context, as in jsdom/happy-dom", () => {
    expect(() =>
      render(<VadTimeline probabilities={probs} threshold={0.5} />),
    ).not.toThrow();
    expect(screen.getByRole("img")).toBeInTheDocument();
  });

  it("announces the frame count and the current threshold", () => {
    render(<VadTimeline probabilities={probs} threshold={0.5} />);
    expect(
      screen.getByRole("img", { name: /5 frames, threshold 0\.50/ }),
    ).toBeInTheDocument();
  });

  it("re-announces when the threshold moves", () => {
    const { rerender } = render(
      <VadTimeline probabilities={probs} threshold={0.5} />,
    );
    rerender(<VadTimeline probabilities={probs} threshold={0.9} />);
    expect(
      screen.getByRole("img", { name: /threshold 0\.90/ }),
    ).toBeInTheDocument();
  });

  it("handles an empty result rather than dividing by zero frames", () => {
    expect(() =>
      render(<VadTimeline probabilities={new Float32Array(0)} threshold={0.5} />),
    ).not.toThrow();
    expect(
      screen.getByRole("img", { name: /0 frames/ }),
    ).toBeInTheDocument();
  });

  it("takes its colour from a text class, so both themes work", () => {
    render(
      <VadTimeline
        probabilities={probs}
        threshold={0.5}
        className="text-primary"
      />,
    );
    // The canvas paints in its own resolved `color` — a hard-coded fill would
    // vanish against one of the two theme backgrounds.
    expect(screen.getByRole("img")).toHaveClass("text-primary");
  });
});
