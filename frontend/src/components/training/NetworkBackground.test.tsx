import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { NetworkBackground } from "./NetworkBackground";

// happy-dom's canvas has no 2D context, so the component takes its guarded
// early-return path. These tests assert it mounts, exposes an aria-hidden
// canvas, and survives a weights-prop change without throwing (the rAF loop and
// pooling run regardless of whether a drawing context is available).
describe("NetworkBackground", () => {
  it("renders an aria-hidden canvas", () => {
    const { container } = render(
      <NetworkBackground weights={new Float32Array(784 * 10)} />,
    );
    const canvas = container.querySelector("canvas");
    expect(canvas).not.toBeNull();
    expect(canvas).toHaveAttribute("aria-hidden");
  });

  it("re-pools without throwing when the weights buffer changes", () => {
    const weights = new Float32Array(784 * 10);
    const bias = new Float32Array(10);
    const { rerender } = render(
      <NetworkBackground weights={weights} bias={bias} active={false} />,
    );
    const next = new Float32Array(784 * 10);
    next.fill(0.5);
    expect(() =>
      rerender(
        <NetworkBackground weights={next} bias={bias} active={true} />,
      ),
    ).not.toThrow();
  });

  it("tolerates a short (empty) weights buffer", () => {
    expect(() =>
      render(<NetworkBackground weights={new Float32Array(0)} />),
    ).not.toThrow();
  });
});
