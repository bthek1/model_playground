import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { OverlayCanvas, type CanvasPick } from "./OverlayCanvas";

// happy-dom has no canvas implementation, so `getContext("2d")` returns null and
// `getBoundingClientRect` reports zeros. Both are deliberate cases here: the
// component must survive the first and fall back sanely on the second.

/** Give the canvas the on-screen box a real layout would. */
function layout(canvas: HTMLElement, box: Partial<DOMRect>) {
  vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue({
    left: 0,
    top: 0,
    width: 0,
    height: 0,
    right: 0,
    bottom: 0,
    x: 0,
    y: 0,
    toJSON: () => ({}),
    ...box,
  } as DOMRect);
}

beforeEach(() => vi.restoreAllMocks());

describe("OverlayCanvas", () => {
  it("names itself for a screen reader, because a canvas is otherwise opaque", () => {
    render(
      <OverlayCanvas width={4} height={4} draw={() => {}} label="Two detections" />,
    );
    expect(screen.getByRole("img", { name: "Two detections" })).toBeInTheDocument();
  });

  it("does not throw when the browser gives it no 2D context", () => {
    // An OUTPUT slot that throws here takes the whole route down with it — and
    // this is the normal case under happy-dom and on a memory-starved canvas.
    const draw = vi.fn();
    expect(() =>
      render(<OverlayCanvas width={4} height={4} draw={draw} label="x" />),
    ).not.toThrow();
    expect(draw).not.toHaveBeenCalled();
  });

  it("is not clickable unless a route asks for it", () => {
    render(<OverlayCanvas width={4} height={4} draw={() => {}} label="x" testId="c" />);
    const canvas = screen.getByTestId("c");
    // No handler, and no cursor affordance promising one.
    expect(canvas.className).not.toMatch(/cursor-crosshair/);
    fireEvent.click(canvas, { clientX: 2, clientY: 2 });
    // Nothing to assert but the absence of a throw — which is the point.
  });

  describe("onPick", () => {
    it("reports the click in source pixels, not display pixels", () => {
      // **The silent failure this conversion exists to prevent.** The canvas is
      // sized to the source image and scaled down by CSS, so a raw `clientX` is
      // in display pixels and wrong by the scale factor. On /mask-generation
      // that error is invisible: SAM returns a perfectly plausible mask of
      // whatever happens to be at the mis-mapped point.
      const onPick = vi.fn<(p: CanvasPick) => void>();
      render(
        <OverlayCanvas
          width={640}
          height={480}
          draw={() => {}}
          label="x"
          testId="c"
          onPick={onPick}
        />,
      );
      const canvas = screen.getByTestId("c");
      // Displayed at half size: 320x240 on screen for a 640x480 source.
      layout(canvas, { left: 0, top: 0, width: 320, height: 240 });

      fireEvent.click(canvas, { clientX: 160, clientY: 120 });
      expect(onPick).toHaveBeenCalledWith({ x: 320, y: 240, alt: false });
    });

    it("subtracts the canvas' own offset on the page", () => {
      // A click at the top-left *of the canvas* is (0, 0) in source pixels,
      // however far down the page the canvas happens to be.
      const onPick = vi.fn<(p: CanvasPick) => void>();
      render(
        <OverlayCanvas
          width={100}
          height={100}
          draw={() => {}}
          label="x"
          testId="c"
          onPick={onPick}
        />,
      );
      const canvas = screen.getByTestId("c");
      layout(canvas, { left: 40, top: 200, width: 100, height: 100 });

      fireEvent.click(canvas, { clientX: 40, clientY: 200 });
      expect(onPick).toHaveBeenCalledWith({ x: 0, y: 0, alt: false });
    });

    it("carries the Alt modifier, which is a negative point on /mask-generation", () => {
      // 1 means "the object", 0 means "not the object". Losing the modifier
      // inverts every mask a second click was meant to refine.
      const onPick = vi.fn<(p: CanvasPick) => void>();
      render(
        <OverlayCanvas
          width={10}
          height={10}
          draw={() => {}}
          label="x"
          testId="c"
          onPick={onPick}
        />,
      );
      const canvas = screen.getByTestId("c");
      layout(canvas, { width: 10, height: 10 });

      fireEvent.click(canvas, { clientX: 5, clientY: 5, altKey: true });
      expect(onPick.mock.calls[0][0].alt).toBe(true);
    });

    it("falls back to 1:1 rather than reporting Infinity on a zero-sized box", () => {
      // happy-dom reports a zero rect, and so does a canvas that has not been
      // laid out yet. Dividing by that width would hand the route `Infinity`
      // and put every point outside the image.
      const onPick = vi.fn<(p: CanvasPick) => void>();
      render(
        <OverlayCanvas
          width={100}
          height={100}
          draw={() => {}}
          label="x"
          testId="c"
          onPick={onPick}
        />,
      );
      fireEvent.click(screen.getByTestId("c"), { clientX: 7, clientY: 9 });

      const pick = onPick.mock.calls[0][0];
      expect(Number.isFinite(pick.x)).toBe(true);
      expect(Number.isFinite(pick.y)).toBe(true);
      expect(pick).toMatchObject({ x: 7, y: 9 });
    });

    it("advertises itself as clickable", () => {
      render(
        <OverlayCanvas
          width={4}
          height={4}
          draw={() => {}}
          label="x"
          testId="c"
          onPick={() => {}}
        />,
      );
      expect(screen.getByTestId("c").className).toMatch(/cursor-crosshair/);
    });
  });
});
