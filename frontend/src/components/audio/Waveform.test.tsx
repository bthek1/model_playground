// The waveform canvases, which every audio route draws into. As with
// `VadTimeline`, happy-dom has no 2-D context and no real Web Audio, so the
// pixels are not on trial — what is, is that neither component throws in an
// environment missing those APIs. Both are rendered by five routes, so a throw
// here takes the whole page down, not just the picture.

import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { LiveWaveform, Waveform } from "./Waveform";

const samples = Float32Array.from([0, 0.5, -0.5, 0.9, -0.9, 0.1]);

afterEach(() => vi.unstubAllGlobals());

describe("Waveform", () => {
  it("renders a labelled canvas without a 2-D context", () => {
    expect(() => render(<Waveform samples={samples} />)).not.toThrow();
    expect(
      screen.getByRole("img", { name: /audio waveform/i }),
    ).toBeInTheDocument();
  });

  it("survives an empty clip", () => {
    expect(() => render(<Waveform samples={new Float32Array(0)} />)).not.toThrow();
  });

  it("survives an environment with no ResizeObserver", () => {
    // Older Safari and some test environments; the component checks for it
    // rather than assuming, so the static draw still happens.
    vi.stubGlobal("ResizeObserver", undefined);
    expect(() => render(<Waveform samples={samples} />)).not.toThrow();
  });

  it("keeps the caller's colour class alongside its own defaults", () => {
    render(<Waveform samples={samples} className="text-muted-foreground" />);
    const canvas = screen.getByRole("img", { name: /audio waveform/i });
    expect(canvas).toHaveClass("text-muted-foreground");
    expect(canvas).toHaveClass("w-full");
  });
});

describe("LiveWaveform", () => {
  it("renders without a real AudioContext instead of throwing", () => {
    // The mic view mounts as soon as recording starts; an env without Web Audio
    // must degrade to an empty canvas, not an error boundary.
    vi.stubGlobal("AudioContext", undefined);
    const stream = { getTracks: () => [] } as unknown as MediaStream;
    expect(() => render(<LiveWaveform stream={stream} />)).not.toThrow();
  });
});
