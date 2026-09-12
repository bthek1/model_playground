import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { AudioSample } from "@/audio/samples";
import type { PickedAudio } from "@/hooks/useAudioPick";

// happy-dom has no canvas 2D context; Waveform renders its own guarded fallback
// and is covered by its own tests.
vi.mock("@/components/audio/Waveform", () => ({
  Waveform: () => <div data-testid="waveform" />,
  LiveWaveform: () => <div data-testid="live-waveform" />,
}));

const { AudioSourcePanel } = await import("./AudioSourcePanel");

const samples: AudioSample[] = [
  { id: "jfk", label: "JFK", url: "u1", reference: "ask not", hint: "~11 s" },
  { id: "mlk", label: "MLK", url: "u2", reference: "i have a dream", hint: "short" },
];

const clip: PickedAudio = {
  audio: new Float32Array(16_000 * 2), // 2 s at 16 kHz
  sampleRate: 16_000,
  name: "jfk.wav",
  sample: samples[0],
};

function renderPanel(
  props: Partial<Parameters<typeof AudioSourcePanel>[0]> = {},
) {
  const onFile = vi.fn();
  const onSample = vi.fn();
  const onRecord = vi.fn();
  const view = render(
    <AudioSourcePanel
      clip={null}
      preparing={null}
      samples={samples}
      sampleHint="Samples"
      onFile={onFile}
      onSample={onSample}
      onRecord={onRecord}
      recordSeconds={5}
      busy={false}
      {...props}
    />,
  );
  return { ...view, onFile, onSample, onRecord };
}

describe("AudioSourcePanel", () => {
  it("says there is nothing to run on before a clip is chosen", () => {
    renderPanel();
    expect(screen.getByTestId("audio-input-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("waveform")).not.toBeInTheDocument();
  });

  it("shows the held clip — its name, duration, rate and waveform", () => {
    // This surface is the whole point of separating INPUT from RUN: there is a
    // "what am I about to run on?" state on screen, and it survives across runs
    // and parameter changes.
    renderPanel({ clip });

    expect(screen.queryByTestId("audio-input-empty")).not.toBeInTheDocument();
    expect(screen.getByText("jfk.wav")).toBeInTheDocument();
    expect(screen.getByText(/2\.0s/)).toBeInTheDocument();
    expect(screen.getByText(/16 kHz/)).toBeInTheDocument();
    expect(screen.getByTestId("waveform")).toBeInTheDocument();
  });

  it("shows a 48 kHz clip's real rate — this panel serves both routes", () => {
    renderPanel({
      clip: { ...clip, sampleRate: 48_000, audio: new Float32Array(48_000) },
    });
    expect(screen.getByText(/48 kHz/)).toBeInTheDocument();
    expect(screen.getByText(/1\.0s/)).toBeInTheDocument();
  });

  // **The rule this component encodes.** Getting audio in is not running a
  // model, so none of these are gated on `ready` — the panel is never even
  // told about it. Gating them forced a download before the user was allowed
  // to choose what to run it on.
  it("offers every source with no model loaded", () => {
    renderPanel();
    expect(screen.getByRole("button", { name: /record 5s/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /upload audio/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /^JFK$/ })).toBeEnabled();
  });

  it("hands a chosen sample back, and starts nothing itself", () => {
    const { onSample, onFile, onRecord } = renderPanel();

    fireEvent.click(screen.getByRole("button", { name: /^MLK$/ }));

    expect(onSample).toHaveBeenCalledWith(samples[1]);
    expect(onFile).not.toHaveBeenCalled();
    expect(onRecord).not.toHaveBeenCalled();
  });

  it("asks for a recording of the length the route chose", () => {
    const { onRecord } = renderPanel({ recordSeconds: 6 });
    fireEvent.click(screen.getByRole("button", { name: /record 6s/i }));
    expect(onRecord).toHaveBeenCalledWith(6);
  });

  it("hands over a selected file, and lets the same one be picked twice", () => {
    // The input's value is cleared after each change, or re-selecting the file
    // you just chose fires no event at all.
    const { onFile } = renderPanel();
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(["x"], "take.wav", { type: "audio/wav" });

    fireEvent.change(input, { target: { files: [file] } });

    expect(onFile).toHaveBeenCalledWith(file);
    expect(input.value).toBe("");
  });

  it("says which sample the held clip came from, not just styles it", () => {
    renderPanel({ clip });
    // JFK produced the clip; MLK did not. Announced, so the selection is
    // legible to a screen reader and not only to the eye.
    expect(screen.getByRole("button", { name: /^JFK$/ })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: /^MLK$/ })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("disables every source while a run is in flight", () => {
    // Not because a model is missing — because the clip must not change under
    // an inference that is already reading it.
    renderPanel({ busy: true });
    expect(screen.getByRole("button", { name: /record 5s/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /upload audio/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /^JFK$/ })).toBeDisabled();
  });

  it("names the source that is preparing, so the right button spins", () => {
    const { rerender } = renderPanel({ preparing: "file", busy: true });
    expect(screen.getByRole("button", { name: /decoding/i })).toBeInTheDocument();

    rerender(
      <AudioSourcePanel
        clip={null}
        preparing="mic"
        samples={samples}
        sampleHint="Samples"
        onFile={vi.fn()}
        onSample={vi.fn()}
        onRecord={vi.fn()}
        recordSeconds={5}
        busy
      />,
    );
    expect(screen.getByRole("button", { name: /recording/i })).toBeInTheDocument();
  });

  it("hides the mic on a route with no capture path", () => {
    renderPanel({ onRecord: undefined });
    expect(
      screen.queryByRole("button", { name: /record/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /upload audio/i })).toBeInTheDocument();
  });

  it("omits the sample row when a route bundles no clips", () => {
    renderPanel({ samples: [] });
    expect(screen.queryByRole("button", { name: /^JFK$/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /upload audio/i })).toBeInTheDocument();
  });

  it("renders a route's own extra controls beneath the clip", () => {
    // CLAP's prompt editor and the like. They belong to INPUT, and editing one
    // is a reason to press GENERATE again — never a reason to re-capture.
    renderPanel({ children: <label>Labels to score against</label> });
    expect(screen.getByText(/labels to score against/i)).toBeInTheDocument();
  });
});
