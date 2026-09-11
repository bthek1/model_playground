import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AUDIO_SAMPLES } from "@/audio/samples";

// No Web Audio in happy-dom. `play` returns a fake AudioContext whose transport
// calls the tests assert on — the pause/resume path is the whole reason this
// component owns state rather than being pure markup.
const playCtx = {
  state: "running",
  suspend: vi.fn(),
  resume: vi.fn(),
  close: vi.fn(),
};
const playMock = vi.fn<(...args: unknown[]) => typeof playCtx>(() => playCtx);
vi.mock("@/audio/io", () => ({
  play: (...args: unknown[]) => playMock(...args),
  toWavBlob: vi.fn(() => new Blob()),
}));

// The canvas visualisations have no 2D context here; they render their own
// guarded fallback and are covered by Waveform's own tests.
vi.mock("@/components/audio/Waveform", () => ({
  Waveform: () => <div data-testid="waveform" />,
  LiveWaveform: () => <div data-testid="live-waveform" />,
}));

const { AudioTake, SampleClips } = await import("./AsrTransport");

const CLIP = new Float32Array(16000); // exactly 1s at 16 kHz

function takeProps(over: Partial<Parameters<typeof AudioTake>[0]> = {}) {
  return {
    recording: false,
    stream: null,
    clip: CLIP,
    sampleRate: 16000,
    ...over,
  };
}

beforeEach(() => {
  playCtx.state = "running";
});
afterEach(() => vi.clearAllMocks());

describe("AudioTake", () => {
  it("renders nothing before there is a take or a live signal", () => {
    const { container } = render(
      <AudioTake {...takeProps({ clip: null })} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the live waveform while recording, not the retained take", () => {
    render(
      <AudioTake
        {...takeProps({
          recording: true,
          stream: {} as MediaStream,
          clip: null,
        })}
      />,
    );
    expect(screen.getByTestId("live-waveform")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /play/i })).toBeNull();
  });

  it("shows the take's duration and rate once recording stops", () => {
    render(<AudioTake {...takeProps()} />);
    expect(screen.getByText(/16 kHz mono/)).toBeInTheDocument();
    expect(screen.getByTestId("waveform")).toBeInTheDocument();
  });

  it("plays, pauses, resumes, then stops — the four-state transport", async () => {
    render(<AudioTake {...takeProps()} />);

    await userEvent.click(screen.getByRole("button", { name: /^play$/i }));
    expect(playMock).toHaveBeenCalledWith(CLIP, 16000, expect.any(Function));

    // Playing → the button becomes Pause and a Stop appears.
    const pause = screen.getByRole("button", { name: /pause/i });
    await userEvent.click(pause);
    expect(playCtx.suspend).toHaveBeenCalledOnce();

    // Paused → Resume, in place, rather than restarting from the top.
    await userEvent.click(screen.getByRole("button", { name: /resume/i }));
    expect(playCtx.resume).toHaveBeenCalledOnce();
    expect(playMock).toHaveBeenCalledOnce();

    await userEvent.click(screen.getByRole("button", { name: /^stop$/i }));
    expect(playCtx.close).toHaveBeenCalled();
    // Back to idle: Stop is gone and the button offers Play again.
    expect(screen.queryByRole("button", { name: /^stop$/i })).toBeNull();
    expect(screen.getByRole("button", { name: /^play$/i })).toBeInTheDocument();
  });

  it("returns to idle when the clip reaches its natural end", async () => {
    render(<AudioTake {...takeProps()} />);
    await userEvent.click(screen.getByRole("button", { name: /^play$/i }));
    expect(screen.getByRole("button", { name: /pause/i })).toBeInTheDocument();

    // `play` hands back an onEnded callback; firing it must reset the transport.
    const onEnded = playMock.mock.calls[0][2] as () => void;
    onEnded();
    expect(await screen.findByRole("button", { name: /^play$/i })).toBeInTheDocument();
  });

  it("resets the transport when a new take replaces the old one", async () => {
    const { rerender } = render(<AudioTake {...takeProps()} />);
    await userEvent.click(screen.getByRole("button", { name: /^play$/i }));
    expect(screen.getByRole("button", { name: /pause/i })).toBeInTheDocument();

    rerender(<AudioTake {...takeProps({ clip: new Float32Array(8000) })} />);
    expect(screen.getByRole("button", { name: /^play$/i })).toBeInTheDocument();
  });

  // This surface used to carry its own "Transcribe clip" button, which made two
  // RUN triggers on one page. The take now feeds the route's single Transcribe
  // button, and what is left here is playback and download — local, cheap, and
  // nothing to do with a model.
  it("runs nothing itself: no transcribe control lives here", () => {
    render(<AudioTake {...takeProps()} />);
    expect(
      screen.queryByRole("button", { name: /transcrib/i }),
    ).not.toBeInTheDocument();
  });

  it("keeps playback available regardless of what the model is doing", () => {
    // Listening back is local and cheap; gating it on the model would be rude.
    render(<AudioTake {...takeProps()} />);
    expect(screen.getByRole("button", { name: /^play$/i })).toBeEnabled();
  });
});

describe("SampleClips", () => {
  const props = {
    selected: null,
    loadingId: null,
    disabled: false,
    onSelect: vi.fn(),
  };

  it("offers every catalogue clip", () => {
    render(<SampleClips {...props} />);
    for (const sample of AUDIO_SAMPLES) {
      expect(
        screen.getByRole("button", { name: sample.label }),
      ).toBeInTheDocument();
    }
  });

  // Picking a clip loads it into the input. It does *not* transcribe it: with
  // three clips on screen, one of them 60 s long, "browse the samples" used to
  // mean "spend three inferences".
  it("selects the clip the user picked, and starts nothing", async () => {
    const onSelect = vi.fn();
    render(<SampleClips {...props} onSelect={onSelect} />);

    const first = AUDIO_SAMPLES[0];
    await userEvent.click(screen.getByRole("button", { name: first.label }));
    expect(onSelect).toHaveBeenCalledWith(first);
  });

  it("shows the reference transcript so the output can be eyeballed", () => {
    const withRef = AUDIO_SAMPLES.find((s) => s.reference);
    if (!withRef?.reference) return; // catalogue has none — nothing to assert
    render(<SampleClips {...props} selected={withRef} />);
    expect(screen.getByText(/reference:/i)).toBeInTheDocument();
    expect(screen.getByText(new RegExp(withRef.reference.slice(0, 20)))).toBeInTheDocument();
  });

  it("explains what to check for a clip with no fixed reference", () => {
    const noRef = AUDIO_SAMPLES.find((s) => !s.reference);
    if (!noRef) return;
    render(<SampleClips {...props} selected={noRef} />);
    expect(screen.getByText(/coherent English/i)).toBeInTheDocument();
  });

  it("disables the clips until a model can run them", () => {
    render(<SampleClips {...props} disabled />);
    for (const sample of AUDIO_SAMPLES) {
      expect(
        screen.getByRole("button", { name: sample.label }),
      ).toBeDisabled();
    }
  });
});
