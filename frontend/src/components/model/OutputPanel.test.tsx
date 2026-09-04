import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { OutputPanel } from "./OutputPanel";

const base = { title: "Speech", empty: "Press Speak to hear it." } as const;

describe("OutputPanel", () => {
  it("always renders, so the page cannot jump when a result lands", () => {
    render(<OutputPanel {...base} running={false} />);
    expect(screen.getByTestId("output-panel")).toBeInTheDocument();
  });

  it("describes the coming result while empty", () => {
    render(<OutputPanel {...base} running={false} />);
    expect(screen.getByTestId("output-empty")).toHaveTextContent(
      "Press Speak to hear it.",
    );
  });

  it("swaps the empty state for a spinner while running", () => {
    render(<OutputPanel {...base} running runningLabel="Synthesising…" />);
    expect(screen.getByTestId("output-running")).toHaveTextContent(
      "Synthesising…",
    );
    expect(screen.queryByTestId("output-empty")).not.toBeInTheDocument();
  });

  it("keeps the previous result visible while the next run is in flight", () => {
    render(
      <OutputPanel {...base} running>
        <div>waveform</div>
      </OutputPanel>,
    );
    expect(screen.getByText("waveform")).toBeInTheDocument();
    expect(screen.queryByTestId("output-running")).not.toBeInTheDocument();
  });

  it("shows result actions only alongside a result", () => {
    const actions = <button type="button">Download</button>;
    const { rerender } = render(
      <OutputPanel {...base} running={false} actions={actions} />,
    );
    expect(screen.queryByRole("button", { name: "Download" })).toBeNull();

    rerender(
      <OutputPanel {...base} running={false} actions={actions}>
        <div>waveform</div>
      </OutputPanel>,
    );
    expect(screen.getByRole("button", { name: "Download" })).toBeInTheDocument();
  });

  it("renders an inference error here, leaving the panel usable", () => {
    render(
      <OutputPanel {...base} running={false} error="Inference failed">
        <div>waveform</div>
      </OutputPanel>,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Inference failed");
    expect(screen.getByText("waveform")).toBeInTheDocument();
  });
});
