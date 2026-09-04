import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { InputPanel } from "./InputPanel";

describe("InputPanel", () => {
  it("renders the task's fields above its transport controls", () => {
    render(
      <InputPanel ready controls={<button type="button">Speak</button>}>
        <textarea aria-label="Text" />
      </InputPanel>,
    );
    expect(screen.getByLabelText("Text")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Speak" })).toBeInTheDocument();
  });

  it("explains why the controls are dead before a model is loaded", () => {
    render(
      <InputPanel
        ready={false}
        controls={<button type="button" disabled>Speak</button>}
      />,
    );
    expect(screen.getByText(/load a model/i)).toBeInTheDocument();
  });

  it("drops the hint once ready", () => {
    render(<InputPanel ready controls={<button type="button">Speak</button>} />);
    expect(screen.queryByText(/load a model/i)).toBeNull();
  });

  it("shows input-side errors here rather than in the output panel", () => {
    render(
      <InputPanel
        ready
        controls={<button type="button">Speak</button>}
        error="Microphone permission denied"
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Microphone permission denied",
    );
  });
});
