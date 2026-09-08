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
  it("keeps the transport directly after the fields, not pushed to the foot", () => {
    // The transport row is sticky, not bottom-pinned. `mt-auto` reads fine for a
    // full-height textarea and opens a chasm on a task whose input is three
    // buttons, so the controls stay in flow immediately after the fields and
    // only detach once a tall input scrolls them out of reach.
    const { container } = render(
      <InputPanel
        ready
        controls={<button type="button">Speak</button>}
        error="Bad shape"
      >
        <textarea aria-label="Text" />
      </InputPanel>,
    );
    const root = container.firstElementChild!;
    // Two children: the task's fields, then the transport group.
    expect(root.children).toHaveLength(2);
    expect(root.children[0]).toContainElement(screen.getByLabelText("Text"));

    const transport = root.children[1];
    expect(transport).toContainElement(
      screen.getByRole("button", { name: "Speak" }),
    );
    // The error belongs to the transport group, so it travels with the controls
    // rather than being stranded above a scrolled input.
    expect(transport).toContainElement(screen.getByRole("alert"));
    expect(transport.className).not.toMatch(/\bmt-auto\b/);
  });

  it("stretches to fill its workbench column", () => {
    // ModelPage hands the band's stretch down to whatever the route put in it;
    // without flex-1 here a textarea cannot grow to the column height.
    const { container } = render(
      <InputPanel ready controls={<button type="button">Speak</button>} />,
    );
    expect(container.firstElementChild?.className).toMatch(/\bflex-1\b/);
  });

  it("attaches that reason to the controls, not just to the page", () => {
    render(
      <InputPanel
        ready={false}
        disabledHint="Load a model to synthesise speech."
        controls={<button type="button" disabled>Speak</button>}
      />,
    );
    // A greyed-out button with the reason floating nearby is a dead end for
    // anyone driving the page by screen reader.
    const hint = screen.getByText(/load a model to synthesise/i);
    const controls = screen.getByRole("button", { name: "Speak" }).parentElement;
    expect(controls).toHaveAttribute("aria-describedby", hint.id);
  });

  it("drops the description once the controls actually work", () => {
    render(
      <InputPanel ready controls={<button type="button">Speak</button>} />,
    );
    expect(
      screen.getByRole("button", { name: "Speak" }).parentElement,
    ).not.toHaveAttribute("aria-describedby");
  });
});
