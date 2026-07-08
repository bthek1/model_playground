import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { NumberField, pct, Stat } from "./controls";

describe("pct", () => {
  it("formats a fraction as a 2-dp percentage", () => {
    expect(pct(0)).toBe("0.00%");
    expect(pct(0.5)).toBe("50.00%");
    expect(pct(0.1234)).toBe("12.34%");
    expect(pct(1)).toBe("100.00%");
  });
});

describe("NumberField", () => {
  it("renders its label and current value", () => {
    render(
      <NumberField
        label="Learning rate"
        value={0.5}
        step={0.05}
        min={0.001}
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByText("Learning rate")).toBeInTheDocument();
    expect(screen.getByRole("spinbutton")).toHaveValue(0.5);
  });

  it("calls onChange with the parsed number", () => {
    const onChange = vi.fn();
    render(
      <NumberField
        label="Epochs"
        value={10}
        step={1}
        min={1}
        onChange={onChange}
      />,
    );
    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "12" } });
    expect(onChange).toHaveBeenCalledWith(12);
  });

  it("passes fractional values through unrounded", () => {
    const onChange = vi.fn();
    render(
      <NumberField
        label="Learning rate"
        value={0.5}
        step={0.05}
        min={0.001}
        onChange={onChange}
      />,
    );
    fireEvent.change(screen.getByRole("spinbutton"), {
      target: { value: "0.35" },
    });
    expect(onChange).toHaveBeenCalledWith(0.35);
  });

  it("is disabled when disabled is set", () => {
    render(
      <NumberField
        label="Epochs"
        value={10}
        step={1}
        min={1}
        disabled
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByRole("spinbutton")).toBeDisabled();
  });
});

describe("Stat", () => {
  it("renders label, value, and optional sub", () => {
    render(<Stat label="Test acc" value="92.10%" sub="after epoch 3" />);
    expect(screen.getByText("Test acc")).toBeInTheDocument();
    expect(screen.getByText("92.10%")).toBeInTheDocument();
    expect(screen.getByText("after epoch 3")).toBeInTheDocument();
  });

  it("omits the sub line when not provided", () => {
    const { container } = render(<Stat label="Epoch" value="1" />);
    // label + value only — no third paragraph.
    expect(container.querySelectorAll("p")).toHaveLength(2);
  });
});
