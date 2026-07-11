import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ModelArchitecture } from "./ModelArchitecture";

describe("ModelArchitecture", () => {
  it("shows the title and parameter budget", () => {
    render(<ModelArchitecture weights={new Float32Array(784 * 10)} />);
    expect(screen.getByText(/linear softmax classifier/i)).toBeInTheDocument();
    // 784×10 = 7,840 weights, +10 biases = 7,850 total.
    expect(screen.getByText("7,840")).toBeInTheDocument();
    expect(screen.getByText("7,850")).toBeInTheDocument();
  });

  it("shows the input as a 28×28 pixel grid", () => {
    render(<ModelArchitecture weights={new Float32Array(784 * 10)} />);
    expect(
      screen.getByLabelText(/28×28 input pixel grid/i),
    ).toBeInTheDocument();
  });

  it("renders a 28×28 weight template for every class", () => {
    render(
      <ModelArchitecture
        weights={new Float32Array(784 * 10)}
        bias={new Float32Array(10)}
      />,
    );
    // Every parameter is shown as a per-class 28×28 template (one canvas each).
    expect(
      screen.getAllByLabelText(/weight template for digit/i),
    ).toHaveLength(10);
    expect(
      screen.getByLabelText(/weight template for digit 0/i),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(/weight template for digit 9/i),
    ).toBeInTheDocument();
  });

  it("renders the templates at the enlarged tile size", () => {
    render(<ModelArchitecture weights={new Float32Array(784 * 10)} />);
    // The schematic scales the templates up to size-32 (128px).
    expect(screen.getByLabelText(/weight template for digit 0/i)).toHaveClass(
      "size-32",
    );
  });

  it("prompts to train until the weights carry signal", () => {
    // All-zero weights (pre-training) → the templates are blank, so show a hint.
    render(<ModelArchitecture weights={new Float32Array(784 * 10)} />);
    expect(screen.getByText(/start training to watch/i)).toBeInTheDocument();
  });
});
