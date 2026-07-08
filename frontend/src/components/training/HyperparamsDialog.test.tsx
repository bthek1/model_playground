import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Button } from "@/components/ui/button";
import type { TrainingSettings } from "@/hooks/useLinearTraining";

import { HyperparamsDialog } from "./HyperparamsDialog";

const SETTINGS: TrainingSettings = {
  learningRate: 0.5,
  batchSize: 64,
  epochs: 10,
  trainSize: 8000,
  testSize: 2000,
};

function renderDialog(overrides: Partial<Parameters<typeof HyperparamsDialog>[0]> = {}) {
  const set = vi.fn();
  render(
    <HyperparamsDialog
      trigger={<Button>Tune</Button>}
      settings={SETTINGS}
      set={set}
      disabled={false}
      {...overrides}
    />,
  );
  return { set };
}

describe("HyperparamsDialog", () => {
  it("stays closed until the trigger is clicked", () => {
    renderDialog();
    expect(screen.queryByText("Hyperparameters")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /tune/i }));
    expect(screen.getByText("Hyperparameters")).toBeInTheDocument();
  });

  it("edits a field through set()", () => {
    const { set } = renderDialog();
    fireEvent.click(screen.getByRole("button", { name: /tune/i }));
    const [lr] = screen.getAllByRole("spinbutton");
    fireEvent.change(lr, { target: { value: "0.25" } });
    expect(set).toHaveBeenCalledWith("learningRate", 0.25);
  });

  it("rounds integer fields", () => {
    const { set } = renderDialog();
    fireEvent.click(screen.getByRole("button", { name: /tune/i }));
    const epochs = screen.getByDisplayValue("10");
    fireEvent.change(epochs, { target: { value: "7.9" } });
    expect(set).toHaveBeenCalledWith("epochs", 8);
  });

  it("locks fields while training", () => {
    renderDialog({ disabled: true });
    fireEvent.click(screen.getByRole("button", { name: /tune/i }));
    for (const field of screen.getAllByRole("spinbutton")) {
      expect(field).toBeDisabled();
    }
  });
});
