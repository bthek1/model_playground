import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Button } from "@/components/ui/button";
import type { TrainingSettings } from "@/hooks/useLinearTraining";

import { DatasetDialog } from "./DatasetDialog";

const SETTINGS: TrainingSettings = {
  learningRate: 0.5,
  batchSize: 64,
  epochs: 10,
  trainSize: 8000,
  testSize: 2000,
};

type Props = Parameters<typeof DatasetDialog>[0];

function renderDialog(overrides: Partial<Props> = {}) {
  const set = vi.fn();
  const loadData = vi.fn();
  render(
    <DatasetDialog
      trigger={<Button>Dataset</Button>}
      settings={SETTINGS}
      set={set}
      disabled={false}
      datasetStatus="idle"
      datasetProgress={0}
      datasetError={null}
      poolCount={0}
      loadData={loadData}
      {...overrides}
    />,
  );
  return { set, loadData };
}

function open() {
  fireEvent.click(screen.getByRole("button", { name: /^dataset$/i }));
}

describe("DatasetDialog", () => {
  it("opens on trigger click", () => {
    renderDialog();
    expect(screen.queryByText(/fetched from a cdn/i)).not.toBeInTheDocument();
    open();
    expect(screen.getByText(/fetched from a cdn/i)).toBeInTheDocument();
  });

  it("loads MNIST for the combined train+test size", () => {
    const { loadData } = renderDialog();
    open();
    fireEvent.click(screen.getByRole("button", { name: /load mnist/i }));
    expect(loadData).toHaveBeenCalledWith(
      SETTINGS.trainSize + SETTINGS.testSize,
    );
  });

  it("edits the train-size field through set()", () => {
    const { set } = renderDialog();
    open();
    fireEvent.change(screen.getByDisplayValue("8000"), {
      target: { value: "5000" },
    });
    expect(set).toHaveBeenCalledWith("trainSize", 5000);
  });

  it("shows a ready summary with the loaded count", () => {
    renderDialog({ datasetStatus: "ready", poolCount: 12345 });
    open();
    expect(screen.getByText(/loaded 12,345 images/i)).toBeInTheDocument();
  });

  it("surfaces a dataset error", () => {
    renderDialog({ datasetError: "could not load sprite" });
    open();
    expect(screen.getByText(/could not load sprite/i)).toBeInTheDocument();
  });

  it("disables the load button while loading", () => {
    renderDialog({ datasetStatus: "loading", datasetProgress: 0.4 });
    open();
    expect(screen.getByRole("button", { name: /loading/i })).toBeDisabled();
  });
});
