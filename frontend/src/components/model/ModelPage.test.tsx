import { render, screen, within } from "@testing-library/react";
import { Volume2 } from "lucide-react";
import { describe, expect, it } from "vitest";

import { ModelPage } from "./ModelPage";

// The shell's whole job is that the four slots are always present, always in
// order, and always labelled — the guarantee that makes every task page read the
// same way (model-page-pattern.md §4).
function renderPage(props: Partial<Parameters<typeof ModelPage>[0]> = {}) {
  return render(
    <ModelPage
      icon={Volume2}
      title="Text to Speech"
      description="Runs in your browser."
      select={<div>picker</div>}
      load={<div>status</div>}
      run={<div>controls</div>}
      output={<div>result</div>}
      {...props}
    />,
  );
}

describe("ModelPage", () => {
  it("renders the four slots in Select → Load → Run → Output order", () => {
    renderPage();
    const slots = screen.getAllByRole("region");
    expect(slots).toHaveLength(4);
    expect(slots.map((s) => s.dataset.testid)).toEqual([
      "slot-1",
      "slot-2",
      "slot-3",
      "slot-4",
    ]);
  });

  it("puts each slot's content inside its own band", () => {
    renderPage();
    expect(
      within(screen.getByTestId("slot-1")).getByText("picker"),
    ).toBeInTheDocument();
    expect(
      within(screen.getByTestId("slot-4")).getByText("result"),
    ).toBeInTheDocument();
  });

  it("labels the bands, so the stage is never ambiguous", () => {
    renderPage();
    for (const label of ["Model", "Load", "Input", "Output"]) {
      expect(screen.getByRole("heading", { name: label })).toBeInTheDocument();
    }
  });

  it("lets a task rename a band without changing the order", () => {
    renderPage({ labels: { run: "Text", output: "Speech" } });
    expect(screen.getByRole("heading", { name: "Text" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Speech" })).toBeInTheDocument();
    expect(screen.getAllByRole("region")[2]).toHaveAttribute(
      "data-testid",
      "slot-3",
    );
  });

  it("renders the OUTPUT band even when the task passes nothing", () => {
    renderPage({ output: null });
    expect(screen.getByTestId("slot-4")).toBeInTheDocument();
  });

  it("shows the title and description", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { level: 1, name: /text to speech/i }),
    ).toBeInTheDocument();
    expect(screen.getByText("Runs in your browser.")).toBeInTheDocument();
  });
});
