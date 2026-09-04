import { render, screen, within } from "@testing-library/react";
import { Volume2 } from "lucide-react";
import { describe, expect, it } from "vitest";

import { ModelPage, ModelSlot } from "./ModelPage";

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
  // The layout is a grid placed by *area*, which is exactly the mechanism that
  // could silently reorder the pipeline. This is the assertion that stops it:
  // source order is pipeline order at every breakpoint, so Tab and a screen
  // reader always walk Model → Load → Input → Output.
  it("keeps DOM order equal to pipeline order regardless of grid placement", () => {
    renderPage();
    const headings = screen
      .getAllByRole("heading", { level: 2 })
      .map((h) => h.textContent);
    expect(headings).toEqual(["Model", "Load", "Input", "Output"]);
  });

  it("groups SELECT and LOAD into one setup surface, apart from the workbench", () => {
    // Setup is done once a session; input and output are touched every run.
    // The split is what lets the result sit beside the input instead of below it.
    renderPage();
    const rail = screen.getByTestId("slot-1").parentElement!;
    expect(rail).toContainElement(screen.getByTestId("slot-2"));
    expect(rail).not.toContainElement(screen.getByTestId("slot-3"));
    expect(rail).not.toContainElement(screen.getByTestId("slot-4"));
  });

  it("puts the INPUT and OUTPUT bands in separate workbench columns", () => {
    renderPage();
    expect(screen.getByTestId("slot-3").parentElement).not.toBe(
      screen.getByTestId("slot-4").parentElement,
    );
  });

  it("renders an aside in the header when the task has a one-line status", () => {
    // DeviceStatus-style answers don't deserve a band, but they do deserve
    // somewhere that isn't buried in the rail.
    renderPage({ aside: <span>WEBGPU</span> });
    expect(screen.getByText("WEBGPU")).toBeInTheDocument();
    // ...and it stays out of the four slots.
    for (const slot of ["slot-1", "slot-2", "slot-3", "slot-4"]) {
      expect(screen.getByTestId(slot)).not.toContainElement(
        screen.getByText("WEBGPU"),
      );
    }
  });

  it("omits the aside entirely when none is given", () => {
    const { container } = renderPage();
    expect(container.querySelector("header")?.children).toHaveLength(1);
  });
});

describe("ModelSlot", () => {
  it("renders the same structure dense as it does full", () => {
    // The rail uses `dense` for spacing only — the step number, the heading and
    // the region wrapper are the band's identity and must not vary with it.
    const { rerender } = render(
      <ModelSlot step={2} label="Load" dense>
        <div>status</div>
      </ModelSlot>,
    );
    const dense = screen.getByTestId("slot-2");
    expect(dense).toHaveAttribute("aria-labelledby", "slot-load");
    expect(screen.getByRole("heading", { name: "Load" })).toBeInTheDocument();
    expect(dense).toHaveTextContent("2");

    rerender(
      <ModelSlot step={2} label="Load">
        <div>status</div>
      </ModelSlot>,
    );
    expect(screen.getByTestId("slot-2")).toHaveAttribute(
      "aria-labelledby",
      "slot-load",
    );
    expect(screen.getByRole("heading", { name: "Load" })).toBeInTheDocument();
  });
});
