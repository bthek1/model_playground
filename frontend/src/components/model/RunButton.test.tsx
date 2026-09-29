import { fireEvent, render, screen } from "@testing-library/react";
import { Tags } from "lucide-react";
import { describe, expect, it, vi } from "vitest";

import { RunButton } from "./RunButton";

describe("RunButton", () => {
  it("shows the verb while idle and the running label while a run is in flight", () => {
    const { rerender } = render(
      <RunButton icon={Tags} running={false} runningLabel="Classifying…" onRun={vi.fn()}>
        Classify
      </RunButton>,
    );
    expect(screen.getByRole("button", { name: "Classify" })).toBeInTheDocument();

    rerender(
      <RunButton icon={Tags} running runningLabel="Classifying…" onRun={vi.fn()}>
        Classify
      </RunButton>,
    );
    expect(screen.getByRole("button", { name: "Classifying…" })).toBeInTheDocument();
    expect(screen.getByRole("button").querySelector(".animate-spin")).not.toBeNull();
  });

  it("renders a bare verb when given no icon", () => {
    render(
      <RunButton running={false} runningLabel="Computing…" onRun={vi.fn()}>
        Compute
      </RunButton>,
    );
    const button = screen.getByRole("button");
    expect(button.textContent).toBe("Compute");
    expect(button.querySelector("svg")).toBeNull();
  });

  it("runs on click, and not while disabled", () => {
    const onRun = vi.fn();
    const { rerender } = render(
      <RunButton icon={Tags} running={false} runningLabel="…" onRun={onRun} disabled>
        Classify
      </RunButton>,
    );
    fireEvent.click(screen.getByRole("button"));
    expect(onRun).not.toHaveBeenCalled();

    rerender(
      <RunButton icon={Tags} running={false} runningLabel="…" onRun={onRun}>
        Classify
      </RunButton>,
    );
    fireEvent.click(screen.getByRole("button"));
    expect(onRun).toHaveBeenCalledOnce();
  });

  it("swallows a rejected run — the hook reports it in OUTPUT", async () => {
    const unhandled = vi.fn();
    process.on("unhandledRejection", unhandled);
    try {
      render(
        <RunButton
          icon={Tags}
          running={false}
          runningLabel="…"
          onRun={() => Promise.reject(new Error("boom"))}
        >
          Classify
        </RunButton>,
      );
      fireEvent.click(screen.getByRole("button"));
      await new Promise((r) => setTimeout(r, 0));
      expect(unhandled).not.toHaveBeenCalled();
    } finally {
      process.off("unhandledRejection", unhandled);
    }
  });

  it("passes Button props through", () => {
    render(
      <RunButton
        icon={Tags}
        running={false}
        runningLabel="…"
        onRun={vi.fn()}
        variant="outline"
        title="Ask the model"
        data-testid="search"
      >
        Search
      </RunButton>,
    );
    const button = screen.getByTestId("search");
    expect(button).toHaveAttribute("title", "Ask the model");
  });
});
