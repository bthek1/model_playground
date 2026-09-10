import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PhraseList } from "./PhraseList";

function renderList(
  props: Partial<Parameters<typeof PhraseList>[0]> = {},
) {
  const onChange = vi.fn();
  const view = render(
    <PhraseList
      id="new-label"
      title="Labels"
      items={["cat", "dog"]}
      onChange={onChange}
      {...props}
    />,
  );
  return { ...view, onChange };
}

describe("PhraseList", () => {
  it("labels its field, so a route test can find it by name", () => {
    // The wiring most likely to be dropped on a copy — and the reason this was
    // extracted at the second caller rather than the third.
    renderList();
    expect(screen.getByLabelText("Labels")).toBeInstanceOf(HTMLInputElement);
  });

  it("shows every item as its own chip", () => {
    renderList();
    expect(screen.getByText("cat")).toBeInTheDocument();
    expect(screen.getByText("dog")).toBeInTheDocument();
  });

  it("gives each remove button its own name", () => {
    // "Remove" twelve times over is unusable by voice or by screen reader.
    const { onChange } = renderList();
    fireEvent.click(screen.getByRole("button", { name: "Remove dog" }));
    expect(onChange).toHaveBeenCalledWith(["cat"]);
  });

  it("adds the trimmed draft, and clears the field", () => {
    const { onChange } = renderList();
    const field = screen.getByLabelText("Labels");
    fireEvent.change(field, { target: { value: "  a bicycle  " } });
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));

    expect(onChange).toHaveBeenCalledWith(["cat", "dog", "a bicycle"]);
    expect(field).toHaveValue("");
  });

  it("adds on Enter without submitting anything", () => {
    // These lists sit inside form-shaped layouts; Enter must add a chip, never
    // navigate away from a half-finished list.
    const { onChange } = renderList();
    fireEvent.change(screen.getByLabelText("Labels"), {
      target: { value: "a bicycle" },
    });
    const event = fireEvent.keyDown(screen.getByLabelText("Labels"), {
      key: "Enter",
    });

    expect(onChange).toHaveBeenCalledWith(["cat", "dog", "a bicycle"]);
    // `fireEvent` returns false when the handler called `preventDefault`.
    expect(event).toBe(false);
  });

  it("ignores a blank draft", () => {
    const { onChange } = renderList();
    fireEvent.change(screen.getByLabelText("Labels"), {
      target: { value: "   " },
    });
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));
    expect(onChange).not.toHaveBeenCalled();
  });

  it("ignores a duplicate silently, without a rejection message", () => {
    // The model would score the same string twice and the second row would be
    // noise — but telling someone off for typing a word already on screen is
    // worse than doing nothing.
    const { onChange } = renderList();
    fireEvent.change(screen.getByLabelText("Labels"), {
      target: { value: "cat" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("says what to do when the list is empty", () => {
    renderList({ items: [], emptyHint: "Add at least one query." });
    expect(screen.getByText("Add at least one query.")).toBeInTheDocument();
  });

  it("locks every control while a run is in flight", () => {
    renderList({ disabled: true });
    expect(screen.getByLabelText("Labels")).toBeDisabled();
    expect(screen.getByRole("button", { name: /^add$/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Remove cat" })).toBeDisabled();
  });

  it("renders the caller's hint under the field", () => {
    renderList({ hint: "Sent to the model exactly as written." });
    expect(
      screen.getByText(/sent to the model exactly as written/i),
    ).toBeInTheDocument();
  });

  it("owns only the draft, so the list stays the caller's state", () => {
    // A controlled component: re-rendering with new items must show them, and
    // must not resurrect a draft the caller has already consumed.
    const { rerender, onChange } = renderList();
    fireEvent.change(screen.getByLabelText("Labels"), {
      target: { value: "fox" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^add$/i }));
    expect(onChange).toHaveBeenCalledWith(["cat", "dog", "fox"]);

    rerender(
      <PhraseList
        id="new-label"
        title="Labels"
        items={["cat", "dog", "fox"]}
        onChange={onChange}
      />,
    );
    expect(screen.getByText("fox")).toBeInTheDocument();
    expect(screen.getByLabelText("Labels")).toHaveValue("");
  });
});
