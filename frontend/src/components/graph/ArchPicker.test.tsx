import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { GNN_ARCHITECTURES } from "@/webgpu/gnn";

import { ArchPicker } from "./ArchPicker";

describe("ArchPicker", () => {
  it("offers the four architectures, marks the selected one, and reports a pick", () => {
    const onChange = vi.fn();
    render(<ArchPicker value="gcn" onChange={onChange} disabled={false} />);
    const buttons = screen.getAllByRole("button");
    expect(buttons).toHaveLength(4);
    expect(buttons[0]).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(buttons[3]);
    expect(onChange).toHaveBeenCalledWith("gat");
  });

  it("prints each architecture's note only when asked", () => {
    const note = GNN_ARCHITECTURES.gcn.note;
    const { rerender } = render(
      <ArchPicker value="gcn" onChange={vi.fn()} disabled={false} />,
    );
    expect(screen.queryByText(note)).toBeNull();
    rerender(<ArchPicker value="gcn" onChange={vi.fn()} disabled={false} showNotes />);
    expect(screen.getByText(note)).toBeInTheDocument();
  });
});
