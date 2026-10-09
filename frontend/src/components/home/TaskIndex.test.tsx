import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-router", async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  Link: ({ children, to }: { children: React.ReactNode; to: string }) => <a href={to}>{children}</a>,
}));

import { taskCategories } from "@/components/layout/taskTaxonomy";
import { TaskIndex } from "./TaskIndex";

describe("TaskIndex", () => {
  it("links every task in the taxonomy, with its description", () => {
    render(<TaskIndex />);
    for (const t of taskCategories.flatMap((c) => c.tasks)) {
      expect(screen.getByRole("link", { name: t.label })).toHaveAttribute("href", t.to);
      expect(screen.getByText(t.description)).toBeInTheDocument();
    }
  });

  // The breadcrumbs and the structured data link to `/home#<category>`.
  it("anchors each category where the breadcrumbs point", () => {
    const { container } = render(<TaskIndex />);
    const vision = container.querySelector("#computer-vision") as HTMLElement;
    expect(within(vision).getByRole("heading", { name: "Computer Vision" })).toBeInTheDocument();
    expect(within(vision).getByRole("link", { name: "Depth Estimation" })).toBeInTheDocument();
  });
});
