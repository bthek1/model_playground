import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    useRouterState: vi.fn().mockReturnValue("/playground"),
    Link: ({
      children,
      to,
      ...props
    }: {
      children: React.ReactNode;
      to: string;
      [key: string]: unknown;
    }) => (
      <a href={to as string} {...props}>
        {children}
      </a>
    ),
  };
});

import { Sidebar, SidebarNav } from "./Sidebar";
import { useUIStore } from "@/store/ui";

describe("SidebarNav", () => {
  it("renders all nav items", () => {
    render(<SidebarNav />);
    expect(screen.getByText("Playground")).toBeInTheDocument();
  });

  it('marks the active link with aria-current="page"', () => {
    render(<SidebarNav />);
    const link = screen.getByText("Playground").closest("a");
    expect(link).toHaveAttribute("aria-current", "page");
  });

  it("renders icon-only labels as title attributes when collapsed", () => {
    render(<SidebarNav collapsed />);
    expect(screen.queryByText("Playground")).not.toBeInTheDocument();
    const link = screen.getByTitle("Playground");
    expect(link).toBeInTheDocument();
  });
});

describe("Sidebar", () => {
  afterEach(() => {
    // reset to the store default so tests don't leak state
    useUIStore.setState({ sidebarOpen: true });
  });

  it("renders with full width class when open", () => {
    // sidebarOpen defaults to true in the store
    const { container } = render(<Sidebar />);
    const aside = container.querySelector("aside");
    expect(aside?.className).toContain("w-64");
  });

  it("renders the full brand name when open", () => {
    render(<Sidebar />);
    expect(screen.getByText("Model Playground")).toBeInTheDocument();
  });

  it("renders the collapsed brand initials when closed", () => {
    useUIStore.setState({ sidebarOpen: false });
    const { container } = render(<Sidebar />);
    expect(screen.getByText("MP")).toBeInTheDocument();
    expect(screen.queryByText("Model Playground")).not.toBeInTheDocument();
    expect(container.querySelector("aside")?.className).toContain("w-16");
  });
});
