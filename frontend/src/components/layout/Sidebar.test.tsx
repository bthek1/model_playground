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

const DEFAULT_UI = { sidebarOpen: true, expandedCategories: {} };

describe("SidebarNav", () => {
  afterEach(() => {
    useUIStore.setState(DEFAULT_UI);
  });

  it("renders a top-level Home link", () => {
    render(<SidebarNav />);
    const home = screen.getByText("Home").closest("a");
    expect(home).toHaveAttribute("href", "/home");
  });

  it("renders every category header", () => {
    render(<SidebarNav />);
    for (const label of [
      "Audio",
      "Computer Vision",
      "Multimodal",
      "Natural Language Processing",
      "Other",
      "Reinforcement Learning",
      "Tabular",
      "Theory",
    ]) {
      expect(screen.getByRole("button", { name: new RegExp(label) })).toBeInTheDocument();
    }
  });

  it("hides a category's tasks until it is expanded", () => {
    const { rerender } = render(<SidebarNav />);
    // Audio starts collapsed
    expect(screen.queryByText("Text to Speech")).not.toBeInTheDocument();

    useUIStore.setState({ expandedCategories: { Audio: true } });
    rerender(<SidebarNav />);
    expect(screen.getByText("Text to Speech")).toBeInTheDocument();
  });

  it("auto-expands the category owning the active route", () => {
    // pathname is mocked to /playground, which Text Generation (NLP) maps to
    render(<SidebarNav />);
    expect(screen.getByText("Text Generation")).toBeInTheDocument();
  });

  it('marks the active task with aria-current="page"', () => {
    render(<SidebarNav />);
    const link = screen.getByText("Text Generation").closest("a");
    expect(link).toHaveAttribute("aria-current", "page");
  });

  it("renders category icons with titles and no task labels when collapsed", () => {
    render(<SidebarNav collapsed />);
    expect(screen.queryByText("Text to Speech")).not.toBeInTheDocument();
    expect(screen.getByTitle("Audio")).toBeInTheDocument();
  });

  it("links the Theory tools to their real routes when expanded", () => {
    useUIStore.setState({ expandedCategories: { Theory: true } });
    render(<SidebarNav />);
    const training = screen.getByText("Linear Model Training").closest("a");
    const tensor = screen.getByText("Tensor Arithmetic").closest("a");
    expect(training).toHaveAttribute("href", "/training");
    expect(tensor).toHaveAttribute("href", "/tensor");
  });
});

describe("Sidebar", () => {
  afterEach(() => {
    useUIStore.setState(DEFAULT_UI);
  });

  it("renders with full width class when open", () => {
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
