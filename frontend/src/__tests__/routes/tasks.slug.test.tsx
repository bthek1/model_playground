import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

let mockParams: { slug: string } = { slug: "text-to-speech" };

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    createFileRoute: vi
      .fn()
      .mockImplementation((path: string) => (opts: Record<string, unknown>) => ({
        path,
        options: opts,
        useParams: () => mockParams,
      })),
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

const { Route } = await import("@/routes/tasks.$slug");
const TaskPage = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!TaskPage) throw new Error("Task route component not found");
  render(<TaskPage />);
}

describe("TaskPlaceholderPage", () => {
  beforeEach(() => {
    mockParams = { slug: "text-to-speech" };
  });

  it("renders the task label and its owning category for a known slug", () => {
    renderPage();
    // Appears in both the title and the body sentence.
    expect(screen.getAllByText("Text to Speech").length).toBeGreaterThan(0);
    expect(screen.getByText("Audio")).toBeInTheDocument();
    expect(
      screen.getByText(/isn't available in the playground yet/i),
    ).toBeInTheDocument();
  });

  it("renders an unknown-task fallback for a slug not in the taxonomy", () => {
    mockParams = { slug: "not-a-real-task" };
    renderPage();
    expect(screen.getByText(/unknown task/i)).toBeInTheDocument();
    expect(screen.getByText("not-a-real-task")).toBeInTheDocument();
  });

  it("renders a placeholder for a Theory task without a real route", () => {
    // Discrete Maths has no implemented route, so it falls through to here even
    // though its Theory siblings (Linear Model Training, Tensor Arithmetic) do.
    mockParams = { slug: "discrete-maths" };
    renderPage();
    expect(screen.getAllByText("Discrete Maths").length).toBeGreaterThan(0);
    expect(screen.getByText("Theory")).toBeInTheDocument();
    expect(
      screen.getByText(/isn't available in the playground yet/i),
    ).toBeInTheDocument();
  });
});
