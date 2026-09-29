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
    // The category now leads the description sentence rather than sitting in
    // its own eyebrow line.
    expect(screen.getByText(/^Audio ·/)).toBeInTheDocument();
    expect(screen.getByText(/not available yet/i)).toBeInTheDocument();
  });

  it("renders an unknown-task fallback for a slug not in the taxonomy", () => {
    mockParams = { slug: "not-a-real-task" };
    renderPage();
    expect(screen.getByText(/unknown task/i)).toBeInTheDocument();
    expect(screen.getByText("not-a-real-task")).toBeInTheDocument();
  });

  it("renders a placeholder for a task without a real route", () => {
    // Table Question Answering does not port (nlp.md), so it falls through to
    // here even though every other row in its category has a page. (This used
    // to be Discrete Maths, until #56 gave Theory's last row a route.)
    mockParams = { slug: "table-question-answering" };
    renderPage();
    expect(screen.getAllByText("Table Question Answering").length).toBeGreaterThan(0);
    expect(screen.getByText(/^Natural Language Processing ·/)).toBeInTheDocument();
    expect(
      screen.getByText(/not available yet/i),
    ).toBeInTheDocument();
  });
});
