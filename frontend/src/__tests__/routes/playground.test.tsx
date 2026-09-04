import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    createFileRoute: vi
      .fn()
      .mockImplementation(
        (path: string) => (opts: Record<string, unknown>) => ({ path, options: opts }),
      ),
  };
});

// The catalog card fetches from the registry API; this route's own content is
// what's under test.
vi.mock("@/components/home/ModelCatalogCard", () => ({
  ModelCatalogCard: () => <div data-testid="model-catalog" />,
}));

const { Route } = await import("@/routes/playground");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Playground route component not found");
  render(<Page />);
}

// The playground is where someone lands before they know how any of this works,
// so it teaches the pipeline every task route implements
// (docs/standards/model-page-pattern.md).
describe("PlaygroundPage", () => {
  it("renders the heading and the model catalog", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /playground/i }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("model-catalog")).toBeInTheDocument();
  });

  it("explains all four stages, in order", () => {
    renderPage();
    const steps = screen.getAllByRole("listitem");
    expect(steps).toHaveLength(4);
    expect(steps.map((li) => li.textContent)).toEqual([
      expect.stringContaining("Select"),
      expect.stringContaining("Load"),
      expect.stringContaining("Run"),
      expect.stringContaining("Output"),
    ]);
  });

  it("states the deferred-load promise, since that is the surprising one", () => {
    // Users arriving from another AI playground expect a download on arrival.
    renderPage();
    expect(screen.getByText(/nothing downloads until you ask/i)).toBeInTheDocument();
  });

  it("tells the user where to go next, including for unbuilt tasks", () => {
    renderPage();
    expect(screen.getByText(/pick a task from the sidebar/i)).toBeInTheDocument();
    expect(screen.getByText(/same page with empty steps/i)).toBeInTheDocument();
  });
});
