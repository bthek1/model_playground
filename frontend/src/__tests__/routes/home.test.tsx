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

vi.mock("@/hooks/useWebGPU", () => ({
  useWebGPU: () => ({ capabilities: null, loading: false }),
}));

vi.mock("@/hooks/useGpuBenchmark", () => ({
  useGpuBenchmark: () => ({
    running: false,
    result: null,
    error: null,
    run: vi.fn(),
  }),
}));

vi.mock("@/hooks/useModels", () => ({
  useModels: () => ({ data: [], isLoading: false, isError: false }),
}));

const { Route } = await import("@/routes/home");
const HomePage = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!HomePage) throw new Error("Home route component not found");
  render(<HomePage />);
}

describe("HomePage", () => {
  it("renders the page heading", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /model playground/i }),
    ).toBeInTheDocument();
  });

  it("surfaces the GPU capabilities, benchmark, and catalog cards", () => {
    renderPage();
    expect(screen.getByText("GPU Capabilities")).toBeInTheDocument();
    expect(screen.getByText("Compute Benchmark")).toBeInTheDocument();
    expect(screen.getByText("Model Catalog")).toBeInTheDocument();
  });
});
