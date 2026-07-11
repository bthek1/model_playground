import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { TensorOpJob, TensorOpResult } from "@/webgpu/types";

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

const mockRun = vi.fn<(job: TensorOpJob) => Promise<void>>();
const mockReset = vi.fn();
const mockHookState = {
  running: false,
  result: null as TensorOpResult | null,
  error: null as string | null,
  run: mockRun,
  reset: mockReset,
};

vi.mock("@/hooks/useTensorOp", () => ({
  useTensorOp: () => mockHookState,
}));

const { Route } = await import("@/routes/tensor");
const TensorPage = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!TensorPage) throw new Error("Tensor route component not found");
  render(<TensorPage />);
}

describe("TensorArithmeticPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockHookState.running = false;
    mockHookState.result = null;
    mockHookState.error = null;
  });

  it("renders the page heading", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /tensor arithmetic/i }),
    ).toBeInTheDocument();
  });

  it("shows Matrix B for a binary op (add) and hides it for a unary op (transpose)", () => {
    renderPage();
    // Default op is "add" — both operands are shown.
    expect(screen.getByText("Matrix A")).toBeInTheDocument();
    expect(screen.getByText("Matrix B")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /transpose/i }));
    expect(screen.getByText("Matrix A")).toBeInTheDocument();
    expect(screen.queryByText("Matrix B")).not.toBeInTheDocument();
  });

  it("reveals the scalar input only for the scale op", () => {
    renderPage();
    expect(screen.queryByLabelText(/multiply every element/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /scale/i }));
    expect(screen.getByLabelText(/multiply every element/i)).toBeInTheDocument();
  });

  it("parses the operands and dispatches a job on Compute", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /^compute$/i }));

    expect(mockRun).toHaveBeenCalledTimes(1);
    const job = mockRun.mock.calls[0][0];
    expect(job.op).toBe("add");
    // Sample A/B default to a 2×3 grid.
    expect(job.aRows).toBe(2);
    expect(job.aCols).toBe(3);
    expect(job.bRows).toBe(2);
    expect(job.bCols).toBe(3);
    expect(Array.from(job.a)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("surfaces a parse error and does not dispatch when A is ragged", () => {
    renderPage();
    const textareas = screen.getAllByRole("textbox");
    fireEvent.change(textareas[0], { target: { value: "1 2 3\n4 5" } });

    fireEvent.click(screen.getByRole("button", { name: /^compute$/i }));

    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByText(/same length/i)).toBeInTheDocument();
  });

  it("renders the result grid and timing when a result is present", () => {
    // Use a 2×2 result so its shape badge is distinct from the 2×3 input badges.
    mockHookState.result = {
      data: new Float32Array([8, 10, 12, 14]),
      rows: 2,
      cols: 2,
      gpuTimeMs: 1.5,
    };
    renderPage();

    expect(screen.getByText(/2×2/)).toBeInTheDocument();
    expect(screen.getByText(/1\.50 ms/)).toBeInTheDocument();
    for (const value of ["8", "10", "12", "14"]) {
      expect(screen.getByText(value)).toBeInTheDocument();
    }
  });

  it("renders the result as a diverging heatmap with its legend", () => {
    mockHookState.result = {
      data: new Float32Array([8, 10, 12, 14]),
      rows: 2,
      cols: 2,
      gpuTimeMs: 1.5,
    };
    renderPage();

    // Canvas heatmap (per the model-visualization standard) + the value→colour
    // key that must accompany it.
    expect(screen.getByLabelText(/result value heatmap/i)).toBeInTheDocument();
    expect(screen.getByText(/positive/i)).toBeInTheDocument();
    expect(screen.getByText(/negative/i)).toBeInTheDocument();
  });

  it("lays out operands as a dataflow schematic with an op chip", () => {
    renderPage();
    // Op summary chip (accent ParamChip) reflects the selected operation.
    expect(screen.getByText("op")).toBeInTheDocument();
    expect(screen.getByText("binary")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /transpose/i }));
    expect(screen.getByText("unary")).toBeInTheDocument();
  });

  it("shows a compute error from the hook", () => {
    mockHookState.error = "No WebGPU adapter available";
    renderPage();
    expect(
      screen.getByText(/no webgpu adapter available/i),
    ).toBeInTheDocument();
  });
});
