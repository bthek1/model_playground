import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { sizeEstimate } from "@/audio/size";

import { ModelStatus } from "./ModelStatus";

const base = { backend: null, progress: null } as const;

describe("ModelStatus", () => {
  describe("idle — nothing has downloaded yet", () => {
    it("offers the load action and quotes the download first", () => {
      render(
        <ModelStatus
          {...base}
          status="idle"
          size={sizeEstimate(74)}
          onLoad={() => {}}
        />,
      );
      expect(
        screen.getByRole("button", { name: /load model/i }),
      ).toBeEnabled();
      // The estimate is the point of the idle state — it must precede the click.
      expect(screen.getByText(/on WebGPU/)).toBeInTheDocument();
    });

    it("fires onLoad", async () => {
      const onLoad = vi.fn();
      render(<ModelStatus {...base} status="idle" onLoad={onLoad} />);
      await userEvent.click(screen.getByRole("button", { name: /load model/i }));
      expect(onLoad).toHaveBeenCalledOnce();
    });

    it("disables the action when the page says so", () => {
      render(<ModelStatus {...base} status="idle" onLoad={() => {}} disabled />);
      expect(screen.getByRole("button", { name: /load model/i })).toBeDisabled();
    });
  });

  describe("loading", () => {
    it("shows the file and percentage", () => {
      render(
        <ModelStatus
          {...base}
          status="loading"
          progress={{ status: "progress", file: "encoder.onnx", progress: 42 }}
        />,
      );
      expect(screen.getByText(/encoder\.onnx/)).toBeInTheDocument();
      expect(screen.getByText(/42%/)).toBeInTheDocument();
    });

    it("distinguishes warm-up from downloading", () => {
      render(
        <ModelStatus {...base} status="loading" progress={{ status: "warmup" }} />,
      );
      expect(screen.getByText(/warming up/i)).toBeInTheDocument();
    });
  });

  it("names the resolved backend when ready", () => {
    render(<ModelStatus {...base} status="ready" backend="webgpu" />);
    expect(screen.getByTestId("model-ready")).toHaveTextContent(/webgpu/i);
  });

  describe("error — a load failure, recoverable", () => {
    it("shows the message and a retry", async () => {
      const onRetry = vi.fn();
      render(
        <ModelStatus
          {...base}
          status="error"
          error="Unauthorized access to file"
          onRetry={onRetry}
        />,
      );
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Unauthorized access to file",
      );
      await userEvent.click(screen.getByRole("button", { name: /retry/i }));
      expect(onRetry).toHaveBeenCalledOnce();
    });

    it("still explains itself with no message from the worker", () => {
      render(<ModelStatus {...base} status="error" />);
      expect(screen.getByRole("alert")).toHaveTextContent(/failed to load/i);
    });
  });
});
