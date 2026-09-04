import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { LoadProgress } from "@/model/progress";

import { ModelStatus } from "./ModelStatus";

const base = { backend: null, loadProgress: null } as const;

const MB = 1024 * 1024;

function progress(over: Partial<LoadProgress> = {}): LoadProgress {
  return {
    phase: "downloading",
    percent: 42,
    loaded: 42 * MB,
    total: 100 * MB,
    files: { done: 1, count: 3 },
    current: "encoder.onnx",
    elapsedMs: 12_000,
    ...over,
  };
}

describe("ModelStatus", () => {
  describe("idle — nothing has downloaded yet", () => {
    it("offers the load action and says the weights are cached", () => {
      render(<ModelStatus {...base} status="idle" onLoad={() => {}} />);
      expect(screen.getByRole("button", { name: /load model/i })).toBeEnabled();
      expect(screen.getByText(/cached by the browser/i)).toBeInTheDocument();
      // The size-before-load estimate is ModelPicker's job — it sits directly
      // above this in the setup rail, so quoting it here duplicated the number.
      expect(screen.queryByText(/on WebGPU/)).toBeNull();
    });

    it("says so when the weights are already downloaded", () => {
      render(<ModelStatus {...base} status="idle" cached onLoad={() => {}} />);
      expect(
        screen.getByRole("button", { name: /load model \(cached\)/i }),
      ).toBeEnabled();
      expect(screen.getByText(/already downloaded/i)).toBeInTheDocument();
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
    it("reports the aggregate, not a single file's percentage", () => {
      render(
        <ModelStatus {...base} status="loading" loadProgress={progress()} />,
      );
      const bar = screen.getByRole("progressbar");
      expect(bar).toHaveAttribute("aria-valuenow", "42");
      expect(screen.getByText(/encoder\.onnx/)).toBeInTheDocument();
      expect(screen.getByText("42%")).toBeInTheDocument();
      expect(screen.getByText("42 MB / 100 MB")).toBeInTheDocument();
      expect(screen.getByText("1 of 3 files")).toBeInTheDocument();
      expect(screen.getByText("12s")).toBeInTheDocument();
    });

    it("goes indeterminate when no size is known yet", () => {
      render(
        <ModelStatus
          {...base}
          status="loading"
          loadProgress={progress({
            phase: "connecting",
            percent: null,
            loaded: 0,
            total: 0,
            files: { done: 0, count: 0 },
            current: undefined,
          })}
        />,
      );
      // No invented number for a screen reader to announce.
      expect(screen.getByRole("progressbar")).not.toHaveAttribute(
        "aria-valuenow",
      );
      expect(screen.getByText(/contacting the model host/i)).toBeInTheDocument();
    });

    it("distinguishes warm-up from downloading", () => {
      render(
        <ModelStatus
          {...base}
          status="loading"
          loadProgress={progress({ phase: "warmup", percent: null })}
        />,
      );
      expect(screen.getByText(/warming up/i)).toBeInTheDocument();
      expect(screen.getByRole("progressbar")).not.toHaveAttribute(
        "aria-valuenow",
      );
    });

    it("labels a cache-backed resume honestly", () => {
      render(
        <ModelStatus
          {...base}
          status="loading"
          restoring
          loadProgress={progress()}
        />,
      );
      expect(screen.getByText(/restoring from cache/i)).toBeInTheDocument();
    });

    it("offers cancel only while loading, and only when the page handles it", async () => {
      const onCancel = vi.fn();
      const { rerender } = render(
        <ModelStatus
          {...base}
          status="loading"
          loadProgress={progress()}
          onCancel={onCancel}
        />,
      );
      await userEvent.click(screen.getByTestId("load-cancel"));
      expect(onCancel).toHaveBeenCalledOnce();

      rerender(<ModelStatus {...base} status="ready" backend="wasm" />);
      expect(screen.queryByTestId("load-cancel")).toBeNull();
    });
  });

  describe("ready", () => {
    it("names the resolved backend", () => {
      render(<ModelStatus {...base} status="ready" backend="webgpu" />);
      expect(screen.getByTestId("model-ready")).toHaveTextContent(/webgpu/i);
    });

    it("reports how long the load took", () => {
      render(
        <ModelStatus
          {...base}
          status="ready"
          backend="webgpu"
          loadedInMs={2300}
        />,
      );
      expect(screen.getByTestId("model-ready")).toHaveTextContent(
        /loaded in 2s/i,
      );
    });

    it("stays quiet about a load too short to be worth a number", () => {
      render(
        <ModelStatus {...base} status="ready" backend="wasm" loadedInMs={300} />,
      );
      expect(screen.getByTestId("model-ready")).not.toHaveTextContent(/loaded in/i);
    });
  });

  describe("error — a load failure, recoverable", () => {
    it("explains the cause and keeps the raw message", async () => {
      const onRetry = vi.fn();
      render(
        <ModelStatus
          {...base}
          status="error"
          error="Unauthorized access to file"
          onRetry={onRetry}
        />,
      );
      const alert = screen.getByRole("alert");
      expect(alert).toHaveTextContent(/could not be found on the hugging face hub/i);
      // Friendly, but never at the cost of the original.
      expect(alert).toHaveTextContent("Unauthorized access to file");

      await userEvent.click(screen.getByRole("button", { name: /^retry$/i }));
      expect(onRetry).toHaveBeenCalledOnce();
    });

    it("offers a CPU fallback when the GPU is the likely culprit", async () => {
      const onRetry = vi.fn();
      render(
        <ModelStatus
          {...base}
          status="error"
          backend="webgpu"
          error="WebGPU device lost"
          onRetry={onRetry}
        />,
      );
      await userEvent.click(
        screen.getByRole("button", { name: /retry on cpu/i }),
      );
      expect(onRetry).toHaveBeenCalledWith({ backend: "wasm" });
    });

    it("does not offer a CPU fallback when it already failed on the CPU", () => {
      render(
        <ModelStatus
          {...base}
          status="error"
          backend="wasm"
          error="something went wrong"
          onRetry={() => {}}
        />,
      );
      expect(screen.queryByRole("button", { name: /retry on cpu/i })).toBeNull();
    });

    it("still explains itself with no message from the worker", () => {
      render(<ModelStatus {...base} status="error" />);
      expect(screen.getByRole("alert")).toHaveTextContent(/failed to load/i);
    });
  });
});
