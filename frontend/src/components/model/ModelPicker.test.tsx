import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { LARGE_MODEL_BYTES } from "@/audio/size";

import { ModelPicker, type PickableModel } from "./ModelPicker";

// Params chosen either side of LARGE_MODEL_BYTES (200 MB) at fp16 (2 bytes per
// param), so the guardrail's threshold is exercised rather than assumed.
const SMALL_PARAMS = 20; // ≈40 MB
const LARGE_PARAMS = Math.ceil((LARGE_MODEL_BYTES / 1e6 / 2) * 1.5);

const MODELS: PickableModel[] = [
  { id: "small", label: "Tiny model", hint: "Fast and rough.", params: SMALL_PARAMS },
  { id: "large", label: "Big model", hint: "Slow and good.", params: LARGE_PARAMS },
];

describe("ModelPicker", () => {
  it("offers every model and marks the selected one", () => {
    render(<ModelPicker models={MODELS} value="small" onChange={() => {}} />);
    expect(screen.getByRole("button", { name: "Tiny model" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Big model" })).toBeInTheDocument();
  });

  it("hands the caller the whole model, not just its id", async () => {
    // The TTS route needs `voices` off the selection to reset its voice picker,
    // so passing the id back alone would push a lookup into every caller.
    const onChange = vi.fn();
    render(<ModelPicker models={MODELS} value="small" onChange={onChange} />);

    await userEvent.click(screen.getByRole("button", { name: "Big model" }));
    expect(onChange).toHaveBeenCalledWith(MODELS[1]);
  });

  it("quotes the selected model's download for both backends", () => {
    render(<ModelPicker models={MODELS} value="small" onChange={() => {}} />);
    const note = screen.getByTestId("model-size-note");
    expect(note).toHaveTextContent("Fast and rough.");
    expect(note).toHaveTextContent(`${SMALL_PARAMS}M params`);
    // Both, because the backend isn't resolved until the worker loads.
    expect(note).toHaveTextContent(/on WebGPU · .* on WASM/);
  });

  it("warns past the large-model threshold and stays quiet below it", () => {
    const { rerender } = render(
      <ModelPicker models={MODELS} value="small" onChange={() => {}} />,
    );
    // A warning on everything is a warning on nothing.
    expect(screen.queryByTestId("model-size-warning")).toBeNull();

    rerender(<ModelPicker models={MODELS} value="large" onChange={() => {}} />);
    expect(screen.getByTestId("model-size-warning")).toHaveTextContent(
      /large model/i,
    );
  });

  it("prefers measured bytes where the params estimate would mislead", () => {
    // ASR on WASM keeps an fp32 decoder, so a uniform-q8 estimate is ~3x under.
    const measured: PickableModel[] = [
      {
        id: "asr",
        label: "Whisper",
        hint: "Timestamps.",
        params: 74,
        bytes: { webgpu: 146_276_352, wasm: 231_735_296 },
      },
    ];
    render(<ModelPicker models={measured} value="asr" onChange={() => {}} />);
    // 221 MB measured on WASM crosses the threshold; 74M × 1 byte would not.
    expect(screen.getByTestId("model-size-warning")).toBeInTheDocument();
  });

  it("disables every button while a load or run is in flight", async () => {
    // Switching models mid-flight would discard work the user is waiting on.
    const onChange = vi.fn();
    render(
      <ModelPicker models={MODELS} value="small" onChange={onChange} disabled />,
    );
    for (const label of ["Tiny model", "Big model"]) {
      expect(screen.getByRole("button", { name: label })).toBeDisabled();
    }
    await userEvent.click(screen.getByRole("button", { name: "Big model" }));
    expect(onChange).not.toHaveBeenCalled();
  });

  it("renders without a size note when the value matches no model", () => {
    render(<ModelPicker models={MODELS} value="gone" onChange={() => {}} />);
    expect(screen.queryByTestId("model-size-note")).toBeNull();
    expect(screen.getByRole("button", { name: "Tiny model" })).toBeInTheDocument();
  });
});
