import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { LARGE_MODEL_BYTES } from "@/model/size";

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
    expect(screen.getByRole("button", { name: /tiny model/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /big model/i })).toBeInTheDocument();
  });

  it("hands the caller the whole model, not just its id", async () => {
    // The TTS route needs `voices` off the selection to reset its voice picker,
    // so passing the id back alone would push a lookup into every caller.
    const onChange = vi.fn();
    render(<ModelPicker models={MODELS} value="small" onChange={onChange} />);

    await userEvent.click(screen.getByRole("button", { name: /big model/i }));
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
      expect(screen.getByRole("button", { name: new RegExp(label, "i") })).toBeDisabled();
    }
    await userEvent.click(screen.getByRole("button", { name: /big model/i }));
    expect(onChange).not.toHaveBeenCalled();
  });

  it("renders without a size note when the value matches no model", () => {
    render(<ModelPicker models={MODELS} value="gone" onChange={() => {}} />);
    expect(screen.queryByTestId("model-size-note")).toBeNull();
    expect(screen.getByRole("button", { name: /tiny model/i })).toBeInTheDocument();
  });
  it("shows each model's hint on its own row in the rail's list layout", () => {
    // At ~20rem the old chip row hid the hint behind a `title` tooltip. The
    // rail has vertical room instead, so every row carries label + hint.
    render(<ModelPicker models={MODELS} value="small" onChange={() => {}} />);
    const row = screen.getByRole("button", { name: /tiny model/i });
    expect(row).toHaveTextContent("Fast and rough.");
    expect(row).toHaveAttribute("aria-pressed", "true");
    expect(
      screen.getByRole("button", { name: /big model/i }),
    ).toHaveAttribute("aria-pressed", "false");
  });

  it("still offers the compact chip row where a route has the width", () => {
    render(
      <ModelPicker models={MODELS} value="small" onChange={() => {}} layout="row" />,
    );
    // Exact names: the chip carries the label alone, hint goes to the tooltip.
    expect(screen.getByRole("button", { name: "Tiny model" })).toBeInTheDocument();
    const big = screen.getByRole("button", { name: "Big model" });
    expect(big).not.toHaveTextContent("Slow and good.");
    expect(big.getAttribute("title")).toMatch(/^Slow and good\. — /);
  });

  it("keeps the size guardrail identical in both layouts", () => {
    const { rerender } = render(
      <ModelPicker models={MODELS} value="large" onChange={() => {}} />,
    );
    const listNote = screen.getByTestId("model-size-note").textContent;
    expect(screen.getByTestId("model-size-warning")).toBeInTheDocument();

    rerender(
      <ModelPicker models={MODELS} value="large" onChange={() => {}} layout="row" />,
    );
    expect(screen.getByTestId("model-size-note")).toHaveTextContent(
      listNote!.replace(/\s+/g, " ").trim(),
    );
    expect(screen.getByTestId("model-size-warning")).toBeInTheDocument();
  });

  it("disables the list rows too, not just the chips", async () => {
    const onChange = vi.fn();
    render(
      <ModelPicker models={MODELS} value="small" onChange={onChange} disabled />,
    );
    await userEvent.click(screen.getByRole("button", { name: /big model/i }));
    expect(onChange).not.toHaveBeenCalled();
  });
  describe("cached weights", () => {
    it("badges the models already downloaded, and only those", () => {
      render(
        <ModelPicker
          models={MODELS}
          value="small"
          onChange={() => {}}
          cached={new Set(["large"])}
        />,
      );
      expect(screen.getByTestId("model-cached-large")).toBeInTheDocument();
      expect(screen.queryByTestId("model-cached-small")).toBeNull();
    });

    it("says the selected model costs nothing to load again", () => {
      render(
        <ModelPicker
          models={MODELS}
          value="large"
          onChange={() => {}}
          cached={new Set(["large"])}
        />,
      );
      expect(screen.getByTestId("model-size-note")).toHaveTextContent(
        /already downloaded/i,
      );
      // The large-model warning is about a download that is no longer going to
      // happen — repeating it here would just be noise.
      expect(screen.queryByTestId("model-size-warning")).toBeNull();
    });

    it("offers eviction only for a cached model, and only when handled", async () => {
      const onEvict = vi.fn();
      const { rerender } = render(
        <ModelPicker
          models={MODELS}
          value="large"
          onChange={() => {}}
          cached={new Set(["large"])}
          onEvict={onEvict}
        />,
      );
      await userEvent.click(screen.getByTestId("model-evict"));
      expect(onEvict).toHaveBeenCalledWith(MODELS[1]);

      rerender(
        <ModelPicker
          models={MODELS}
          value="small"
          onChange={() => {}}
          cached={new Set(["large"])}
          onEvict={onEvict}
        />,
      );
      expect(screen.queryByTestId("model-evict")).toBeNull();
    });
  });

  describe("backends", () => {
    // `PickableModel.backends` was declarable from Wave 0 and read by nothing
    // until /image-to-text needed it. Declaring a constraint that nothing
    // enforces is worse than not declaring it: the limitation surfaced as a
    // failed 275 MB download instead of a greyed-out row.
    const GATED: PickableModel[] = [
      { ...MODELS[0], backends: ["webgpu"] },
      MODELS[1],
    ];

    it("gates nothing while the probe is still undecided", () => {
      // `null` means "ask again in 10 ms", not "no GPU". Treating it as WASM
      // would grey out every WebGPU model for a frame on each page load.
      render(
        <ModelPicker models={GATED} value="large" onChange={() => {}} backend={null} />,
      );
      expect(screen.getByRole("button", { name: /tiny model/i })).toBeEnabled();
      expect(screen.queryByTestId("model-unsupported-small")).toBeNull();
    });

    it("gates nothing when no probe result is passed at all", () => {
      // Every route that predates the probe keeps its old behaviour.
      render(<ModelPicker models={GATED} value="large" onChange={() => {}} />);
      expect(screen.getByRole("button", { name: /tiny model/i })).toBeEnabled();
    });

    it("disables a model this machine cannot run, and says why", () => {
      render(
        <ModelPicker
          models={GATED}
          value="large"
          onChange={() => {}}
          backend="wasm"
        />,
      );
      expect(screen.getByRole("button", { name: /tiny model/i })).toBeDisabled();
      const note = screen.getByTestId("model-unsupported-small");
      expect(note).toHaveTextContent(/needs webgpu/i);
      expect(note).toHaveTextContent(/wasm/i);
    });

    it("leaves the models that do run selectable", () => {
      // A route whose every model is gated has no model at all.
      render(
        <ModelPicker
          models={GATED}
          value="large"
          onChange={() => {}}
          backend="wasm"
        />,
      );
      expect(screen.getByRole("button", { name: /big model/i })).toBeEnabled();
    });

    it("offers a gated model normally once the backend supports it", () => {
      render(
        <ModelPicker
          models={GATED}
          value="large"
          onChange={() => {}}
          backend="webgpu"
        />,
      );
      expect(screen.getByRole("button", { name: /tiny model/i })).toBeEnabled();
      expect(screen.queryByTestId("model-unsupported-small")).toBeNull();
    });

    it("gates the compact chip row too, not just the rail's list", async () => {
      const onChange = vi.fn();
      render(
        <ModelPicker
          models={GATED}
          value="large"
          onChange={onChange}
          layout="row"
          backend="wasm"
        />,
      );
      const chip = screen.getByRole("button", { name: /tiny model/i });
      expect(chip).toBeDisabled();
      await userEvent.click(chip);
      expect(onChange).not.toHaveBeenCalled();
    });
  });
});
