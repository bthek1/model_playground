import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { UseDocVqaResult } from "@/hooks/useDocVqa";

const fakeImage = { width: 8, height: 8, channels: 3, data: [] } as never;
const fromFile = vi.fn().mockResolvedValue(fakeImage);
const fromUrl = vi.fn().mockResolvedValue(fakeImage);
const downscale = vi.fn(async (img: unknown) => img);
vi.mock("@/vision/image", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    fromFile: (...a: unknown[]) => fromFile(...a),
    fromUrl: (...a: unknown[]) => fromUrl(...a),
    downscale: (...a: unknown[]) => downscale(...(a as [never])),
  };
});

const pickBackend = vi.fn(async () => "wasm");
vi.mock("@/model/backend", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, pickBackend: () => pickBackend() };
});

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    createFileRoute: vi
      .fn()
      .mockImplementation(
        (path: string) => (opts: Record<string, unknown>) => ({
          path,
          options: opts,
        }),
      ),
  };
});

const mockRun = vi.fn();
const mockLoad = vi.fn();

const baseState = (): UseDocVqaResult => ({
  status: "idle",
  idle: true,
  loading: false,
  ready: false,
  progress: null,
  loadProgress: null,
  loadedInMs: null,
  backend: null,
  running: false,
  error: null,
  result: null,
  run: mockRun,
  load: mockLoad,
  retry: vi.fn(),
  cancel: vi.fn(),
});
let mockState: UseDocVqaResult = baseState();
const useDocVqa = vi.fn(() => mockState);

vi.mock("@/hooks/useDocVqa", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return { ...actual, useDocVqa: (...a: unknown[]) => useDocVqa(...(a as [])) };
});

const { Route } = await import("@/routes/document-question-answering");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Document-QA route component not found");
  return render(<Page />);
}

const RUN_BUTTON = /^generate$/i;
const SAMPLE = /^invoice$/i;

const ready = (extra: Partial<UseDocVqaResult> = {}): UseDocVqaResult => ({
  ...baseState(),
  status: "ready",
  idle: false,
  ready: true,
  ...extra,
});

async function pick() {
  fireEvent.click(screen.getByRole("button", { name: SAMPLE }));
  await waitFor(() =>
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled(),
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockState = baseState();
  mockRun.mockResolvedValue({
    answer: "us-001",
    question: "What is the invoice number?",
    ms: 2400,
  });
  pickBackend.mockResolvedValue("wasm");
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => vi.clearAllMocks());

describe("DocumentQuestionAnsweringPage — the four-slot contract", () => {
  it("renders the heading and the catalogue", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /document question answering/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /donut base/i })).toBeInTheDocument();
  });

  it("renders all four slots with an empty OUTPUT before any run", () => {
    renderPage();
    expect(screen.getAllByRole("region")).toHaveLength(4);
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
  });

  it("downloads nothing on arrival, and loads only on request", () => {
    renderPage();
    // No `autoLoad` argument: the hook's own default is `idle`.
    expect(useDocVqa).toHaveBeenCalledWith("Xenova/donut-base-finetuned-docvqa");
    expect(mockLoad).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /load model/i }));
    expect(mockLoad).toHaveBeenCalledOnce();
  });

  it("keeps GENERATE disabled until ready, while the input sources stay live", async () => {
    renderPage();
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeDisabled();
    expect(screen.getByRole("button", { name: SAMPLE })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: SAMPLE }));
    await waitFor(() => expect(fromUrl).toHaveBeenCalled());
    expect(mockRun).not.toHaveBeenCalled();
  });

  it("offers both backends — neither was measured, so neither is gated", async () => {
    // Unlike the chat decoders in multimodal/types.ts, this is an encoder plus a
    // short extractive decode: the CPU path is real.
    pickBackend.mockResolvedValue("wasm");
    renderPage();
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /donut base/i })).toBeEnabled(),
    );
    expect(
      screen.queryByTestId(
        "model-unsupported-Xenova/donut-base-finetuned-docvqa",
      ),
    ).not.toBeInTheDocument();
  });
});

describe("DocumentQuestionAnsweringPage — nothing runs until asked", () => {
  it("picks a sample without running, then generates when asked", async () => {
    mockState = ready();
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: SAMPLE }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeEnabled(),
    );
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
  });

  it("bounds the source at the processor's own 2560, never below it", async () => {
    // The cap is a memory bound, not preprocessing: `thumbnail()` never
    // upscales, so a smaller source is padded — fewer real pixels of print at
    // identical compute. Capping at 512 like the sibling VLM route would trade
    // legibility for nothing.
    mockState = ready();
    renderPage();
    await pick();
    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(downscale).toHaveBeenCalledWith(fakeImage, 2560));
  });

  it("editing the question runs nothing", async () => {
    mockState = ready();
    renderPage();
    await pick();

    fireEvent.change(screen.getByTestId("question-input"), {
      target: { value: "What is the total?" },
    });
    expect(mockRun).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() =>
      expect(mockRun).toHaveBeenCalledWith(fakeImage, "What is the total?"),
    );
  });

  it("a preset fills the box and runs nothing", async () => {
    mockState = ready();
    renderPage();
    await pick();

    fireEvent.click(
      within(screen.getByTestId("presets")).getByRole("button", {
        name: /what is the date/i,
      }),
    );
    expect(mockRun).not.toHaveBeenCalled();
    expect(screen.getByTestId("question-input")).toHaveValue(
      "What is the date?",
    );
  });

  it("survives its run: two questions cost one decode", async () => {
    mockState = ready();
    renderPage();
    await pick();

    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: RUN_BUTTON }));
    await waitFor(() => expect(mockRun).toHaveBeenCalledTimes(2));

    expect(fromUrl).toHaveBeenCalledTimes(1);
  });

  it("will not run on an empty question", async () => {
    mockState = ready();
    renderPage();
    await pick();
    fireEvent.change(screen.getByTestId("question-input"), {
      target: { value: "  " },
    });
    expect(screen.getByRole("button", { name: RUN_BUTTON })).toBeDisabled();
  });
});

describe("DocumentQuestionAnsweringPage — what the page claims", () => {
  it("states the privacy argument in the page, not a tooltip", () => {
    renderPage();
    expect(screen.getByTestId("privacy-note")).toHaveTextContent(
      /never leaves your device/i,
    );
  });

  it("explains why it does not shrink the image, unlike every sibling route", () => {
    renderPage();
    expect(screen.getByTestId("resolution-note")).toHaveTextContent(
      /pads rather than enlarges/i,
    );
  });

  it("warns that the CPU path is slow, before anything is downloaded", async () => {
    // Measured: one question on the fp32 WASM decoder did not finish inside six
    // minutes, while the same question answers in ~8s where the decoder can be
    // quantized. Keyed off the probe rather than the resolved backend so it
    // lands before the 597 MB, not after.
    pickBackend.mockResolvedValue("wasm");
    renderPage();
    await waitFor(() =>
      expect(screen.getByTestId("cpu-speed-note")).toHaveTextContent(
        /minutes.*per question/i,
      ),
    );
  });

  it("shows no CPU warning on a machine with a GPU", async () => {
    pickBackend.mockResolvedValue("webgpu");
    renderPage();
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /donut base/i })).toBeEnabled(),
    );
    expect(screen.queryByTestId("cpu-speed-note")).not.toBeInTheDocument();
  });

  it("says the answer is extracted, not reasoned", () => {
    mockState = ready({
      result: { answer: "us-001", question: "What is the invoice number?", ms: 2400 },
    });
    renderPage();
    expect(screen.getByTestId("extractive-note")).toHaveTextContent(
      /extraction, not reasoning/i,
    );
  });
});

describe("DocumentQuestionAnsweringPage — the answer", () => {
  it("renders the extracted span with the question it was asked", () => {
    mockState = ready({
      result: { answer: "us-001", question: "What is the invoice number?", ms: 2400 },
    });
    renderPage();
    expect(screen.getByTestId("answer-text")).toHaveTextContent("us-001");
    expect(screen.getByTestId("answer-asked")).toHaveTextContent(
      /What is the invoice number\?/,
    );
    expect(screen.getByTestId("generate-ms")).toHaveTextContent(/2\.4s/);
  });

  it("explains a null answer instead of showing a blank panel", () => {
    // The pipeline returns `{ answer: null }` when its `<s_answer>` match
    // misses. Nothing throws, so this is an ordinary outcome the page owes an
    // explanation for — on exactly the documents the model found hardest.
    mockState = ready({
      result: { answer: null, question: "What is the total?", ms: 1900 },
    });
    renderPage();
    expect(screen.getByTestId("answer-none")).toHaveTextContent(
      /did not find an answer/i,
    );
    expect(screen.queryByTestId("answer-text")).not.toBeInTheDocument();
  });
});

describe("DocumentQuestionAnsweringPage — errors", () => {
  it("puts a load failure in the LOAD slot, not in OUTPUT", () => {
    mockState = { ...baseState(), status: "error", idle: false, error: "404 not found" };
    renderPage();
    const note = screen.getByText(/404 not found/i);
    expect(screen.getByTestId("slot-2")).toContainElement(note);
    expect(screen.getByTestId("slot-4")).not.toContainElement(note);
  });

  it("puts an inference failure in OUTPUT, where the model stays loaded", () => {
    mockState = ready({ error: "Non-zero status code" });
    renderPage();
    expect(screen.getByTestId("slot-4")).toContainElement(
      screen.getByText(/non-zero status code/i),
    );
  });

  it("surfaces a failed decode in the RUN slot, not the output", async () => {
    mockState = ready();
    fromUrl.mockRejectedValueOnce(new Error("Unsupported image type"));
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: SAMPLE }));
    const note = await screen.findByText(/unsupported image type/i);
    expect(screen.getByTestId("slot-3")).toContainElement(note);
    expect(mockRun).not.toHaveBeenCalled();
  });
});
