import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UseLiveAsrResult } from "@/hooks/useLiveAsr";

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

const mockStart = vi.fn();
const mockStop = vi.fn();
const mockTranscribeClip = vi.fn();
const baseState: UseLiveAsrResult = {
  status: "loading",
  loading: true,
  ready: false,
  progress: null,
  backend: null,
  recording: false,
  text: "",
  error: null,
  start: mockStart,
  stop: mockStop,
  transcribeClip: mockTranscribeClip,
};
let mockState: UseLiveAsrResult = { ...baseState };

vi.mock("@/hooks/useLiveAsr", () => ({
  useLiveAsr: () => mockState,
}));

const { Route } = await import("@/routes/asr");
const AsrPage = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!AsrPage) throw new Error("ASR route component not found");
  render(<AsrPage />);
}

describe("AsrPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState = { ...baseState };
  });

  it("renders the heading and both model options", () => {
    renderPage();
    expect(
      screen.getByRole("heading", { name: /automatic speech recognition/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /whisper base/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /moonshine tiny/i })).toBeInTheDocument();
  });

  it("disables the listen/upload controls until the model is ready", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /start listening/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /upload audio/i })).toBeDisabled();
    expect(screen.getByText(/loading model/i)).toBeInTheDocument();
  });

  it("enables controls and shows the backend once ready", () => {
    mockState = {
      ...baseState,
      status: "ready",
      loading: false,
      ready: true,
      backend: "webgpu",
    };
    renderPage();
    expect(screen.getByRole("button", { name: /start listening/i })).toBeEnabled();
    expect(screen.getByText(/running on/i)).toBeInTheDocument();
    // The intro copy also mentions "WebGPU"; the status badge holds the exact
    // lowercase backend value.
    expect(screen.getByText("webgpu")).toBeInTheDocument();
  });

  it("shows a Stop button and a live indicator while recording", () => {
    mockState = {
      ...baseState,
      status: "ready",
      loading: false,
      ready: true,
      backend: "wasm",
      recording: true,
      text: "hello",
    };
    renderPage();
    expect(screen.getByRole("button", { name: /stop/i })).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /start listening/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/listening…/i)).toBeInTheDocument();
    expect(screen.getByText("hello")).toBeInTheDocument();
  });

  it("renders the final transcript text when present and idle", () => {
    mockState = {
      ...baseState,
      status: "ready",
      loading: false,
      ready: true,
      backend: "wasm",
      text: "the quick brown fox",
    };
    renderPage();
    expect(screen.getByText("the quick brown fox")).toBeInTheDocument();
    expect(screen.getByText(/final transcript/i)).toBeInTheDocument();
  });

  it("surfaces a load error from the hook", () => {
    mockState = { ...baseState, status: "error", loading: false, error: "download failed" };
    renderPage();
    expect(screen.getByText(/download failed/i)).toBeInTheDocument();
  });
});
