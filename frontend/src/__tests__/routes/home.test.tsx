import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

// The privacy link is a router `Link`, and this test renders without a router.
vi.mock("@tanstack/react-router", async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  Link: ({ children, to }: { children: React.ReactNode; to: string }) => <a href={to}>{children}</a>,
}));

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
const { taskCategories } = await import("@/components/layout/taskTaxonomy");
const { HOME_LEAD, HOME_PRIVACY } = await import("@/seo/pages");
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

  // The sidebar starts collapsed, so this is the page that links every task in
  // body text — for a first visit and for a crawler (#65).
  it("links every task in the taxonomy", () => {
    renderPage();
    for (const t of taskCategories.flatMap((c) => c.tasks)) {
      expect(screen.getByRole("link", { name: t.label })).toHaveAttribute("href", t.to);
    }
  });

  it("says what the site runs, in words a visitor would search for", () => {
    renderPage();
    expect(screen.getByText(/machine-learning models in your browser/i)).toBeInTheDocument();
  });

  // The static HTML paints these words as one paragraph at first paint; React
  // must not paint a larger one later, or LCP moves to when the entry chunk
  // has run (#67). Two paragraphs, each a part of the static one.
  it("splits the intro into two paragraphs, each smaller than the static one", () => {
    renderPage();
    const lead = screen.getByText(HOME_LEAD);
    expect(lead.tagName).toBe("P");
    expect(lead.textContent).toBe(HOME_LEAD);
    const privacy = screen.getByText((_, el) => el?.tagName === "P" && el.textContent!.startsWith(HOME_PRIVACY));
    expect(privacy).not.toBe(lead);
  });

  it("links its privacy claim to the privacy notice (#62)", () => {
    renderPage();
    expect(screen.getByRole("link", { name: "privacy" })).toHaveAttribute("href", "/privacy");
  });
});
