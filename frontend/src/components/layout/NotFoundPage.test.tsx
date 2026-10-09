import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-router", async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useRouterState: () => "/no-such-page",
  Link: ({ children, to }: { children: React.ReactNode; to: string }) => <a href={to}>{children}</a>,
}));

import { Route } from "@/routes/__root";
import { NotFoundPage } from "./NotFoundPage";

const robots = () => document.head.querySelector('meta[name="robots"]');

describe("NotFoundPage", () => {
  it("is the root route's not-found page", () => {
    expect(Route.options.notFoundComponent).toBe(NotFoundPage);
  });

  it("names the missing path and links back into the site", () => {
    render(<NotFoundPage />);
    expect(screen.getByRole("heading", { name: "Page not found" })).toBeInTheDocument();
    expect(screen.getByText("/no-such-page")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Browse every task" })).toHaveAttribute("href", "/home");
  });

  // The edge answers an unknown path with a 200 — it cannot know the routes —
  // so the page itself has to keep it out of the index (#63).
  it("asks not to be indexed while mounted, and only then", () => {
    const { unmount } = render(<NotFoundPage />);
    expect(robots()).toHaveAttribute("content", "noindex");
    unmount();
    expect(robots()).toBeNull();
  });
});
