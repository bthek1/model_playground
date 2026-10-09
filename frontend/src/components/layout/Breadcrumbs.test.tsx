import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const path = vi.hoisted(() => ({ current: "/asr" }));

vi.mock("@tanstack/react-router", async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useRouterState: () => path.current,
  Link: ({ children, to, hash }: { children: React.ReactNode; to: string; hash?: string }) => (
    <a href={hash ? `${to}#${hash}` : to}>{children}</a>
  ),
}));

import { Breadcrumbs } from "./Breadcrumbs";

describe("Breadcrumbs", () => {
  beforeEach(() => {
    path.current = "/asr";
  });

  it("shows the task's category, linked to its section of home, then the task", () => {
    render(<Breadcrumbs />);
    const nav = screen.getByRole("navigation", { name: "Breadcrumb" });
    expect(nav).toHaveTextContent("AudioAutomatic Speech Recognition");
    expect(screen.getByRole("link", { name: "Audio" })).toHaveAttribute("href", "/home#audio");
    expect(screen.getByText("Automatic Speech Recognition")).toHaveAttribute("aria-current", "page");
  });

  it.each(["/home", "/privacy", "/login", "/nope"])("renders nothing on %s", (p) => {
    path.current = p;
    const { container } = render(<Breadcrumbs />);
    expect(container).toBeEmptyDOMElement();
  });
});
