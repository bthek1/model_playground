import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    Link: ({
      children,
      to,
      ...props
    }: {
      children: React.ReactNode;
      to: string;
      [key: string]: unknown;
    }) => (
      <a href={to as string} {...props}>
        {children}
      </a>
    ),
  };
});

import { HeroBanner } from "./HeroBanner";

describe("HeroBanner", () => {
  it("renders the brand name in the heading", () => {
    render(<HeroBanner />);
    expect(screen.getByText("Model Playground")).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /Welcome to Model Playground/i }),
    ).toBeInTheDocument();
  });

  it("links to the sign in and sign up pages", () => {
    render(<HeroBanner />);
    expect(screen.getByRole("link", { name: "Sign in" })).toHaveAttribute(
      "href",
      "/login",
    );
    expect(
      screen.getByRole("link", { name: "Create account" }),
    ).toHaveAttribute("href", "/signup");
  });
});
