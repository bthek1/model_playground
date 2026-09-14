import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { APP_NAME, Logo, LogoMark } from "./Logo";

describe("Logo", () => {
  it("renders the product name as real text", () => {
    render(<Logo />);
    expect(screen.getByText(APP_NAME)).toBeInTheDocument();
  });

  it("hides the mark from assistive tech — the name beside it is the label", () => {
    const { container } = render(<Logo />);
    const svg = container.querySelector("svg");
    expect(svg).toHaveAttribute("aria-hidden", "true");
  });

  it("renders the mark alone with no text", () => {
    const { container } = render(<LogoMark />);
    expect(container.querySelector("svg")).toBeInTheDocument();
    expect(container.textContent).toBe("");
  });

  it("takes a size from className", () => {
    const { container } = render(<LogoMark className="h-16 w-16" />);
    expect(container.querySelector("svg")).toHaveClass("h-16", "w-16");
  });
});
