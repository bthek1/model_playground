import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ErrorNote } from "./ErrorNote";

// The shared error rendering. Its contract matters more than it looks: the page
// pattern requires an error to appear in the slot that produced it, so this must
// be safe to drop anywhere — including in a slot that usually has no error.
describe("ErrorNote", () => {
  it("renders nothing for a null or empty message", () => {
    const { container, rerender } = render(<ErrorNote message={null} />);
    expect(container).toBeEmptyDOMElement();

    rerender(<ErrorNote message={undefined} />);
    expect(container).toBeEmptyDOMElement();

    // An empty string is still "no error" — a blank red box would be worse than
    // nothing, and callers pass `error ?? ""` more often than they should.
    rerender(<ErrorNote message="" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("announces a real message to assistive tech", () => {
    render(<ErrorNote message="Unauthorized access to file" />);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Unauthorized access to file");
    expect(alert).toHaveAttribute("data-testid", "error-note");
  });

  it("renders a recovery action beside the message", () => {
    render(
      <ErrorNote
        message="download failed"
        action={<button type="button">Retry</button>}
      />,
    );
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("omits the action slot entirely when there is no recovery", () => {
    render(<ErrorNote message="download failed" />);
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("accepts a className so a caller can place it in a tight slot", () => {
    render(<ErrorNote message="boom" className="mt-4" />);
    expect(screen.getByRole("alert")).toHaveClass("mt-4");
  });
});
