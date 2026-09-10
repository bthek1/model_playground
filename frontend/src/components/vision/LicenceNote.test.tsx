import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { ModelLicence } from "@/vision/backgroundRemoval";
import { LicenceNote } from "./LicenceNote";

const PERMISSIVE: ModelLicence = {
  name: "Apache-2.0",
  commercial: true,
  url: "https://huggingface.co/Xenova/modnet",
};

const RESTRICTED: ModelLicence = {
  name: "bria-rmbg-1.4",
  commercial: false,
  url: "https://bria.ai/bria-huggingface-model-license-agreement/",
  note: "Released under a Creative Commons licence for non-commercial use. Commercial use needs a separate paid agreement with BRIA.",
};

describe("LicenceNote", () => {
  it("states a permissive licence quietly, and links to it", () => {
    render(<LicenceNote licence={PERMISSIVE} />);
    const note = screen.getByTestId("model-licence");
    expect(note).toHaveTextContent(/Apache-2\.0/);
    expect(note).toHaveTextContent(/commercial use permitted/i);
    expect(screen.getByRole("link", { name: "Apache-2.0" })).toHaveAttribute(
      "href",
      PERMISSIVE.url,
    );
  });

  it("spells out a restriction rather than only naming the licence", () => {
    // "bria-rmbg-1.4" tells a reader nothing. The sentence is the point: a
    // restriction the UI cannot explain is one the user cannot act on.
    render(<LicenceNote licence={RESTRICTED} />);
    const note = screen.getByTestId("model-licence");
    expect(note).toHaveTextContent(/non-commercial/i);
    expect(note).toHaveTextContent(/BRIA/);
    expect(
      screen.getByRole("link", { name: /read the licence/i }),
    ).toHaveAttribute("href", RESTRICTED.url);
  });

  it("falls back to a plain statement when an entry has no note", () => {
    render(<LicenceNote licence={{ ...RESTRICTED, note: undefined }} />);
    expect(screen.getByTestId("model-licence")).toHaveTextContent(
      /not licensed for commercial use/i,
    );
  });

  it("marks the restricted case as a warning, not as body text", () => {
    // Same amber as the size-before-load guardrail, and for the same reason: it
    // is a cost the user is about to take on without being told.
    const { rerender } = render(<LicenceNote licence={PERMISSIVE} />);
    expect(screen.getByTestId("model-licence").className).toContain(
      "text-muted-foreground",
    );

    rerender(<LicenceNote licence={RESTRICTED} />);
    expect(screen.getByTestId("model-licence").className).toContain("amber");
  });

  it("opens the licence safely in a new tab", () => {
    render(<LicenceNote licence={RESTRICTED} />);
    const link = screen.getByRole("link", { name: /read the licence/i });
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", expect.stringContaining("noreferrer"));
  });
});
