import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ScoreList } from "./ScoreList";

describe("ScoreList", () => {
  it("renders one row per label, sorted highest first", () => {
    render(
      <ScoreList
        scores={[
          { label: "neutral", score: 0.12 },
          { label: "positive", score: 0.71 },
          { label: "negative", score: 0.17 },
        ]}
      />,
    );
    const rows = screen.getAllByRole("listitem");
    expect(rows).toHaveLength(3);
    // Sorted here rather than trusted from the caller: a page that merges two
    // models' lists does not hand them over ranked.
    expect(rows[0]).toHaveTextContent("positive");
    expect(rows[2]).toHaveTextContent("neutral");
  });

  it("prints every score, so a near-tie is readable as numbers", () => {
    render(
      <ScoreList
        scores={[
          { label: "POSITIVE", score: 0.51 },
          { label: "NEGATIVE", score: 0.49 },
        ]}
      />,
    );
    expect(screen.getByText("0.51")).toBeInTheDocument();
    expect(screen.getByText("0.49")).toBeInTheDocument();
  });

  it("says so in words when the top two are within a hair of each other", () => {
    render(
      <ScoreList
        scores={[
          { label: "POSITIVE", score: 0.51 },
          { label: "NEGATIVE", score: 0.49 },
        ]}
      />,
    );
    expect(screen.getByTestId("score-near-tie")).toHaveTextContent(
      /not confident/i,
    );
  });

  it("stays quiet when the model actually is confident", () => {
    render(
      <ScoreList
        scores={[
          { label: "POSITIVE", score: 0.98 },
          { label: "NEGATIVE", score: 0.02 },
        ]}
      />,
    );
    expect(screen.queryByTestId("score-near-tie")).not.toBeInTheDocument();
  });

  it("calls out a single-label result rather than dressing it up", () => {
    render(<ScoreList scores={[{ label: "POSITIVE", score: 0.99 }]} />);
    expect(screen.getByTestId("score-single")).toBeInTheDocument();
  });

  it("renders nothing at all for an empty list", () => {
    const { container } = render(<ScoreList scores={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("can be silenced where the page makes the point itself", () => {
    render(
      <ScoreList
        quiet
        scores={[
          { label: "a", score: 0.51 },
          { label: "b", score: 0.49 },
        ]}
      />,
    );
    expect(screen.queryByTestId("score-near-tie")).not.toBeInTheDocument();
  });
});
