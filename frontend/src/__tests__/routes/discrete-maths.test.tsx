import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { Route } = await import("@/routes/discrete-maths");
const Page = Route?.options?.component as React.ComponentType | undefined;

function renderPage() {
  if (!Page) throw new Error("Discrete maths route component not found");
  render(<Page />);
}

const step = () => fireEvent.click(screen.getByRole("button", { name: /^step$/i }));
const chooseGraph = (name: RegExp) => fireEvent.click(screen.getByRole("button", { name }));
const result = () => within(screen.getByTestId("slot-3"));
/** A node in OUTPUT's diagram — RUN draws the bare graph too. */
const outNode = (v: number) => result().getByTestId(`node-${v}`);

describe("DiscreteMathsPage", () => {
  beforeEach(() => vi.clearAllMocks());

  it("renders three bands, and says why the load band is missing", () => {
    renderPage();
    expect(screen.getByTestId("slot-1")).toBeInTheDocument();
    expect(screen.getByTestId("slot-2")).toBeInTheDocument();
    expect(screen.getByTestId("slot-3")).toBeInTheDocument();
    expect(screen.queryByTestId("slot-4")).toBeNull();
    expect(screen.getByTestId("no-load-band")).toHaveTextContent(/nothing to load/i);
    expect(screen.getByTestId("no-load-band")).toHaveTextContent(/no model, no\s+worker/i);
  });

  it("computes nothing on arrival: OUTPUT is empty until Step", () => {
    renderPage();
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("walk-matrix")).toBeNull();
    expect(screen.getByTestId("step-count")).toHaveTextContent("step 0 of 4"); // C₈
  });

  // The page's equivalent of "choosing an input runs nothing": a graph or a
  // source is a choice, and it resets the search rather than advancing it.
  it("choosing a graph or a source runs nothing and clears the result", () => {
    renderPage();
    step();
    expect(screen.queryByTestId("output-empty")).toBeNull();

    fireEvent.change(screen.getByRole("combobox", { name: /source node/i }), { target: { value: "3" } });
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();

    step();
    chooseGraph(/petersen/i);
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
    expect(screen.getByTestId("graph-facts")).toHaveTextContent("10 nodes · 15 edges · not bipartite");
    expect(screen.getByRole("combobox", { name: /source node/i })).toHaveValue("0");
  });

  it("steps BFS one layer per press and marks the frontier", () => {
    renderPage(); // C₈ from node 0
    step();
    expect(screen.getByTestId("stat-ball")).toHaveTextContent("3 of 8");
    expect(screen.getByTestId("stat-frontier")).toHaveTextContent("2");
    expect(outNode(1)).toHaveAttribute("data-dist", "1");
    expect(outNode(7)).toHaveAttribute("data-dist", "1");
    expect(outNode(2)).toHaveAttribute("data-reached", "false");

    step();
    expect(screen.getByTestId("stat-ball")).toHaveTextContent("5 of 8");
    expect(outNode(2)).toHaveAttribute("data-dist", "2");
  });

  it("runs to the end, says BFS is finished, and stops stepping", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: /run to end/i }));
    expect(screen.getByTestId("step-count")).toHaveTextContent("step 4 of 4");
    expect(screen.getByTestId("bfs-done")).toHaveTextContent(/every node is within 4 hops/i);
    expect(screen.getByRole("button", { name: /^step$/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /run to end/i })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /reset/i }));
    expect(screen.getByTestId("output-empty")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^step$/i })).toBeEnabled();
  });

  // The page's claim, rendered: with A + I the row's support is the BFS ball.
  it("shows row s of (A+I)ᵏ is nonzero exactly on the BFS ball", () => {
    renderPage();
    step();
    step();
    expect(screen.getByTestId("walk-support")).toHaveAttribute("data-count", "5");
    expect(screen.getByTestId("walk-support")).toHaveTextContent("0, 1, 2, 6, 7");
    expect(screen.getByTestId("verdict")).toHaveAttribute("data-match", "true");
    // (A+I)²[0] on C₈: 3 walks back to 0, 2 to each neighbour, 1 to each at distance 2.
    expect(outNode(0)).toHaveAttribute("data-walks", "3");
    expect(outNode(1)).toHaveAttribute("data-walks", "2");
    expect(outNode(2)).toHaveAttribute("data-walks", "1");
  });

  // …and without the self-loop, a bipartite graph's row flickers by parity.
  it("shows that without the self-loop the source cannot hear itself at odd k on C₈", () => {
    renderPage();
    step();
    fireEvent.click(screen.getByRole("button", { name: /a walk must move/i }));

    const verdict = screen.getByTestId("verdict");
    expect(verdict).toHaveAttribute("data-match", "false");
    expect(verdict).toHaveTextContent(/bipartite/i);
    expect(verdict).toHaveTextContent(/node 0 cannot even hear itself/i);
    expect(screen.getByTestId("walk-support")).toHaveTextContent("1, 7");
    expect(outNode(0)).toHaveAttribute("data-walks", "0");
  });

  it("switching the matrix re-reads the same k, and does not step", () => {
    renderPage();
    step();
    fireEvent.click(screen.getByRole("button", { name: /a walk must move/i }));
    expect(screen.getByTestId("step-count")).toHaveTextContent("step 1 of 4");
    fireEvent.click(screen.getByRole("button", { name: /a walk may pause/i }));
    expect(screen.getByTestId("step-count")).toHaveTextContent("step 1 of 4");
    expect(screen.getByTestId("verdict")).toHaveAttribute("data-match", "true");
  });

  it("on Petersen, reaches every node in two steps", () => {
    renderPage();
    chooseGraph(/petersen/i);
    step();
    expect(screen.getByTestId("stat-ball")).toHaveTextContent("4 of 10");
    step();
    expect(screen.getByTestId("stat-ball")).toHaveTextContent("10 of 10");
    expect(screen.getByTestId("bfs-done")).toBeInTheDocument();
  });

  it("credits the karate club's data source", () => {
    renderPage();
    chooseGraph(/karate/i);
    expect(screen.getByTestId("slot-1")).toHaveTextContent(/Zachary \(1977\)/);
    expect(screen.getByTestId("graph-facts")).toHaveTextContent("34 nodes · 78 edges");
  });

  it("names every drawing for a screen reader", () => {
    renderPage();
    step();
    expect(result().getByRole("img", { name: /after 1 BFS step from node 0: 3 of 8 nodes reached/i })).toBeInTheDocument();
    expect(result().getByRole("img", { name: /\(A \+ I\)\^1: a 8 by 8 matrix/i })).toBeInTheDocument();
  });

  it("constructs no worker and makes no request at any point", () => {
    const spy = vi.fn();
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const original = (globalThis as { Worker?: unknown }).Worker;
    (globalThis as { Worker?: unknown }).Worker = spy;
    try {
      renderPage();
      chooseGraph(/karate/i);
      fireEvent.click(screen.getByRole("button", { name: /run to end/i }));
      fireEvent.click(screen.getByRole("button", { name: /a walk must move/i }));
      expect(spy).not.toHaveBeenCalled();
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      (globalThis as { Worker?: unknown }).Worker = original;
      fetchSpy.mockRestore();
    }
  });
});
