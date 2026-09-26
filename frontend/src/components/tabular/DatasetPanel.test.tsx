import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { parseCsv } from "@/tabular/csv";
import { MAX_ROWS } from "@/tabular/limits";
import { SAMPLES } from "@/tabular/samples";
import type { Dataset, ParseIssue } from "@/tabular/types";

import { DatasetPanel } from "./DatasetPanel";

const dataset = parseCsv("age,city,defaulted\n30,north,yes\n40,south,no\n50,,yes\n", {
  name: "loans.csv",
}).dataset;

function setup(
  over: {
    dataset?: Dataset | null;
    sample?: (typeof SAMPLES)[number] | null;
    parsing?: boolean;
    error?: string | null;
    issues?: ParseIssue[];
    disabled?: boolean;
  } = {},
) {
  const onSample = vi.fn();
  const onFile = vi.fn();
  render(
    <DatasetPanel
      dataset={over.dataset ?? null}
      sample={over.sample ?? null}
      parsing={over.parsing ?? false}
      error={over.error ?? null}
      issues={over.issues ?? []}
      onSample={onSample}
      onFile={onFile}
      disabled={over.disabled ?? false}
    />,
  );
  return { onSample, onFile };
}

describe("DatasetPanel", () => {
  it("states the privacy claim before anything is chosen", () => {
    // The sentence is the reason this page exists, so it is on screen in the
    // place where the user is about to act on it — not in a footer.
    setup();
    const claim = screen.getByText(/never leaves this device/i);
    expect(claim).toBeInTheDocument();
    expect(claim.closest("p")).toHaveTextContent(/nothing is cached to disk/i);
    expect(claim.closest("p")).toHaveTextContent(/reloading the page loses the file/i);
  });

  it("offers every bundled sample and reports the choice", () => {
    const { onSample } = setup();
    for (const s of SAMPLES) {
      expect(screen.getByRole("button", { name: s.label })).toBeInTheDocument();
    }
    fireEvent.click(screen.getByRole("button", { name: SAMPLES[1].label }));
    expect(onSample).toHaveBeenCalledWith(SAMPLES[1]);
  });

  it("credits the chosen sample's source and licence", () => {
    // #24: a licence is the one thing that can invalidate a finished route.
    setup({ sample: SAMPLES[1] });
    expect(screen.getByRole("button", { name: SAMPLES[1].label })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    // Matched as a substring rather than a regex: several licence strings
    // contain brackets, which a naive `new RegExp` turns into a group.
    expect(
      screen.getByText((_, el) => el?.textContent?.includes(SAMPLES[1].licence) ?? false, {
        selector: "p",
      }),
    ).toBeInTheDocument();
  });

  it("quotes the row cap next to the file picker", () => {
    setup();
    expect(screen.getByText(new RegExp(`${MAX_ROWS.toLocaleString()} rows`))).toBeInTheDocument();
    expect(screen.getByText(/sampled evenly across the file/i)).toBeInTheDocument();
  });

  it("accepts a dropped file as well as a chosen one", () => {
    const { onFile } = setup();
    const file = new File(["a,b\n1,2\n"], "drop.csv", { type: "text/csv" });
    const zone = screen.getByText(/or drop one here/i).closest("div")!;
    fireEvent.drop(zone, { dataTransfer: { files: [file] } });
    expect(onFile).toHaveBeenCalledWith(file);
  });

  it("leaves the input sources usable while a fit is impossible", () => {
    // The corollary of "only FIT and PREDICT spend": choosing data is free.
    setup();
    expect(screen.getByLabelText(/csv file/i)).toBeEnabled();
    expect(screen.getByRole("button", { name: SAMPLES[0].label })).toBeEnabled();
  });

  it("locks the sources while a fit is in flight", () => {
    setup({ disabled: true });
    expect(screen.getByRole("button", { name: SAMPLES[0].label })).toBeDisabled();
  });

  it("says it is parsing, because a big file is a visible pause", () => {
    setup({ parsing: true });
    expect(screen.getByText(/parsing…/i)).toBeInTheDocument();
  });

  it("renders a parse failure as an alert and no summary", () => {
    setup({ error: "The file is empty." });
    expect(screen.getByRole("alert")).toHaveTextContent(/the file is empty/i);
    expect(screen.queryByTestId("dataset-summary")).toBeNull();
  });

  it("summarises the shape of the dataset it was handed", () => {
    setup({ dataset });
    const summary = screen.getByTestId("dataset-summary");
    expect(summary).toHaveTextContent("loans.csv");
    expect(summary).toHaveTextContent("3 rows × 3 columns");
  });

  it("says both counts when the cap bit, never just the kept one", () => {
    // "50 000 rows" about the first 50 000 rows of a file sorted by date is true
    // and misleading; the source count is what makes it honest.
    // A real sampled dataset: the arrays hold what `rowCount` says, and only
    // `sourceRowCount` records what the file had.
    const sampled = parseCsv("age,city,defaulted\n30,north,yes\n40,south,no\n50,,yes\n", {
      name: "big.csv",
    }).dataset;
    setup({ dataset: { ...sampled, sourceRowCount: 900_000, sampled: true } });
    const summary = screen.getByTestId("dataset-summary");
    expect(summary).toHaveTextContent("900,000 rows");
    expect(summary).toHaveTextContent(/sampled evenly across it/i);
    expect(summary).toHaveTextContent(/describes that sample/i);
  });

  it("names the skipped rows and where the first one is", () => {
    // Rejected with a line number, never padded — a padded row shifts every
    // column after the gap and trains happily.
    setup({ dataset, issues: [{ line: 42, message: "3 fields, expected 4" }] });
    const summary = screen.getByTestId("dataset-summary");
    expect(summary).toHaveTextContent("1 row skipped");
    expect(summary).toHaveTextContent("line 42");
    expect(summary).toHaveTextContent(/shifts every column after the gap/i);
  });

  it("pluralises the skipped-row count", () => {
    setup({
      dataset,
      issues: [
        { line: 4, message: "x" },
        { line: 9, message: "x" },
      ],
    });
    expect(screen.getByTestId("dataset-summary")).toHaveTextContent("2 rows skipped");
  });

  it("previews the real cells, and marks a missing one as missing rather than as 0", () => {
    // `Number("")` is 0 — the coercion that shifts every mean a column touches.
    // The preview is where the user can see that it did not happen.
    setup({ dataset });
    const table = within(screen.getByTestId("dataset-summary")).getByRole("table");
    expect(within(table).getAllByRole("row")).toHaveLength(4);
    expect(table).toHaveTextContent("north");
    expect(within(table).getAllByText(/^missing$/i).length).toBe(1);
  });

  it("marks each column's kind in the preview header", () => {
    setup({ dataset });
    const headers = within(screen.getByTestId("dataset-summary")).getAllByRole("columnheader");
    // Numeric columns are marked `#` and categorical ones `abc`, so the reader
    // can see how the parser typed each column before fitting anything on it.
    expect(headers[0]).toHaveTextContent(/^age\s*#$/);
    expect(headers[1]).toHaveTextContent(/^city\s*abc$/);
  });
});
