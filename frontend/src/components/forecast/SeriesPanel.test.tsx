import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { FORECAST_SAMPLES } from "@/forecast/samples";
import { parseSeries, type Series } from "@/forecast/series";

import { SeriesPanel } from "./SeriesPanel";

function setup(
  over: {
    series?: Series | null;
    sample?: (typeof FORECAST_SAMPLES)[number] | null;
    text?: string;
    error?: string | null;
  } = {},
) {
  const onText = vi.fn();
  const onSample = vi.fn();
  const onFile = vi.fn();
  render(
    <SeriesPanel
      series={over.series ?? null}
      sample={over.sample ?? null}
      text={over.text ?? ""}
      onText={onText}
      error={over.error ?? null}
      onSample={onSample}
      onFile={onFile}
    />,
  );
  return { onText, onSample, onFile };
}

const clean = parseSeries("2024-01-01,1\n2024-01-02,2\n2024-01-03,3\n2024-01-04,4\n", "clean");
const gappy = parseSeries("2024-01-01,1\n2024-01-02,2\n2024-01-05,3\n2024-01-06,4\n", "gappy");
const irregular = parseSeries(
  [
    "2024-01-01T00:00:00Z,1",
    "2024-01-01T01:00:00Z,2",
    "2024-01-01T01:37:00Z,3",
    "2024-01-01T02:37:00Z,4",
  ].join("\n"),
  "irregular",
);

describe("SeriesPanel", () => {
  it("says nothing is downloaded and nothing leaves the device", () => {
    setup();
    const claim = screen.getByText(/nothing leaves this device/i);
    expect(claim.closest("p")).toHaveTextContent(/nothing is downloaded/i);
    expect(claim.closest("p")).toHaveTextContent(/main thread in this tab/i);
  });

  it("offers every bundled series and reports the choice", () => {
    const { onSample } = setup();
    for (const s of FORECAST_SAMPLES) {
      expect(screen.getByRole("button", { name: s.label })).toBeInTheDocument();
    }
    fireEvent.click(screen.getByRole("button", { name: FORECAST_SAMPLES[0].label }));
    expect(onSample).toHaveBeenCalledWith(FORECAST_SAMPLES[0]);
  });

  it("credits the chosen series' source and licence", () => {
    const sample = FORECAST_SAMPLES[0];
    setup({ sample });
    expect(screen.getByRole("button", { name: sample.label })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(
      screen.getByText((_, el) => el?.textContent?.includes(sample.licence) ?? false, {
        selector: "p",
      }),
    ).toBeInTheDocument();
  });

  it("accepts a pasted column and reports every keystroke", () => {
    // The text box *is* the state: there is no parse step to trigger, so the gap
    // report is live while the user is still looking at what caused it.
    const { onText } = setup();
    fireEvent.change(screen.getByLabelText(/paste a column/i), { target: { value: "1\n2\n3\n" } });
    expect(onText).toHaveBeenCalledWith("1\n2\n3\n");
  });

  it("accepts a dropped file and a chosen one", () => {
    const { onFile } = setup();
    const file = new File(["1\n2\n3\n"], "series.csv", { type: "text/csv" });
    fireEvent.drop(screen.getByLabelText(/paste a column/i).parentElement!, {
      dataTransfer: { files: [file] },
    });
    expect(onFile).toHaveBeenCalledWith(file);
  });

  it("summarises the series, with its detected frequency", () => {
    setup({ series: clean });
    const summary = screen.getByTestId("series-summary");
    expect(summary).toHaveTextContent("clean");
    expect(summary).toHaveTextContent("4 points");
    expect(summary).toHaveTextContent("daily");
  });

  it("says when no dates were supplied rather than implying a frequency", () => {
    setup({ series: parseSeries("1\n2\n3\n", "bare") });
    expect(screen.getByTestId("series-summary")).toHaveTextContent(/no dates supplied/i);
  });

  it("reports a gap, and says it is reported rather than filled", () => {
    // A silently interpolated gap gives a seasonal forecast that is confidently
    // off by a phase, and the error is then blamed on the method.
    setup({ series: gappy });
    const note = screen.getByTestId("series-gaps");
    expect(note).toHaveTextContent("1 gap");
    expect(note).toHaveTextContent("2 periods missing");
    expect(note).toHaveTextContent(/reported, not filled/i);
    expect(note).toHaveTextContent(/off by a phase/i);
  });

  it("shows no gap note on a regular series", () => {
    setup({ series: clean });
    expect(screen.queryByTestId("series-gaps")).toBeNull();
    expect(screen.queryByTestId("series-irregular")).toBeNull();
  });

  it("flags irregular spacing, and says what that means for the controls", () => {
    // Every method here assumes an even step, so the season and the horizon stop
    // being durations and become counts of points.
    setup({ series: irregular });
    const note = screen.getByTestId("series-irregular");
    expect(note).toHaveTextContent(/not consistent/i);
    expect(note).toHaveTextContent(/counts of points/i);
  });

  it("renders a parse error as an alert, with no summary beside it", () => {
    setup({ error: "Line 3: “oops” is not a number.", text: "1\n2\noops\n" });
    expect(screen.getByRole("alert")).toHaveTextContent(/line 3/i);
    expect(screen.queryByTestId("series-summary")).toBeNull();
  });
});
