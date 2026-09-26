// Time Series Forecasting — the baselines, and why one split lies.
//
// This is the only route in the repo with **no model, no download and no
// worker**, and each of those is a decision rather than an omission.
//
// No model: the two foundation forecasters the roadmap named — TimesFM 2.0 and
// PatchTST — publish no ONNX weights (re-checked 2026-09-25), which is a missing
// *export* rather than a size problem, so no amount of quantization reaches
// them. That does not stop the page, because the baselines are the page: naive,
// seasonal naive and drift are twenty lines each, and a forecasting model that
// cannot beat them has not earned its download.
//
// No worker: naive is `last`, seasonal naive is `last season`, and a rolling
// backtest is a loop over slices. The whole page is O(points × windows) on a few
// thousand points. Wrapping pure arithmetic in a worker to look consistent with
// the other modalities would add a protocol, a mock and an asynchronous
// boundary with nothing to put across it.
//
// **So it is a three-band page**, and that is written up in
// `docs/standards/model-page-pattern.md` §7 rather than quietly shipped, because
// §8's testid contract assumes four. SELECT is the method and the season; RUN is
// the series, the horizon and the backtest settings; OUTPUT is the forecast, the
// metrics and the window spread. There is no FIT band because there is nothing
// to fit — and the page says so where the band would have been, since an
// unexplained absence reads as an oversight.
//
// Every control re-derives on the main thread and **nothing here is gated**:
// this page has no state machine to be `ready` in.

import { useEffect, useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Activity } from "lucide-react";

import { BacktestStrip, type MetricKey } from "@/components/forecast/BacktestStrip";
import { MetricTable, type MethodRow } from "@/components/forecast/MetricTable";
import { SeriesChart } from "@/components/forecast/SeriesChart";
import { SeriesPanel } from "@/components/forecast/SeriesPanel";
import { ModelSlot } from "@/components/model/ModelPage";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { useSeries } from "@/hooks/useSeries";
import { backtest, type BacktestMode } from "@/forecast/backtest";
import { BASELINES, forecast, type BaselineId } from "@/forecast/baselines";
import { forecastMetrics } from "@/forecast/metrics";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/time-series-forecasting")({
  component: TimeSeriesForecastingPage,
});

const METHOD_IDS: BaselineId[] = ["naive", "seasonal-naive", "drift", "mean"];

function TimeSeriesForecastingPage() {
  const input = useSeries();
  const series = input.series;

  const [method, setMethod] = useState<BaselineId>("seasonal-naive");
  const [season, setSeason] = useState(12);
  const [horizon, setHorizon] = useState(12);
  const [windows, setWindows] = useState(8);
  const [stride, setStride] = useState(6);
  const [mode, setMode] = useState<BacktestMode>("expanding");
  const [windowLength, setWindowLength] = useState(60);
  const [metric, setMetric] = useState<MetricKey>("mase");

  // A sample brings its own season and a sensible horizon. Loading one does not
  // compute anything the page was not already computing — every number here is
  // a `useMemo` away, so there is nothing to "run".
  const sample = input.sample;
  useEffect(() => {
    if (!sample) return;
    setSeason(sample.season);
    setHorizon(sample.horizon);
  }, [sample]);

  const n = series?.values.length ?? 0;
  const maxHorizon = Math.max(1, Math.floor(n / 3));
  const safeHorizon = Math.min(horizon, maxHorizon);

  /**
   * Everything below is derived. No control on this page is gated on anything
   * and none of them starts a task: they re-read a series that is already in
   * memory, which is the same rule `/vad`'s threshold and detection's score
   * floor follow, taken to its limit.
   */
  const derived = useMemo(() => {
    if (!series || n < safeHorizon + 2) return null;
    const testStart = n - safeHorizon;
    const history = series.values.subarray(0, testStart);
    const actual = Float32Array.from(series.values.subarray(testStart));

    const rows: MethodRow[] = METHOD_IDS.map((id) => {
      const predicted = forecast(id, Float32Array.from(history), safeHorizon, season);
      return {
        id,
        label: BASELINES.find((b) => b.id === id)?.label ?? id,
        metrics: forecastMetrics(actual, predicted, Float32Array.from(history), season),
      };
    });

    const lines = METHOD_IDS.map((id) => ({
      label: BASELINES.find((b) => b.id === id)?.label ?? id,
      values: forecast(id, Float32Array.from(history), safeHorizon, season),
    }));

    let backtested: ReturnType<typeof backtest> | null = null;
    let backtestError: string | null = null;
    try {
      backtested = backtest(series.values, method, {
        horizon: safeHorizon,
        windows,
        stride,
        mode,
        windowLength,
        season,
      });
    } catch (e) {
      backtestError = e instanceof Error ? e.message : String(e);
    }

    return { testStart, rows, lines, backtested, backtestError };
  }, [series, n, safeHorizon, season, method, windows, stride, mode, windowLength]);

  return (
    <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-6 md:h-full md:min-h-0">
      <header className="min-w-0">
        <h1 className="mb-1 flex items-center gap-2 text-2xl font-semibold">
          <Activity className="size-6" /> Time Series Forecasting
        </h1>
        <p className="text-sm text-muted-foreground">
          The baselines you have to beat — and a rolling-origin backtest that
          shows why a single train/test split is not an evaluation. Nothing is
          downloaded, nothing is uploaded, and there is no model here at all.
        </p>
      </header>

      {/* Three bands, not four. See the note at the top of this file and
          model-page-pattern.md §7: there is nothing to fit, so rather than
          leaving a gap the page says why the band is missing — an unexplained
          absence reads as an oversight, and the absence is the category's
          point. */}
      <div
        className={cn(
          "grid min-h-0 grid-cols-1 gap-6 md:flex-1",
          "md:grid-cols-2 md:grid-rows-[auto_minmax(0,1fr)]",
          "md:[grid-template-areas:'setup_setup'_'work-a_work-b']",
          "xl:grid-cols-[minmax(17rem,20rem)_minmax(0,1fr)_minmax(0,1fr)]",
          "xl:grid-rows-[minmax(0,1fr)]",
          "xl:[grid-template-areas:'setup_work-a_work-b']",
        )}
      >
        <div className="flex min-w-0 flex-col gap-5 rounded-lg border bg-muted/30 p-4 md:[grid-area:setup] md:flex-row md:gap-8 xl:max-h-full xl:flex-col xl:gap-5 xl:self-start xl:overflow-y-auto">
          <ModelSlot step={1} label="Method" dense className="min-w-0 md:flex-1 xl:flex-none">
            <MethodPicker
              method={method}
              onMethod={setMethod}
              season={season}
              onSeason={setSeason}
              suggested={series?.suggestedSeason ?? null}
              maxSeason={Math.max(2, Math.floor(n / 3))}
            />
          </ModelSlot>

          <div
            className="min-w-0 space-y-1.5 md:w-80 md:shrink-0 xl:w-auto"
            data-testid="no-fit-band"
          >
            <p className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
              No model to fit
            </p>
            <p className="text-xs leading-snug text-muted-foreground">
              Every other page in this app has a band here for downloading
              weights or fitting a model. This one has nothing to put in it:
              naive, seasonal naive and drift are closed-form arithmetic over the
              series you have already given it, so there is nothing to download,
              nothing to train and nothing to wait for. Every control on this
              page re-computes instantly.
            </p>
            <p className="text-xs leading-snug text-muted-foreground">
              The two foundation forecasters worth wanting — TimesFM 2.0 and
              PatchTST — publish no ONNX weights, so they cannot run in a
              browser at all. That is a missing export rather than a size
              problem: quantization does not reach it.
            </p>
          </div>
        </div>

        <div className="flex min-h-0 min-w-0 flex-col md:overflow-y-auto md:[grid-area:work-a]">
          <ModelSlot
            step={2}
            label="Series"
            className="flex min-h-0 flex-1 flex-col [&>*:last-child]:min-h-0 [&>*:last-child]:flex-1"
          >
            <div className="space-y-4">
              <SeriesPanel
                series={series}
                sample={input.sample}
                text={input.text}
                onText={input.setText}
                error={input.error}
                onSample={(s) => void input.loadSample(s)}
                onFile={(f) => void input.loadFile(f)}
              />
              {series && (
                <BacktestControls
                  horizon={safeHorizon}
                  maxHorizon={maxHorizon}
                  onHorizon={setHorizon}
                  windows={windows}
                  onWindows={setWindows}
                  stride={stride}
                  onStride={setStride}
                  mode={mode}
                  onMode={setMode}
                  windowLength={windowLength}
                  onWindowLength={setWindowLength}
                  maxWindowLength={Math.max(4, n - safeHorizon)}
                />
              )}
            </div>
          </ModelSlot>
        </div>

        <div className="flex min-h-0 min-w-0 flex-col md:overflow-y-auto md:[grid-area:work-b]">
          <ModelSlot
            step={3}
            label="Result"
            className="flex min-h-0 flex-1 flex-col [&>*:last-child]:min-h-0 [&>*:last-child]:flex-1"
          >
            <OutputPanel
              title="Forecast and backtest"
              description={
                series
                  ? `${series.values.length} points · horizon ${safeHorizon}${series.frequency ? ` · ${series.frequency}` : ""}`
                  : undefined
              }
              running={false}
              empty={
                <span data-testid="no-model-note">
                  Choose a series, or paste one. You will get four baselines,
                  their errors on a held-out window, and the same method scored
                  across rolling windows.{" "}
                  <strong>
                    There is no learned model on this page — the baselines are
                    the null models, and beating them is what a real forecaster
                    would have to do.
                  </strong>
                </span>
              }
            >
              {series && derived ? (
                <div className="space-y-4">
                  <SeriesChart
                    series={series}
                    testStart={derived.testStart}
                    lines={derived.lines}
                  />
                  <MetricTable rows={derived.rows} selected={method} horizon={safeHorizon} />

                  <div className="space-y-2 border-t pt-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="text-xs font-medium">
                        Rolling-origin backtest — {BASELINES.find((b) => b.id === method)?.label}
                      </p>
                      <div className="flex gap-1">
                        {(["mase", "mae", "rmse"] as MetricKey[]).map((m) => (
                          <Button
                            key={m}
                            size="sm"
                            variant={metric === m ? "secondary" : "ghost"}
                            aria-pressed={metric === m}
                            className="h-6 px-2 text-xs"
                            onClick={() => setMetric(m)}
                          >
                            {m.toUpperCase()}
                          </Button>
                        ))}
                      </div>
                    </div>
                    {derived.backtestError ? (
                      <p className="text-xs text-amber-600 dark:text-amber-500">
                        {derived.backtestError}
                      </p>
                    ) : derived.backtested ? (
                      <>
                        <BacktestStrip result={derived.backtested} metric={metric} />
                        {derived.backtested.note && (
                          <p className="text-xs text-muted-foreground">
                            {derived.backtested.note}
                          </p>
                        )}
                        <p className="text-xs leading-snug text-muted-foreground">
                          Every window trains only on points before its own test
                          window — the split is rolling-origin, never shuffled.
                          Shuffled cross-validation on a time series trains on
                          the future and scores the past, and it fails{" "}
                          <em>upward</em>: the metric improves and nothing
                          complains.
                        </p>
                      </>
                    ) : null}
                  </div>
                </div>
              ) : null}
            </OutputPanel>
          </ModelSlot>
        </div>
      </div>
    </div>
  );
}

function MethodPicker({
  method,
  onMethod,
  season,
  onSeason,
  suggested,
  maxSeason,
}: {
  method: BaselineId;
  onMethod: (id: BaselineId) => void;
  season: number;
  onSeason: (n: number) => void;
  suggested: number | null;
  maxSeason: number;
}) {
  const info = BASELINES.find((b) => b.id === method);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5">
        {BASELINES.map((b) => (
          <Button
            key={b.id}
            size="sm"
            variant={b.id === method ? "default" : "outline"}
            aria-pressed={b.id === method}
            onClick={() => onMethod(b.id)}
          >
            {b.label}
          </Button>
        ))}
      </div>
      <p className="text-xs leading-snug text-muted-foreground">{info?.blurb}</p>
      <p className="text-xs leading-snug text-muted-foreground">
        The method chosen here is the one the <strong>backtest</strong> runs; the
        metric table below scores all four on the same held-out window, so they
        are always comparable.
      </p>

      <div className="space-y-1 border-t pt-3">
        <label
          htmlFor="season-length"
          className="flex items-baseline justify-between text-xs font-medium"
        >
          Season length
          <span className="font-mono text-xs text-muted-foreground tabular-nums">
            {season} points
          </span>
        </label>
        <input
          id="season-length"
          type="range"
          min={1}
          max={maxSeason}
          step={1}
          value={Math.min(season, maxSeason)}
          onChange={(e) => onSeason(Number(e.target.value))}
          className="block w-full"
        />
        <p className="text-xs leading-snug text-muted-foreground">
          <strong>Set by you, never detected.</strong> Guessing the period and
          being wrong produces a confident, plausible forecast that is off by a
          phase — with nothing on screen to say so. It also scales MASE, so it
          changes every number in the table.
          {suggested != null && (
            <>
              {" "}
              This series&apos; spacing suggests <strong>{suggested}</strong>.
            </>
          )}
        </p>
      </div>
    </div>
  );
}

function BacktestControls({
  horizon,
  maxHorizon,
  onHorizon,
  windows,
  onWindows,
  stride,
  onStride,
  mode,
  onMode,
  windowLength,
  onWindowLength,
  maxWindowLength,
}: {
  horizon: number;
  maxHorizon: number;
  onHorizon: (n: number) => void;
  windows: number;
  onWindows: (n: number) => void;
  stride: number;
  onStride: (n: number) => void;
  mode: BacktestMode;
  onMode: (m: BacktestMode) => void;
  windowLength: number;
  onWindowLength: (n: number) => void;
  maxWindowLength: number;
}) {
  return (
    <div className="space-y-2.5 border-t pt-3" data-testid="backtest-controls">
      <p className="text-xs font-medium">Backtest</p>
      <Slider id="horizon" label="Horizon" value={horizon} min={1} max={maxHorizon} onChange={onHorizon} />
      <Slider id="windows" label="Windows" value={windows} min={2} max={30} onChange={onWindows} />
      <Slider id="stride" label="Origin spacing" value={stride} min={1} max={30} onChange={onStride} />
      <div className="space-y-1">
        <p className="text-xs font-medium">Training window</p>
        <div className="flex gap-1.5">
          {(["expanding", "sliding"] as BacktestMode[]).map((m) => (
            <Button
              key={m}
              size="sm"
              variant={mode === m ? "secondary" : "outline"}
              aria-pressed={mode === m}
              className="h-7 px-2 text-xs capitalize"
              onClick={() => onMode(m)}
            >
              {m}
            </Button>
          ))}
        </div>
        <p className="text-xs leading-snug text-muted-foreground">
          Expanding keeps every earlier point; sliding keeps a fixed-length tail,
          which is the honest choice when the series has changed regime.
        </p>
      </div>
      {mode === "sliding" && (
        <Slider
          id="window-length"
          label="Sliding length"
          value={Math.min(windowLength, maxWindowLength)}
          min={4}
          max={maxWindowLength}
          onChange={onWindowLength}
        />
      )}
      <p className="text-xs leading-snug text-muted-foreground">
        All of these re-compute instantly and none of them is gated on anything:
        the whole page is arithmetic over one array, so there is no state to be
        ready in.
      </p>
    </div>
  );
}

function Slider({
  id,
  label,
  value,
  min,
  max,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (n: number) => void;
}) {
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="flex items-baseline justify-between text-xs font-medium">
        {label}
        <span className="font-mono text-xs text-muted-foreground tabular-nums">{value}</span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={Math.max(min, max)}
        step={1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="block w-full"
      />
    </div>
  );
}
