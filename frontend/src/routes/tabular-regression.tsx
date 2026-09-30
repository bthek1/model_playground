// Tabular Regression — the same engine, the same worker, the same hook and the
// same ladder as `/tabular-classification`. A reviewer should be able to diff
// the two route files and see only the **diagnostics** differing, which is the
// rule that stopped `/image-text-to-text`, `/visual-question-answering` and
// `/video-text-to-text` drifting into three catalogues.
//
// It is a second route rather than a branch on the first for the reason #43 and
// #31 give: same engine, different question. A user looking for regression does
// not click "Tabular Classification", the Hub has separate tags, and the
// diagnostics share nothing — residuals replace the confusion matrix and an
// interval replaces a threshold.
//
// Three things genuinely differ:
//
//   the closed-form ridge   `XᵀX` and `Xᵀy` on the GPU where the work is
//                           O(n·d²) in the row count, the `d×d` Cholesky on the
//                           CPU where it is microseconds. The category's
//                           GPU/CPU split, exact rather than rhetorical.
//   quantile regression     a band instead of a number, with its measured
//                           coverage beside it — a band of the wrong width
//                           looks entirely correct.
//   the log-transform trap  a toggle that **spends**, because a fit on
//                           log1p(y) cannot be re-derived from a fit on y. The
//                           demonstration is that the two error metrics are not
//                           comparable, so `transform.ts` owns the units and
//                           this file only renders what it is handed.

import { useEffect, useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { LineChart } from "lucide-react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ColumnPicker } from "@/components/tabular/ColumnPicker";
import { DatasetPanel } from "@/components/tabular/DatasetPanel";
import { FamilyPicker } from "@/components/tabular/FamilyPicker";
import { FitStatus } from "@/components/tabular/FitStatus";
import { ImportanceBars } from "@/components/tabular/ImportanceBars";
import { RegressionMetricBlock } from "@/components/tabular/RegressionMetrics";
import {
  PredictedVsActual,
  ResidualPlot,
} from "@/components/tabular/ResidualPlot";
import { Button } from "@/components/ui/button";
import { useDataset } from "@/hooks/useDataset";
import { useTabularFit } from "@/hooks/useTabularFit";
import { familiesFor, familyInfo } from "@/tabular/families";
import type { Family, FitResult, Hyperparams, PredictResult } from "@/tabular/types";

export const Route = createFileRoute("/tabular-regression")({
  component: TabularRegressionPage,
});

const LADDER = familiesFor("regression");
const QUANTILES = [0.1, 0.5, 0.9];

function TabularRegressionPage() {
  const data = useDataset();
  const fit = useTabularFit(data.dataset);

  const [family, setFamily] = useState<Family>("boosting");
  const [hp, setHp] = useState<Hyperparams>(
    familyInfo("boosting", "regression").defaults,
  );
  const [seed, setSeed] = useState(42);
  const [testFraction, setTestFraction] = useState(0.25);
  const [target, setTarget] = useState(-1);
  const [features, setFeatures] = useState<number[]>([]);
  const [logTarget, setLogTarget] = useState(false);
  const [prediction, setPrediction] = useState<PredictResult | null>(null);
  const [predictRow, setPredictRow] = useState<Record<number, string>>({});

  function chooseFamily(next: Family) {
    setFamily(next);
    // Not inherited from the classification page, and not shared between the
    // rungs: a depth that suits a Gini split is not automatically right for
    // variance reduction. `/link-prediction` cost 0.19 of AUC learning that.
    setHp(familyInfo(next, "regression").defaults);
  }

  const dataset = data.dataset;
  useEffect(() => {
    if (!dataset) {
      setTarget(-1);
      setFeatures([]);
      return;
    }
    const preferred = data.sample
      ? dataset.columns.findIndex((c) => c.name === data.sample?.regressionTarget)
      : -1;
    const fallback = dataset.columns.findIndex((c) => c.kind === "numeric");
    const chosen = preferred >= 0 ? preferred : fallback;
    setTarget(chosen);
    setFeatures(dataset.columns.map((_, i) => i).filter((i) => i !== chosen));
    setPrediction(null);
    setPredictRow({});
  }, [dataset, data.sample]);

  const result = fit.result;
  const trainRows = dataset ? Math.round(dataset.rowCount * (1 - testFraction)) : 0;

  const blocked = !dataset
    ? "Choose a sample or drop a CSV to fit a model."
    : target < 0
      ? "Pick a numeric target column."
      : features.length === 0
        ? "Pick at least one feature column."
        : null;

  async function runFit() {
    if (blocked || !dataset) return;
    setPrediction(null);
    await fit.fit({
      family,
      objective: "regression",
      targetIndex: target,
      featureIndices: features,
      hp,
      seed,
      testFraction,
      logTarget,
      quantiles: family === "quantile" ? QUANTILES : undefined,
    });
  }

  async function runPredict() {
    if (!result || !dataset) return;
    const values = features.map((i) => {
      const raw = predictRow[i] ?? "";
      if (raw === "") return null;
      return dataset.columns[i].kind === "categorical" ? raw : Number(raw);
    });
    setPrediction(await fit.predict(values));
  }

  const loadError = fit.status === "error" ? fit.error : null;
  const runError = fit.status === "error" ? null : fit.error;

  const coefficients = useMemo(
    () => (result?.coefficients ?? []).slice().sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)),
    [result],
  );

  return (
    <ModelPage
      icon={LineChart}
      title="Tabular Regression"
      description={
        <>
          Predict a number from a spreadsheet, in this tab, on your own data —
          ridge in closed form on your GPU, trees on your CPU, and a quantile
          model that returns an interval instead of a point.{" "}
          <strong>Your file is never uploaded.</strong>
        </>
      }
      labels={{ select: "Model", load: "Fit", run: "Data", output: "Result" }}
      select={
        <div className="space-y-4">
          <FamilyPicker
            families={LADDER}
            value={family}
            onChange={chooseFamily}
            hp={hp}
            onHp={setHp}
            seed={seed}
            onSeed={setSeed}
            rows={trainRows}
            classes={1}
            disabled={fit.running}
          />
          <div className="space-y-1 border-t pt-3">
            <label
              htmlFor="test-fraction"
              className="flex items-baseline justify-between text-xs font-medium"
            >
              Held out for testing
              <span className="font-mono text-xs text-muted-foreground tabular-nums">
                {Math.round(testFraction * 100)}%
              </span>
            </label>
            <input
              id="test-fraction"
              type="range"
              min={0.1}
              max={0.5}
              step={0.05}
              value={testFraction}
              disabled={fit.running}
              onChange={(e) => setTestFraction(Number(e.target.value))}
              className="block w-full"
            />
            <p className="text-xs leading-snug text-muted-foreground">
              Every score is measured on rows the fit never saw, and the
              encoding, imputation and scaling are computed from the training
              half alone — fitting them over the whole file raises the held-out
              score and looks like a better page.
            </p>
          </div>
        </div>
      }
      load={
        <FitStatus
          canFit={blocked == null}
          running={fit.running}
          partial={fit.partial}
          result={result}
          error={loadError}
          onFit={() => void runFit()}
          onStop={fit.stop}
          blockedReason={blocked}
        />
      }
      run={
        <InputPanel
          ready={result != null}
          disabledHint="Fit a model first — then you can score a single row you type here."
          error={data.error}
          controls={
            <Button
              size="sm"
              onClick={() => void runPredict()}
              disabled={result == null || fit.running}
              data-testid="predict-button"
            >
              Predict this row
            </Button>
          }
        >
          <div className="space-y-4">
            <DatasetPanel
              dataset={dataset}
              sample={data.sample}
              parsing={data.parsing}
              error={null}
              issues={data.issues}
              onSample={(s) => void data.loadSample(s)}
              onFile={(f) => void data.loadFile(f)}
              disabled={fit.running}
            />
            {dataset && target >= 0 && (
              <ColumnPicker
                dataset={dataset}
                objective="regression"
                target={target}
                onTarget={(i) => {
                  setTarget(i);
                  setFeatures(
                    dataset.columns.map((_, k) => k).filter((k) => k !== i),
                  );
                }}
                features={features}
                onFeatures={setFeatures}
                disabled={fit.running}
              />
            )}

            <LogToggle
              value={logTarget}
              onChange={setLogTarget}
              disabled={fit.running}
              targetName={dataset && target >= 0 ? dataset.columns[target].name : "the target"}
            />

            {result && dataset && (
              <div className="space-y-1.5 border-t pt-3">
                <p className="text-xs font-medium">Predict one row</p>
                <div className="grid grid-cols-2 gap-1.5">
                  {features.map((i) => (
                    <label key={i} className="text-xs">
                      <span className="text-muted-foreground">
                        {dataset.columns[i].name}
                      </span>
                      <input
                        type="text"
                        value={predictRow[i] ?? ""}
                        onChange={(e) =>
                          setPredictRow({ ...predictRow, [i]: e.target.value })
                        }
                        className="mt-0.5 h-7 w-full rounded-md border bg-background px-1.5 text-xs"
                      />
                    </label>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground">
                  Leave a box empty to send it as missing. Typing changes nothing
                  until you press Predict.
                </p>
              </div>
            )}
          </div>
        </InputPanel>
      }
      output={
        <OutputPanel
          title="Held-out performance"
          description={
            result
              ? `${familyInfo(result.spec.family, "regression").label} · ${result.testRows.toLocaleString()} held-out rows`
              : undefined
          }
          running={fit.running}
          runningLabel={fit.partial ? `${fit.partial.phase}…` : "Fitting…"}
          error={runError}
          empty={
            <>
              Pick a dataset and press <strong>Fit model</strong>. You will get
              RMSE, MAE and R² against the train-mean baseline, predicted against
              actual, and the residual plot — which is where a good-looking score
              stops being convincing.
            </>
          }
        >
          {result?.regression && result.predictions && result.actuals ? (
            <div className="space-y-4">
              <RegressionMetricBlock
                metrics={result.regression}
                logSpace={result.logSpaceRegression}
              />

              {result.rankDeficient && (
                <p
                  data-testid="rank-deficient"
                  className="rounded-md border border-amber-600/40 bg-amber-600/5 p-2 text-xs leading-snug text-amber-700 dark:text-amber-500"
                >
                  <strong>The design is rank-deficient.</strong> Two of these
                  feature columns carry the same information, so the normal
                  equations have no unique solution — the factorisation could not
                  complete without a penalty, and one was added to finish the
                  fit. This is exactly what ridge's λ is for, and it is why this
                  page solves by Cholesky rather than inverting: an inverse would
                  have returned one of infinitely many answers and looked fine.
                </p>
              )}

              <PredictedVsActual
                predicted={result.predictions}
                actual={result.actuals}
                units={result.regression.units}
                band={result.quantilePredictions}
                quantiles={result.spec.quantiles}
              />

              {result.quantileCoverage != null && (
                <p data-testid="band-coverage" className="text-xs leading-snug text-muted-foreground">
                  The {QUANTILES[0]}–{QUANTILES[QUANTILES.length - 1]} band covers{" "}
                  <strong>{(result.quantileCoverage * 100).toFixed(1)}%</strong> of
                  the held-out rows. It is supposed to cover about{" "}
                  {Math.round((QUANTILES[QUANTILES.length - 1] - QUANTILES[0]) * 100)}%
                  — a band that covers far more is uselessly wide and one that
                  covers far less is a decoration, and neither looks wrong on the
                  chart.
                </p>
              )}

              <ResidualPlot
                predicted={result.predictions}
                actual={result.actuals}
                units={result.regression.units}
              />

              {coefficients.length > 0 ? (
                <Coefficients rows={coefficients} />
              ) : (
                <ImportanceBars importance={result.importance} unit="R² lost" />
              )}

              <FitNotes result={result} />

              {prediction && (
                <div className="space-y-1 border-t pt-3" data-testid="prediction">
                  <p className="text-xs font-medium">This row</p>
                  {prediction.labels.length > 1 ? (
                    prediction.labels.map((label, i) => (
                      <p key={label} className="font-mono text-xs tabular-nums">
                        {label}: {prediction.scores[i].toFixed(3)}
                      </p>
                    ))
                  ) : (
                    <p className="font-mono text-sm tabular-nums">
                      {prediction.scores[0].toFixed(3)}{" "}
                      <span className="font-sans text-xs text-muted-foreground">
                        {result.regression.units}
                      </span>
                    </p>
                  )}
                </div>
              )}
            </div>
          ) : null}
        </OutputPanel>
      }
    />
  );
}

/**
 * The log-transform toggle.
 *
 * It lives in INPUT and it **spends**: a fit on `log1p(y)` cannot be re-derived
 * from a fit on `y`, so flipping it runs nothing and the next FIT is a real
 * second fit. Same class as `/video-text-to-text`'s reverse toggle and
 * `/zero-shot-classification`'s `multi_label` — and, like both, the page says so
 * before the click rather than letting a control that looks like a filter cost
 * seconds.
 */
function LogToggle({
  value,
  onChange,
  disabled,
  targetName,
}: {
  value: boolean;
  onChange: (v: boolean) => void;
  disabled: boolean;
  targetName: string;
}) {
  return (
    <div className="space-y-1 border-t pt-3" data-testid="log-toggle">
      <label className="flex items-center gap-2 text-xs font-medium">
        <input
          type="checkbox"
          checked={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
        />
        Fit on log(1 + {targetName})
      </label>
      <p className="text-xs leading-snug text-muted-foreground">
        A standard move on a target with a long right tail — and the source of a
        mistake people make constantly. This cannot be re-derived from the fit
        you already have, so <strong>changing it runs nothing and the next Fit
        is a real second fit</strong>. Afterwards the page shows the errors in{" "}
        {targetName}&apos;s own units <em>and</em> in log units, separately,
        because the two are not comparable and putting them side by side
        unlabelled is the mistake itself.
      </p>
    </div>
  );
}

function Coefficients({ rows }: { rows: { name: string; weight: number }[] }) {
  const max = Math.max(...rows.map((r) => Math.abs(r.weight)), 1e-9);
  return (
    <div className="space-y-1" data-testid="coefficients">
      <p className="text-xs font-medium">Coefficients</p>
      <div className="space-y-0.5">
        {rows.slice(0, 12).map((r) => (
          <div key={r.name} className="flex items-center gap-2 text-xs">
            <span className="w-40 shrink-0 truncate" title={r.name}>
              {r.name}
            </span>
            <span className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
              <span
                className={
                  r.weight >= 0
                    ? "block h-full bg-emerald-600/70"
                    : "block h-full bg-red-600/70"
                }
                style={{ width: `${(Math.abs(r.weight) / max) * 100}%` }}
              />
            </span>
            <span className="w-16 shrink-0 text-right font-mono tabular-nums">
              {r.weight.toFixed(3)}
            </span>
          </div>
        ))}
      </div>
      <p className="text-xs leading-snug text-muted-foreground">
        On the standardised design, so they are comparable with each other — a
        coefficient is the change in the prediction per standard deviation of its
        column. That is not a causal claim: two columns carrying the same
        information split the credit between them arbitrarily.
      </p>
    </div>
  );
}

function FitNotes({ result }: { result: FitResult }) {
  const info = familyInfo(result.spec.family, "regression");
  return (
    <div className="space-y-1 border-t pt-3 text-xs leading-snug text-muted-foreground">
      <p>
        {info.label} · {result.trainRows.toLocaleString()} training rows ·{" "}
        {result.testRows.toLocaleString()} held out · seed {result.spec.seed} ·
        fitted in {(result.fitMs / 1000).toFixed(1)}s on{" "}
        {result.compute === "gpu" ? "your GPU" : "your CPU"}
        {result.spec.logTarget ? " · fitted on the log-transformed target" : ""}.
      </p>
      {result.droppedRows > 0 && (
        <p data-testid="dropped-rows">
          {result.droppedRows.toLocaleString()} row
          {result.droppedRows === 1 ? " was" : "s were"} left out because the
          target column was blank there. A row with no answer cannot be trained
          on or scored, and imputing a target would be inventing the thing being
          predicted.
        </p>
      )}
      {result.spec.family === "ridge" && (
        <p>
          Ridge is the one model here with a closed form:{" "}
          <code>XᵀX</code> and <code>Xᵀy</code> are built on the GPU, where the
          work is quadratic in the row count, and the small{" "}
          <code>d×d</code> system is factorised on the CPU by Cholesky — which is
          legitimate precisely because λ &gt; 0 makes the matrix positive
          definite. That is what the penalty buys, and it is why this is ridge
          rather than plain least squares.
        </p>
      )}
    </div>
  );
}
