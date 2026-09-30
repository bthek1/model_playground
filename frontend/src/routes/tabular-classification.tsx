// Tabular Classification — the Tabular category's first route, and the first
// page in the app that **downloads nothing and trains in the tab on the user's
// own data**.
//
// Every other route's privacy story is "the weights come to you". This one's is
// "your CSV never leaves the device", which is a stronger claim and the only one
// that matters for the data people actually mind about: payroll, patients,
// sales. It is also the first page whose *input* is the memory budget rather
// than the model — see `tabular/limits.ts` for the measurement that set the row
// cap and which family set it.
//
// Four slots, with one substitution: LOAD becomes **FIT**, because there are no
// weights to download. Machine A is untouched — the route relabels the band, it
// does not rename a status (`docs/standards/model-page-pattern.md` §7).
//
// The two buttons that spend are **FIT** and **PREDICT**. Choosing a sample,
// dropping a file, picking a target, toggling a feature and moving a
// hyperparameter all land in SELECT or INPUT and stop there. This is the page
// where that rule is easiest to break: "pick a target column and it fits" feels
// responsive, and is the five-samples-five-inferences failure with a dropdown in
// front of it. The threshold slider is the legitimate re-derivation — it re-reads
// a fit in hand and must never refit.

import { useEffect, useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Table } from "lucide-react";

import { InputPanel } from "@/components/model/InputPanel";
import { ModelPage } from "@/components/model/ModelPage";
import { OutputPanel } from "@/components/model/OutputPanel";
import { ColumnPicker } from "@/components/tabular/ColumnPicker";
import { ConfusionMatrixView } from "@/components/tabular/ConfusionMatrixView";
import { DatasetPanel } from "@/components/tabular/DatasetPanel";
import { FamilyPicker } from "@/components/tabular/FamilyPicker";
import { FitStatus } from "@/components/tabular/FitStatus";
import { ImportanceBars } from "@/components/tabular/ImportanceBars";
import { MetricBlock } from "@/components/tabular/MetricBlock";
import { ThresholdControl } from "@/components/tabular/ThresholdControl";
import { Button } from "@/components/ui/button";
import { useDataset } from "@/hooks/useDataset";
import { useTabularFit } from "@/hooks/useTabularFit";
import { familiesFor, familyInfo } from "@/tabular/families";
import { metricsAtThreshold } from "@/tabular/metrics";
import type { Family, FitResult, Hyperparams, PredictResult } from "@/tabular/types";

export const Route = createFileRoute("/tabular-classification")({
  component: TabularClassificationPage,
});

const LADDER = familiesFor("classification");

function TabularClassificationPage() {
  const data = useDataset();
  const fit = useTabularFit(data.dataset);

  const [family, setFamily] = useState<Family>("boosting");
  const [hp, setHp] = useState<Hyperparams>(familyInfo("boosting").defaults);
  const [seed, setSeed] = useState(42);
  const [testFraction, setTestFraction] = useState(0.25);
  const [target, setTarget] = useState(-1);
  const [features, setFeatures] = useState<number[]>([]);
  const [threshold, setThreshold] = useState(0.5);
  const [prediction, setPrediction] = useState<PredictResult | null>(null);
  const [predictRow, setPredictRow] = useState<Record<number, string>>({});

  // The hyperparameters follow the family, because the two are not independent:
  // a depth that suits a bagged tree is not the right depth for a boosting
  // round, and inheriting one page's numbers into another's model is the
  // `/link-prediction` mistake — reuse that looks like a decision is often an
  // inheritance. Switching family fits nothing; it only changes what FIT would do.
  function chooseFamily(next: Family) {
    setFamily(next);
    setHp(familyInfo(next).defaults);
  }

  // Preselect the sample's own target and every other column, so the page is one
  // click from a result — a *click*, not a fit.
  const dataset = data.dataset;
  useEffect(() => {
    if (!dataset) {
      setTarget(-1);
      setFeatures([]);
      return;
    }
    const preferred = data.sample
      ? dataset.columns.findIndex((c) => c.name === data.sample?.target)
      : -1;
    const fallback = dataset.columns.findIndex((c) => c.kind === "categorical");
    const chosen = preferred >= 0 ? preferred : fallback;
    setTarget(chosen);
    setFeatures(dataset.columns.map((_, i) => i).filter((i) => i !== chosen));
    setPrediction(null);
    setPredictRow({});
  }, [dataset, data.sample]);

  const result = fit.result;
  // Memoised because it feeds the metric re-derivation below: a fresh `[]`
  // every render would re-run the whole threshold computation on every
  // keystroke in the predict form.
  const labels = useMemo(() => result?.labels ?? [], [result]);
  const binary = labels.length === 2;

  // The threshold slider's whole point: the metric block is re-derived here, on
  // the main thread, from the probabilities the fit already returned. No worker
  // message, no second fit.
  const shown = useMemo(() => {
    if (!result?.classification) return null;
    if (!binary || !result.probabilities || !result.testLabels || !result.trainLabels) {
      return result.classification;
    }
    return metricsAtThreshold(
      result.probabilities,
      result.testLabels,
      labels,
      result.trainLabels,
      threshold,
    );
  }, [result, binary, labels, threshold]);

  const trainRows = dataset
    ? Math.round(dataset.rowCount * (1 - testFraction))
    : 0;

  const blocked = !dataset
    ? "Choose a sample or drop a CSV to fit a model."
    : target < 0
      ? "Pick a target column."
      : features.length === 0
        ? "Pick at least one feature column."
        : null;

  async function runFit() {
    if (blocked || !dataset) return;
    setPrediction(null);
    setThreshold(0.5);
    await fit.fit({
      family,
      objective: "classification",
      targetIndex: target,
      featureIndices: features,
      hp,
      seed,
      testFraction,
    });
  }

  async function runPredict() {
    if (!result || !dataset) return;
    const values = features.map((i) => {
      const raw = predictRow[i] ?? "";
      const column = dataset.columns[i];
      if (raw === "") return null;
      return column.kind === "categorical" ? raw : Number(raw);
    });
    setPrediction(await fit.predict(values));
  }

  const loadError = fit.status === "error" ? fit.error : null;
  const runError = fit.status === "error" ? null : fit.error;

  return (
    <ModelPage
      icon={Table}
      title="Tabular Classification"
      description={
        <>
          Fit four kinds of model to a spreadsheet and see which one wins — a
          logistic regression, a random forest, gradient boosting and a small
          neural network, all trained in this tab. The linear models run as WGSL
          on your GPU; the trees deliberately do not. <strong>Your file is
          never uploaded.</strong>
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
            classes={Math.max(2, labels.length)}
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
              Stratified on the target, so a rare class appears on both sides.
              Every score on this page is measured on rows the fit never saw —
              and the encoding, the imputation and the scaling are all computed
              from the training half alone, because fitting them over the whole
              file raises the held-out score and looks like a better page.
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
        // The RUN band's transport is PREDICT, not FIT. The two buttons that
        // spend on this page are one per band: FIT lives in slot 2, which is
        // the slot it replaced, and PREDICT lives here with the row it reads.
        // Putting a second Fit button in the transport row would make the one
        // control that costs seconds appear twice and mean the same thing.
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
                objective="classification"
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
                  Leave a box empty to send it as missing. Typing changes
                  nothing until you press Predict.
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
              ? `${familyInfo(result.spec.family).label} · ${result.testRows.toLocaleString()} held-out rows`
              : undefined
          }
          running={fit.running}
          runningLabel={
            fit.partial ? `${fit.partial.phase}…` : "Fitting…"
          }
          error={runError}
          empty={
            <>
              Pick a dataset and press <strong>Fit model</strong>. You will get
              accuracy against the majority-class baseline, the confusion matrix,
              and a bar per column showing what the model actually used. Nothing
              is downloaded and nothing is uploaded.
            </>
          }
        >
          {result && shown ? (
            <div className="space-y-4">
              <MetricBlock metrics={shown} />
              {binary && result.probabilities && (
                <ThresholdControl
                  threshold={threshold}
                  onChange={setThreshold}
                  positiveLabel={labels[1]}
                  metrics={shown}
                />
              )}
              <ConfusionMatrixView matrix={shown.confusion} />
              <ImportanceBars
                importance={result.importance}
                unit="accuracy lost"
              />
              <FitNotes result={result} />
              {prediction && (
                <div className="space-y-1 border-t pt-3" data-testid="prediction">
                  <p className="text-xs font-medium">This row</p>
                  {prediction.labels.map((label, i) => (
                    <p key={label} className="font-mono text-xs tabular-nums">
                      {label} — {(prediction.scores[i] * 100).toFixed(1)}%
                    </p>
                  ))}
                </div>
              )}
            </div>
          ) : null}
        </OutputPanel>
      }
    />
  );
}

/** The sentences that make the result mean something, kept out of the JSX above. */
function FitNotes({ result }: { result: FitResult }) {
  const info = familyInfo(result.spec.family);
  return (
    <div className="space-y-1 border-t pt-3 text-xs leading-snug text-muted-foreground">
      <p>
        {info.label} · {result.trainRows.toLocaleString()} training rows ·{" "}
        {result.testRows.toLocaleString()} held out · seed {result.spec.seed} ·
        fitted in {(result.fitMs / 1000).toFixed(1)}s on{" "}
        {result.compute === "gpu" ? "your GPU" : "your CPU"}.
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
      <p>
        Fit the other three on the same seed and compare. The neural network is
        on this ladder because it usually <em>loses</em> — on tables of a few
        thousand rows, a hundred shallow trees beat it more often than not, and
        that is worth watching happen on your own data rather than being told.
      </p>
    </div>
  );
}
