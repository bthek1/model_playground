// The dataset surface: pick a sample, or drop your own file.
//
// It lives in the RUN band and it is where the page makes its actual claim —
// stated once, plainly, in the place where the user is about to act on it. Not
// a badge, not a footer: the sentence is the reason this page exists.
//
// Nothing here runs a model. Choosing a sample, dropping a file, picking a
// target column and toggling a feature all land in INPUT and stop there; only
// FIT spends. "Pick a target column and it fits" feels responsive and is the
// five-samples-five-inferences failure with a dropdown in front of it.

import { FileWarning, Loader2, ShieldCheck, Table2, Upload } from "lucide-react";
import { useRef, useState } from "react";

import { ErrorNote } from "@/components/model/ErrorNote";
import { Button } from "@/components/ui/button";
import { cellText } from "@/tabular/csv";
import { MAX_ROWS } from "@/tabular/limits";
import { SAMPLES, type SampleDataset } from "@/tabular/samples";
import type { Dataset, ParseIssue } from "@/tabular/types";
import { cn } from "@/lib/utils";

const PREVIEW_ROWS = 5;

export function DatasetPanel({
  dataset,
  sample,
  parsing,
  error,
  issues,
  onSample,
  onFile,
  disabled = false,
}: {
  dataset: Dataset | null;
  sample: SampleDataset | null;
  parsing: boolean;
  error: string | null;
  issues: ParseIssue[];
  onSample: (sample: SampleDataset) => void;
  onFile: (file: File) => void;
  disabled?: boolean;
}) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  return (
    <div className="space-y-3" data-testid="dataset-panel">
      <p className="flex items-start gap-1.5 rounded-md border border-emerald-600/30 bg-emerald-600/5 p-2 text-xs text-emerald-700 dark:text-emerald-400">
        <ShieldCheck className="mt-0.5 size-3.5 shrink-0" />
        <span>
          <strong>Your file never leaves this device.</strong> It is parsed and
          the model is fitted in a Web Worker in this tab. Nothing is uploaded,
          nothing is cached to disk, and no record of the columns or their values
          is sent anywhere — reloading the page loses the file, by design.
        </span>
      </p>

      <div className="space-y-1.5">
        <p className="text-xs font-medium">Sample data</p>
        <div className="flex flex-wrap gap-1.5">
          {SAMPLES.map((s) => (
            <Button
              key={s.id}
              size="sm"
              variant={sample?.id === s.id ? "default" : "outline"}
              disabled={disabled || parsing}
              aria-pressed={sample?.id === s.id}
              onClick={() => onSample(s)}
            >
              {s.label}
            </Button>
          ))}
        </div>
        {sample && (
          <p className="text-xs leading-snug text-muted-foreground">
            {sample.blurb}{" "}
            <span className="whitespace-nowrap">
              {sample.source} · {sample.licence}
            </span>
          </p>
        )}
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files?.[0];
          if (file) onFile(file);
        }}
        className={cn(
          "rounded-md border border-dashed p-3 text-center text-xs text-muted-foreground",
          dragging && "border-primary bg-primary/5",
        )}
      >
        <input
          ref={fileInput}
          type="file"
          accept=".csv,text/csv"
          className="sr-only"
          aria-label="CSV file"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFile(file);
            e.target.value = "";
          }}
        />
        <Button
          size="sm"
          variant="outline"
          disabled={disabled || parsing}
          onClick={() => fileInput.current?.click()}
        >
          <Upload className="size-4" /> Choose a CSV
        </Button>
        <p className="mt-1.5">
          …or drop one here. Up to {MAX_ROWS.toLocaleString()} rows are kept,
          sampled evenly across the file.
        </p>
      </div>

      {parsing && (
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Loader2 className="size-3.5 animate-spin" /> Parsing…
        </p>
      )}

      <ErrorNote message={error} />

      {dataset && (
        <div className="space-y-2" data-testid="dataset-summary">
          <p className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
            <Table2 className="size-3.5" />
            <span className="font-medium text-foreground">{dataset.name}</span>·{" "}
            {dataset.rowCount.toLocaleString()} rows ×{" "}
            {dataset.columns.length} columns
          </p>
          {dataset.sampled && (
            <p className="text-xs text-amber-600 dark:text-amber-500">
              The file has {dataset.sourceRowCount.toLocaleString()} rows;{" "}
              {dataset.rowCount.toLocaleString()} were sampled evenly across it.
              Every number on this page describes that sample.
            </p>
          )}
          {issues.length > 0 && (
            <p className="flex items-start gap-1.5 text-xs text-amber-600 dark:text-amber-500">
              <FileWarning className="mt-0.5 size-3.5 shrink-0" />
              <span>
                {issues.length} row{issues.length === 1 ? "" : "s"} skipped for
                having the wrong number of fields — first at line{" "}
                {issues[0].line}. They are skipped rather than padded: a padded
                row shifts every column after the gap.
              </span>
            </p>
          )}
          <PreviewTable dataset={dataset} />
        </div>
      )}
    </div>
  );
}

function PreviewTable({ dataset }: { dataset: Dataset }) {
  const rows = Math.min(PREVIEW_ROWS, dataset.rowCount);
  return (
    <div className="overflow-x-auto rounded-md border">
      <table className="w-full text-left text-xs">
        <thead className="bg-muted/50">
          <tr>
            {dataset.columns.map((c) => (
              <th key={c.name} className="px-2 py-1 font-medium whitespace-nowrap">
                {c.name}
                <span className="ml-1 font-normal text-muted-foreground">
                  {c.kind === "numeric" ? "#" : "abc"}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: rows }, (_, r) => (
            <tr key={r} className="border-t">
              {dataset.columns.map((c) => (
                <td
                  key={c.name}
                  className="px-2 py-1 whitespace-nowrap tabular-nums"
                >
                  {c.missing[r] ? (
                    <span className="text-muted-foreground italic">missing</span>
                  ) : (
                    cellText(c, r)
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
