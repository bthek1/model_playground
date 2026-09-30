// The input surface: a bundled series, a pasted column, or a CSV.
//
// It is also where the page refuses to be helpful. Gaps and irregular spacing
// are **reported, not filled**: a silently interpolated gap gives a seasonal
// naive forecast that is confidently off by a phase, the chart looks entirely
// reasonable, and the error is then attributed to the method. Saying what was
// found and letting the user decide is the honest design and the more
// instructive one.

import { CalendarClock, FileWarning, ShieldCheck, Upload } from "lucide-react";
import { useRef, useState } from "react";

import { ErrorNote } from "@/components/model/ErrorNote";
import { Button } from "@/components/ui/button";
import { FORECAST_SAMPLES, type ForecastSample } from "@/forecast/samples";
import type { Series } from "@/forecast/series";
import { cn } from "@/lib/utils";

export function SeriesPanel({
  series,
  sample,
  text,
  onText,
  error,
  onSample,
  onFile,
}: {
  series: Series | null;
  sample: ForecastSample | null;
  text: string;
  onText: (value: string) => void;
  error: string | null;
  onSample: (sample: ForecastSample) => void;
  onFile: (file: File) => void;
}) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  return (
    <div className="space-y-3" data-testid="series-panel">
      <p className="flex items-start gap-1.5 rounded-md border border-emerald-600/30 bg-emerald-600/5 p-2 text-xs text-emerald-700 dark:text-emerald-400">
        <ShieldCheck className="mt-0.5 size-3.5 shrink-0" />
        <span>
          <strong>Nothing leaves this device, and nothing is downloaded.</strong>{" "}
          The whole page is arithmetic over one array, computed on the main
          thread in this tab.
        </span>
      </p>

      <div className="space-y-1.5">
        <p className="text-xs font-medium">Sample series</p>
        <div className="flex flex-wrap gap-1.5">
          {FORECAST_SAMPLES.map((s) => (
            <Button
              key={s.id}
              size="sm"
              variant={sample?.id === s.id ? "default" : "outline"}
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
        className={cn("space-y-1.5", dragging && "rounded-md ring-2 ring-primary")}
      >
        <label htmlFor="series-text" className="text-xs font-medium">
          …or paste a column of numbers, or <code>date,value</code> pairs
        </label>
        <textarea
          id="series-text"
          value={text}
          onChange={(e) => onText(e.target.value)}
          spellCheck={false}
          rows={6}
          className="block w-full rounded-md border bg-background p-2 font-mono text-xs"
          placeholder={"2024-01-01,120\n2024-02-01,135\n2024-03-01,128"}
        />
        <input
          ref={fileInput}
          type="file"
          accept=".csv,.txt,text/csv,text/plain"
          className="sr-only"
          aria-label="Series file"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFile(file);
            e.target.value = "";
          }}
        />
        <Button size="sm" variant="outline" onClick={() => fileInput.current?.click()}>
          <Upload className="size-4" /> Choose a file
        </Button>
      </div>

      <ErrorNote message={error} />

      {series && (
        <div className="space-y-1" data-testid="series-summary">
          <p className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
            <CalendarClock className="size-3.5" />
            <span className="font-medium text-foreground">{series.name}</span>·{" "}
            {series.values.length.toLocaleString()} points
            {series.frequency ? ` · ${series.frequency}` : " · no dates supplied"}
          </p>
          {series.gaps.length > 0 && (
            <p
              data-testid="series-gaps"
              className="flex items-start gap-1.5 text-xs text-amber-600 dark:text-amber-500"
            >
              <FileWarning className="mt-0.5 size-3.5 shrink-0" />
              <span>
                {series.gaps.length} gap{series.gaps.length === 1 ? "" : "s"} —{" "}
                {series.gaps.reduce((a, g) => a + g.missing, 0)} period
                {series.gaps.reduce((a, g) => a + g.missing, 0) === 1 ? "" : "s"}{" "}
                missing. They are <strong>reported, not filled</strong>: an
                interpolated gap gives a seasonal forecast that is confidently off
                by a phase, and the chart looks fine.
              </span>
            </p>
          )}
          {series.irregular && (
            <p
              data-testid="series-irregular"
              className="flex items-start gap-1.5 text-xs text-amber-600 dark:text-amber-500"
            >
              <FileWarning className="mt-0.5 size-3.5 shrink-0" />
              <span>
                The spacing between points is not consistent. Every method here
                assumes an even step, so treat the season length and the horizon
                as counts of <em>points</em> rather than of time.
              </span>
            </p>
          )}
        </div>
      )}
    </div>
  );
}
