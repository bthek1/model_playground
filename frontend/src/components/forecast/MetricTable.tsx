// Every method's error on the same held-out window.
//
// The arrangement here is the inverse of every other page in the app: there is
// no model to put the baselines beside, because the baselines **are** the null
// models. So the comparison on screen is between them — and MASE is the column
// that makes it legible, because it is scaled by the in-sample naive error.
// 1.0 means "no better than repeating the last value", which is a sentence; an
// MAE of 47.8 is not.

import type { ForecastMetrics } from "@/forecast/metrics";

export interface MethodRow {
  id: string;
  label: string;
  metrics: ForecastMetrics;
}

function num(x: number | null, digits = 2): string {
  return x == null || !Number.isFinite(x) ? "—" : x.toFixed(digits);
}

export function MetricTable({
  rows,
  /** Which row the user has selected, highlighted rather than filtered. */
  selected,
}: {
  rows: MethodRow[];
  selected?: string;
}) {
  const best = rows.reduce<MethodRow | null>(
    (acc, r) => (acc == null || r.metrics.mae < acc.metrics.mae ? r : acc),
    null,
  );
  const anySkipped = rows.some((r) => r.metrics.mapeSkipped > 0);

  return (
    <div className="space-y-1" data-testid="metric-table">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="text-muted-foreground">
            <tr>
              <th scope="col" className="py-1 pr-2 font-medium">Method</th>
              <th scope="col" className="py-1 pr-2 text-right font-medium">MAE</th>
              <th scope="col" className="py-1 pr-2 text-right font-medium">RMSE</th>
              <th scope="col" className="py-1 pr-2 text-right font-medium">MAPE</th>
              <th scope="col" className="py-1 text-right font-medium">MASE</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.id}
                data-selected={r.id === selected || undefined}
                className={
                  r.id === selected ? "border-t bg-muted/50 font-medium" : "border-t"
                }
              >
                <td className="py-1 pr-2">
                  {r.label}
                  {best?.id === r.id && (
                    <span className="ml-1 text-emerald-600 dark:text-emerald-500">
                      best
                    </span>
                  )}
                </td>
                <td className="py-1 pr-2 text-right tabular-nums">{num(r.metrics.mae)}</td>
                <td className="py-1 pr-2 text-right tabular-nums">{num(r.metrics.rmse)}</td>
                <td className="py-1 pr-2 text-right tabular-nums">
                  {r.metrics.mape == null ? "—" : `${(r.metrics.mape * 100).toFixed(1)}%`}
                </td>
                <td className="py-1 text-right tabular-nums">{num(r.metrics.mase)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs leading-snug text-muted-foreground">
        <strong>MASE is the column to read.</strong> It is the error divided by
        the in-sample naive error, so <strong>1.0 means “no better than
        repeating the last value”</strong> and below 1 is a real improvement over
        the thing you have to beat. MAE and RMSE are in the series' own units and
        say nothing on their own.
      </p>
      {anySkipped && (
        <p className="text-xs leading-snug text-muted-foreground">
          MAPE is blank because the held-out window contains a zero, and MAPE
          divides by the actual. It is left blank rather than averaged over the
          other rows — that average would describe a different test set from every
          other number in the table.
        </p>
      )}
    </div>
  );
}
