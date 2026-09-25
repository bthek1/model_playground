// The confusion matrix — a table, not a chart.
//
// It is here because the aggregate number hides the failure: 92% accuracy on a
// three-class problem is compatible with never predicting the third class at
// all, and the matrix is the only place that shows up. In a notebook it is an
// afterthought; on a page it is the main panel.
//
// Rendered as a real `<table>` with row and column headers rather than a grid of
// coloured divs, so it is readable by a screen reader and the cell counts are
// selectable text. The shading is a secondary encoding, never the only one.

import type { ConfusionMatrix } from "@/tabular/types";

export function ConfusionMatrixView({
  matrix,
  caption,
}: {
  matrix: ConfusionMatrix;
  caption?: string;
}) {
  const k = matrix.labels.length;
  let max = 1;
  for (let i = 0; i < matrix.counts.length; i++) {
    max = Math.max(max, matrix.counts[i]);
  }

  return (
    <div className="space-y-1" data-testid="confusion-matrix">
      <div className="overflow-x-auto">
        <table className="text-left text-xs">
          <caption className="pb-1 text-left text-xs text-muted-foreground">
            {caption ??
              "Rows are the true class, columns what the model said. The diagonal is correct."}
          </caption>
          <thead>
            <tr>
              <th scope="col" className="px-2 py-1 font-normal text-muted-foreground">
                actual \ predicted
              </th>
              {matrix.labels.map((l) => (
                <th key={l} scope="col" className="px-2 py-1 font-medium">
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.labels.map((row, i) => (
              <tr key={row}>
                <th scope="row" className="px-2 py-1 font-medium whitespace-nowrap">
                  {row}
                </th>
                {matrix.labels.map((col, j) => {
                  const n = matrix.counts[i * k + j];
                  const strength = n / max;
                  return (
                    <td
                      key={col}
                      className="px-2 py-1 text-center tabular-nums"
                      style={{
                        backgroundColor:
                          n === 0
                            ? undefined
                            : `color-mix(in srgb, var(--color-primary) ${Math.round(
                                strength * 45,
                              )}%, transparent)`,
                      }}
                    >
                      {n}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
