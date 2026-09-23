// The NLP category's one shared output component: a sorted bar per label with
// the score printed beside it.
//
// **It refuses to render a single row, and that is the whole design.** A
// classifier's argmax is the least informative thing it produces: "POSITIVE"
// alone looks identical at 0.99 and at 0.51, and the second is the case the user
// most needs to see. So every caller passes the full label set (`top_k`), the
// near-tie is called out in words rather than left to be read off two bar
// widths, and a one-class result says so instead of being dressed up as a
// confident answer.
//
// Colour comes from theme tokens, never hex (docs/standards/model-visualization.md
// §2), and the bars are CSS widths rather than a chart library — `echarts` is
// lazy-loaded and heavy, and this is a list.

import { cn } from "@/lib/utils";
import type { ClassLabel } from "@/model/types";

/** Below this gap between the top two, the model is not picking — it is guessing. */
export const NEAR_TIE = 0.1;

export function ScoreList({
  scores,
  /** Small caption under the bars — usually what the scores are *of*. */
  caption,
  /** Silence the near-tie note where the page makes the point itself. */
  quiet = false,
  className,
  "data-testid": testId = "score-list",
}: {
  scores: readonly ClassLabel[];
  caption?: React.ReactNode;
  quiet?: boolean;
  className?: string;
  "data-testid"?: string;
}) {
  if (scores.length === 0) return null;

  // Sorted here rather than trusted from the caller: `top_k` returns ranked
  // output, but a page that merges two models' lists or filters one does not.
  const ranked = [...scores].sort((a, b) => b.score - a.score);
  const margin = ranked.length > 1 ? ranked[0].score - ranked[1].score : null;

  return (
    <div className={cn("space-y-3", className)} data-testid={testId}>
      <ul className="space-y-2">
        {ranked.map((s) => (
          <ScoreRow key={s.label} label={s.label} score={s.score} />
        ))}
      </ul>

      {!quiet && margin != null && margin < NEAR_TIE && (
        <p
          data-testid="score-near-tie"
          className="text-xs text-amber-600 dark:text-amber-500"
        >
          The top two are within {margin.toFixed(2)} of each other — this model
          is not confident, whatever the first row says.
        </p>
      )}

      {!quiet && ranked.length === 1 && (
        <p
          data-testid="score-single"
          className="text-xs text-muted-foreground"
        >
          Only one label came back, so there is nothing to compare it against.
        </p>
      )}

      {caption && (
        <p className="text-xs leading-snug text-muted-foreground">{caption}</p>
      )}
    </div>
  );
}

function ScoreRow({ label, score }: ClassLabel) {
  const pct = Math.round(score * 100);
  return (
    <li className="space-y-1">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="truncate">{label}</span>
        <span className="shrink-0 font-mono text-xs text-muted-foreground tabular-nums">
          {score.toFixed(2)}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary"
          // A 2% floor so a near-zero score is still visibly a row rather than
          // an empty track that reads as a rendering failure.
          style={{ width: `${Math.max(2, pct)}%` }}
        />
      </div>
    </li>
  );
}
