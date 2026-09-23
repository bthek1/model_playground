// Renders `highlight()`'s slices as the user's own text with the model's spans
// marked in place. Shared by `/token-classification`, `/question-answering` and
// `/fill-mask`.
//
// **Every span carries its type as visible text, not as colour alone**, and that
// is a correctness decision rather than a stylistic one. The four `--entity-*`
// hues are validated all-pairs in both themes, but their worst colour-vision
// separation sits in the band that is only legal *with* a secondary encoding —
// a direct label. It is also the better NER surface: a page where you must
// hover to learn whether something was tagged PER or ORG is answering the
// question one entity at a time.
//
// Colours come from theme tokens (`--entity-1…4`), never hex in this file, and
// the text itself stays in normal ink — a mark carries identity, text does not
// wear the mark's colour.

import { entitySlot, type HighlightResult, type Slice } from "@/text/highlight";
import { cn } from "@/lib/utils";

/** Tailwind classes per entity slot. Static strings so the JIT can see them. */
const SLOT: Record<number, string> = {
  1: "bg-entity-1/15 border-entity-1 text-entity-1",
  2: "bg-entity-2/15 border-entity-2 text-entity-2",
  3: "bg-entity-3/15 border-entity-3 text-entity-3",
  4: "bg-entity-4/15 border-entity-4 text-entity-4",
};
const UNKNOWN = "bg-muted border-muted-foreground/40 text-muted-foreground";

export function SpanOverlay({
  result,
  /** Types to render as redacted rather than highlighted. */
  redacted,
  /** Hide the little type tag — for a single-span page that says it elsewhere. */
  showLabels = true,
  className,
  "data-testid": testId = "span-overlay",
}: {
  result: HighlightResult;
  redacted?: ReadonlySet<string>;
  showLabels?: boolean;
  className?: string;
  "data-testid"?: string;
}) {
  return (
    <div className={cn("space-y-2", className)} data-testid={testId}>
      <p className="text-sm leading-8 whitespace-pre-wrap">
        {result.slices.map((slice, i) => (
          <SliceMark
            key={i}
            slice={slice}
            redacted={
              slice.span != null && (redacted?.has(slice.span.label) ?? false)
            }
            showLabel={showLabels}
          />
        ))}
      </p>

      {result.dropped.length > 0 && (
        // Never silent: two spans claiming the same characters cannot both be
        // drawn, and choosing one quietly shows a confident highlight over a
        // range no model proposed.
        <p
          data-testid="span-dropped"
          className="text-xs text-amber-600 dark:text-amber-500"
        >
          {result.dropped.length} span
          {result.dropped.length === 1 ? "" : "s"} could not be drawn — they
          overlap a span already placed, or fall outside the text.
        </p>
      )}
    </div>
  );
}

function SliceMark({
  slice,
  redacted,
  showLabel,
}: {
  slice: Slice;
  redacted: boolean;
  showLabel: boolean;
}) {
  const span = slice.span;
  if (!span) return <>{slice.text}</>;

  const slot = entitySlot(span.label);
  const tone = slot == null ? UNKNOWN : SLOT[slot];

  if (redacted) {
    return (
      <mark
        data-testid="span-redacted"
        data-label={span.label}
        className="mx-0.5 rounded border border-dashed border-muted-foreground/50 bg-muted px-1 font-mono text-xs text-muted-foreground"
      >
        [{span.label}]
      </mark>
    );
  }

  return (
    // `tabIndex` so the score is reachable without a pointer: the type is
    // already on screen, but the confidence is the thing a keyboard user would
    // otherwise never see.
    <mark
      data-testid="span-mark"
      data-label={span.label}
      tabIndex={0}
      title={`${span.label} · ${span.score.toFixed(2)}`}
      className={cn(
        "mx-0.5 rounded border-b-2 bg-transparent px-0.5 text-foreground",
        "focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
        tone,
      )}
    >
      {/* The entity text gets its own hook: the type tag beside it is an
          inline sibling, so reading the mark's own `innerText` returns
          "Priya Raman PER" and an offset assertion against it is vacuous. */}
      <span data-testid="span-text" className="text-foreground">
        {slice.text}
      </span>
      {showLabel && (
        <span className="ml-1 align-middle text-[0.6rem] font-semibold tracking-wide uppercase">
          {span.label}
        </span>
      )}
    </mark>
  );
}
