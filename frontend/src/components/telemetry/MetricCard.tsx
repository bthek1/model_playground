// The shell every panel card shares.
//
// Two things are structural rather than decorative. `note` is the one line that
// says what the number *is* — load or utilisation, whose memory, which realm —
// because a figure in a card labelled "GPU" will otherwise be read as the
// GPU's. And `unavailable` replaces the body entirely: a browser that cannot
// make a measurement gets the reason it cannot, never a zero.

import type { LucideIcon } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export interface MetricCardProps {
  title: string;
  icon: LucideIcon;
  /** What is being measured, and what it is not. One sentence. */
  note: string;
  /** Why there is no measurement. When set, the body is not rendered. */
  unavailable?: string | null;
  testId: string;
  children?: React.ReactNode;
}

export function MetricCard({
  title,
  icon: Icon,
  note,
  unavailable,
  testId,
  children,
}: MetricCardProps) {
  return (
    <Card size="sm" data-testid={testId} className="gap-2">
      <CardHeader className="flex flex-row items-center gap-2">
        <Icon className="size-4 shrink-0 text-muted-foreground" aria-hidden />
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {unavailable ? (
          <p
            data-testid={`${testId}-unavailable`}
            className="text-xs leading-snug text-muted-foreground"
          >
            {unavailable}
          </p>
        ) : (
          <>
            {children}
            <p className="text-[11px] leading-snug text-muted-foreground/80">
              {note}
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}

/** The card's headline figure. */
export function Readout({
  value,
  of,
}: {
  value: string;
  /** The denominator — a capacity, a quota, a ceiling. */
  of?: string;
}) {
  return (
    <p className="font-mono text-xl leading-none">
      {value}
      {of && (
        <span className="ml-1 font-sans text-xs text-muted-foreground">
          / {of}
        </span>
      )}
    </p>
  );
}

/** A `label … value` line for the secondary facts. */
export function Row({
  label,
  value,
  title,
}: {
  label: string;
  value: React.ReactNode;
  title?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-2 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="truncate font-mono" title={title}>
        {value}
      </span>
    </div>
  );
}

/** A proportion bar. `fraction` is clamped, so a bad ratio cannot overflow. */
export function Meter({
  fraction,
  label,
  token = "--chart-1",
}: {
  fraction: number;
  label: string;
  token?: string;
}) {
  const percent = Math.min(100, Math.max(0, fraction * 100));
  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuenow={Math.round(percent)}
      aria-valuemin={0}
      aria-valuemax={100}
      className="h-1.5 w-full overflow-hidden rounded-full bg-muted"
    >
      <div
        className="h-full rounded-full"
        style={{ width: `${percent}%`, background: `var(${token})` }}
      />
    </div>
  );
}
