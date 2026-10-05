import { analyticsAvailable } from "@/analytics";
import { cn } from "@/lib/utils";

/** The disclosure, word for word — tests and the E2E spec pin this string. */
export const ANALYTICS_NOTE =
  "Anonymous counts of which pages and models are used — never your inputs or results.";

/**
 * Said where the privacy claims already live (#60). Rendered only in a build
 * that ships analytics: a build without a key sends nothing, and a sentence
 * describing what it sends would be false there.
 */
export function AnalyticsNote({ className }: { className?: string }) {
  if (!analyticsAvailable()) return null;
  return (
    <span data-testid="analytics-note" className={cn("text-muted-foreground", className)}>
      {ANALYTICS_NOTE} You can turn it off in the system panel.
    </span>
  );
}
