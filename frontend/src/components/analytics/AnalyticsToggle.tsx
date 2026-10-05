import { useState } from "react";

import { analyticsAvailable, isOptedOut, setOptedOut } from "@/analytics";

import { ANALYTICS_NOTE } from "./AnalyticsNote";

/**
 * The opt-out (#60). On by default only because the page says so — the note
 * sits beside the switch. The choice lives in its own localStorage key; opted
 * out, the SDK chunk is never requested at all.
 */
export function AnalyticsToggle() {
  const [enabled, setEnabled] = useState(() => !isOptedOut());
  if (!analyticsAvailable()) return null;

  return (
    <label
      data-testid="analytics-toggle"
      className="flex items-start gap-2 rounded-md border p-2 text-xs"
    >
      <input
        type="checkbox"
        role="switch"
        className="mt-0.5"
        checked={enabled}
        onChange={(e) => {
          setOptedOut(!e.target.checked);
          setEnabled(e.target.checked);
        }}
      />
      <span>
        <span className="font-medium">Anonymous usage analytics</span>
        <span className="block text-muted-foreground">{ANALYTICS_NOTE}</span>
      </span>
    </label>
  );
}
