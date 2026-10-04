import { useEffect } from "react";
import { useRouterState } from "@tanstack/react-router";

import { pageview } from "./index";

/**
 * One `$pageview` per navigation, keyed on the matched route's **id** — the
 * pattern (`/text-classification`), never `location.href`, so a query string or
 * hash can never carry input into an event (#60).
 */
export function usePageviews(): void {
  const routeId = useRouterState({
    select: (s) => s.matches[s.matches.length - 1]?.routeId,
  });
  useEffect(() => {
    if (routeId) pageview(routeId);
  }, [routeId]);
}
