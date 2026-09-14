import { useEffect } from "react";
import { useRouterState } from "@tanstack/react-router";

import { titleForPath } from "@/lib/documentTitle";

/**
 * Keeps `document.title` in step with the current route. Mounted once, in the
 * root route, so it covers every page including the ones that render without
 * the app shell.
 */
export function useDocumentTitle() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  useEffect(() => {
    document.title = titleForPath(pathname);
  }, [pathname]);
}
