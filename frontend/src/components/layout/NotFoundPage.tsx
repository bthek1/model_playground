// The app's answer to a URL it has no route for (#63). The edge serves the app
// shell for any unknown path with a 200 — CloudFront cannot know the router's
// routes — so the status cannot say "not found" and this page has to: a
// `noindex` while it is mounted, which Google honours when it renders the page,
// and a way back into the site rather than a dead end.

import { Link, useRouterState } from "@tanstack/react-router";
import { useEffect } from "react";

import { Button } from "@/components/ui/button";

export function NotFoundPage() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  useEffect(() => {
    const meta = document.createElement("meta");
    meta.name = "robots";
    meta.content = "noindex";
    document.head.appendChild(meta);
    return () => meta.remove();
  }, []);

  return (
    <div
      data-testid="not-found"
      className="mx-auto flex max-w-xl flex-col items-start gap-4 py-12"
    >
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <p className="text-muted-foreground">
        There is no page at <code className="break-all">{pathname}</code>. It
        may have been renamed, or removed when a model turned out too large to
        run in a browser.
      </p>
      <Button render={<Link to="/home">Browse every task</Link>} />
    </div>
  );
}
