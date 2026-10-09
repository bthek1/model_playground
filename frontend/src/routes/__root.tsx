import {
  createRootRoute,
  Outlet,
  useRouterState,
} from "@tanstack/react-router";
import { usePageviews } from "@/analytics/usePageviews";
import { AppLayout } from "@/components/layout/AppLayout";
import { NotFoundPage } from "@/components/layout/NotFoundPage";
import { useDocumentHead } from "@/hooks/useDocumentHead";

// These paths render without the app shell (no navbar/sidebar)
const PUBLIC_PATHS = ["/", "/login", "/signup"];

export const Route = createRootRoute({
  component: RootComponent,
  // Rendered in the root's <Outlet />, so inside the app shell (#63).
  notFoundComponent: NotFoundPage,
});

function RootComponent() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const isPublic = PUBLIC_PATHS.includes(pathname);

  useDocumentHead();
  usePageviews();

  if (isPublic) return <Outlet />;
  return <AppLayout />;
}
