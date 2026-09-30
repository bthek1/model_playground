// The static build (#57): `VITE_BACKEND=off` must mean *no request to /api*,
// not merely "no visible sign-in". On S3 + CloudFront a request to /api is
// rewritten to index.html and answered 200, so a stray call does not fail — it
// hands a JSON parser a page of HTML. The assertion is therefore on the wire:
// MSW sees every fetch and XHR the app makes, and a static build makes none.
//
// These mount the real route tree (not a page in isolation), because the
// redirect off `/` and the navbar's `useMe` both live outside any one page.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  createMemoryHistory,
  createRouter,
  RouterProvider,
} from "@tanstack/react-router";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { routeTree } from "@/routeTree.gen";
import { server } from "@/test/server";

// A getter, so each test can flip the flag the app reads at render time.
const flags = vi.hoisted(() => ({ backend: false }));
vi.mock("@/lib/features", () => ({
  get BACKEND_ENABLED() {
    return flags.backend;
  },
}));

const apiCalls: string[] = [];
function recordApi({ request }: { request: Request }) {
  const path = new URL(request.url).pathname;
  if (path.startsWith("/api/")) apiCalls.push(path);
}

function renderAt(path: string) {
  const router = createRouter({
    routeTree,
    history: createMemoryHistory({ initialEntries: [path] }),
  });
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  return router;
}

/** Let queries and effects fire, so an absent request is really absent. */
const settle = () => new Promise((r) => setTimeout(r, 50));

beforeEach(() => {
  apiCalls.length = 0;
  server.events.on("request:start", recordApi);
  // A stale token from an earlier build is exactly what would make useMe fire.
  localStorage.setItem("access_token", "stale.token");
});

afterEach(() => {
  server.events.removeListener("request:start", recordApi);
  localStorage.clear();
  cleanup();
});

describe("VITE_BACKEND=off", () => {
  beforeEach(() => {
    flags.backend = false;
  });

  it("sends / straight to /home, and asks /api nothing", async () => {
    const router = renderAt("/");
    await waitFor(() => expect(router.state.location.pathname).toBe("/home"));
    await screen.findByRole("heading", { name: "Model Playground", level: 1 });
    await settle();
    expect(apiCalls).toEqual([]);
  });

  it("renders /home with no registry catalogue and no /api call", async () => {
    renderAt("/home");
    await screen.findByRole("heading", { name: "Model Playground", level: 1 });
    await settle();
    expect(screen.queryByText("Model Catalog")).toBeNull();
    expect(apiCalls).toEqual([]);
  });

  it("renders /playground with no registry catalogue and no /api call", async () => {
    renderAt("/playground");
    await screen.findByRole("heading", { name: "Playground", level: 1 });
    await settle();
    expect(screen.queryByText("Model Catalog")).toBeNull();
    expect(apiCalls).toEqual([]);
  });

  it("offers no sign-in or sign-out in the navbar", async () => {
    renderAt("/home");
    await screen.findByRole("heading", { name: "Model Playground", level: 1 });
    await settle();
    expect(screen.queryByRole("link", { name: /sign in/i })).toBeNull();
    expect(screen.queryByRole("link", { name: /create account/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /sign out/i })).toBeNull();
  });

  it.each(["/login", "/signup"])("%s lands on /home instead", async (path) => {
    const router = renderAt(path);
    await waitFor(() => expect(router.state.location.pathname).toBe("/home"));
    await settle();
    expect(apiCalls).toEqual([]);
  });
});

describe("VITE_BACKEND on (the default)", () => {
  beforeEach(() => {
    flags.backend = true;
    server.use(
      http.get("*/api/accounts/me/", () =>
        HttpResponse.json({ detail: "expired" }, { status: 401 }),
      ),
    );
  });

  it("keeps the landing page and asks /api who is signed in", async () => {
    const router = renderAt("/");
    await screen.findByRole("link", { name: "Sign in" });
    await waitFor(() => expect(apiCalls).toContain("/api/accounts/me/"));
    expect(router.state.location.pathname).toBe("/");
  });

  it("renders the registry catalogue on /home", async () => {
    renderAt("/home");
    await screen.findByText("Model Catalog");
    await waitFor(() => expect(apiCalls).toContain("/api/registry/models/"));
  });
});
