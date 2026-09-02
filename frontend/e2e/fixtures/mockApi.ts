import type { Page, Route } from "@playwright/test";

import { mockModels } from "../../src/test/fixtures/models";
import type { User } from "../../src/types/auth";

/**
 * Stub the Django API at the network layer so specs run with no backend and no
 * Postgres. The app always calls /api on its own origin (the Vite dev server
 * proxies it to Django), so one `**\/api/**` route catches every call.
 *
 * Endpoint paths mirror src/api/{auth,models,health}.ts — keep them in step.
 */
export interface MockApiOptions {
  /** Registry models returned by GET /api/registry/models/. */
  models?: typeof mockModels;
  /** Make the registry endpoint fail, to exercise the error state. */
  modelsError?: boolean;
  /** Treat the user as signed in (GET /api/accounts/me/ returns a user). */
  authenticated?: boolean;
}

export const MOCK_USER: User = {
  id: "33333333-3333-4333-8333-333333333333",
  email: "e2e@example.com",
  first_name: "Eee",
  last_name: "Tootoo",
  date_joined: "2026-01-01T00:00:00Z",
};

export async function installMockApi(
  page: Page,
  options: MockApiOptions = {},
): Promise<void> {
  const {
    models = mockModels,
    modelsError = false,
    authenticated = false,
  } = options;

  const json = (route: Route, body: unknown, status = 200) =>
    route.fulfill({
      status,
      contentType: "application/json",
      body: JSON.stringify(body),
    });

  // Match on the pathname, NOT a `**\/api/**` glob: that glob also matches the
  // dev server's own module URLs (e.g. /src/api/client.ts), which would replace
  // the app's API client with JSON and stop it booting at all.
  await page.route(
    (url) => url.pathname.startsWith("/api/"),
    async (route) => {
      const path = new URL(route.request().url()).pathname;

      if (path.startsWith("/api/registry/models/")) {
        if (modelsError) return json(route, { detail: "boom" }, 500);
        return json(route, models);
      }
      if (path.startsWith("/api/registry/runs/")) {
        return json(route, { id: "run-1", status: "completed" }, 201);
      }
      if (path === "/api/accounts/me/") {
        return authenticated
          ? json(route, MOCK_USER)
          : json(route, { detail: "Unauthorized" }, 401);
      }
      if (path === "/api/accounts/register/") {
        return json(route, MOCK_USER, 201);
      }
      if (path === "/api/token/" || path === "/api/token/refresh/") {
        return json(route, { access: "mock.access", refresh: "mock.refresh" });
      }
      if (path === "/api/health/") {
        return json(route, { status: "ok" });
      }
      // Anything unmodelled fails loudly rather than silently reaching a real
      // server, so a newly added endpoint surfaces as an obvious test failure.
      return json(route, { detail: `Unmocked API route: ${path}` }, 501);
    },
  );
}

/** Seed JWTs into localStorage before the app boots, so it starts signed in. */
export async function seedTokens(page: Page): Promise<void> {
  await page.addInitScript(() => {
    localStorage.setItem("access_token", "mock.access");
    localStorage.setItem("refresh_token", "mock.refresh");
  });
}
