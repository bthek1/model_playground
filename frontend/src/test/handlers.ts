import { http, HttpResponse } from "msw"

const API = import.meta.env.VITE_API_BASE_URL ?? ""

/**
 * Default request handlers used by every test. Override per-test with
 * `server.use(...)` from `@/test/server` to simulate errors or edge cases.
 *
 * Shared response bodies live in `./fixtures/models`; the Playwright mock-API
 * fixture imports them from there too, so the two suites can't drift apart.
 */
export const handlers = [
  http.get(`${API}/api/health/`, () =>
    HttpResponse.json({ status: "ok" })
  ),
  http.get(`${API}/api/registry/models/`, () => HttpResponse.json([])),
]
