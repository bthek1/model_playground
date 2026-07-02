import { http, HttpResponse } from "msw"

const API = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8005"

/**
 * Default request handlers used by every test. Override per-test with
 * `server.use(...)` from `@/test/server` to simulate errors or edge cases.
 */
export const handlers = [
  http.get(`${API}/api/health/`, () =>
    HttpResponse.json({ status: "ok" })
  ),
]
