import "@testing-library/jest-dom"
import { afterAll, afterEach, beforeAll } from "vitest"

import { useModelPrefs } from "@/store/models"

import { server } from "./server"

// Node 25 exposes a stub global `localStorage` ({}) that shadows the one
// happy-dom installs, so `localStorage.clear()` etc. throw in tests. Install a
// real in-memory Web Storage implementation to make storage deterministic.
class MemoryStorage implements Storage {
  private store = new Map<string, string>()
  get length() {
    return this.store.size
  }
  clear() {
    this.store.clear()
  }
  getItem(key: string) {
    return this.store.has(key) ? this.store.get(key)! : null
  }
  key(index: number) {
    return Array.from(this.store.keys())[index] ?? null
  }
  removeItem(key: string) {
    this.store.delete(key)
  }
  setItem(key: string, value: string) {
    this.store.set(key, String(value))
  }
}

for (const name of ["localStorage", "sessionStorage"] as const) {
  Object.defineProperty(globalThis, name, {
    value: new MemoryStorage(),
    configurable: true,
    writable: true,
  })
}

// Model-page preferences (selected model, resume intent) persist to
// localStorage, so without this a test that switches models leaks that choice
// into every later test in the file — the page would start on a model the test
// never chose.
afterEach(() => {
  localStorage.clear()
  useModelPrefs.setState({ selected: {}, autoResume: {} })
})

// Start the MSW mock server for the whole suite. Tests that mock `@/api/client`
// directly are unaffected because unhandled requests are bypassed.
beforeAll(() => server.listen({ onUnhandledRequest: "bypass" }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())
