# Plan: Frontend Stack Modernization (React 19, Base UI, ECharts/Recharts, MSW)

**Status:** Complete
**Date:** 2026-06-18
**Completed:** 2026-06-18

---

## Goal

Bring the frontend stack up to its current target: React 19, TypeScript ~6.0, Vite 8,
ESLint 10, shadcn/ui on **Base UI** (`base-nova` style, replacing Radix), **ECharts +
Recharts** for charts (replacing Plotly), **react-markdown** for LLM/Markdown output, and
**MSW** for network-level mocking in tests.

## Background

The frontend was React 18 + TS ~5.6 with shadcn/ui built on Radix (`radix-nova`), Plotly for
charts, and ESLint 9. The decoupled architecture (backend exposes `/api/`, frontend consumes
over HTTP) is unchanged — this is a tooling/library modernization only.

---

## Changes

### 1. Dependencies (`package.json`)

- React `18 → 19`, `@types/react(-dom)` `→ 19`, `@vitejs/plugin-react → ^5.2.0` (Vite 8 + React 19).
- TypeScript `~5.6 → ~6.0`.
- ESLint `9 → 10`; `typescript-eslint → ^8.61.1`, `eslint-plugin-react-hooks → ^7`,
  `eslint-plugin-react-refresh → ^0.5.3` (older majors don't declare ESLint 10 peer support).
- Removed `radix-ui`, `@types/plotly.js`, `plotly.js-dist-min`.
- Added `@base-ui/react`, `echarts`, `echarts-for-react`, `recharts`, `react-markdown`,
  `remark-gfm`, and `msw` (dev).
- Added `frontend/.npmrc` with `legacy-peer-deps=true` — Vite 8 is ahead of the peer ranges a
  few plugins (`@tailwindcss/vite`, `@vitejs/plugin-react`) currently declare.

### 2. shadcn/ui → Base UI (`base-nova`)

- `components.json` style `radix-nova → base-nova`.
- Rewrote `ui/button.tsx`, `ui/input.tsx`, `ui/label.tsx`, `ui/sheet.tsx` to import from
  `@base-ui/react/*`. `ui/card.tsx` had no Radix dependency and was left as-is.
- `ui/form.tsx`: `FormControl` now uses Base UI's `useRender` (no Radix `<Slot>`); `FormLabel`
  types against a native `<label>`.
- Replaced `<Button asChild>` usages with the Base UI `render` prop (`HeroBanner.tsx`, `Sheet`).

### 3. Charts: Plotly → ECharts + Recharts

- Deleted `components/charts/PlotlyChart.tsx` and `types/plotly-dist-min.d.ts`.
- Added lazy `components/charts/EChart.tsx` (wraps `echarts-for-react`; keep code-split).
- `routes/demo.chart.tsx` now demonstrates both an ECharts and a Recharts chart.

### 4. Markdown / LLM output

- Added `components/Markdown.tsx` (`react-markdown` + `remark-gfm`), styled per-element with
  Tailwind so it needs no typography plugin.

### 5. Testing: MSW + localStorage polyfill

- Added `src/test/handlers.ts` and `src/test/server.ts` (MSW node server).
- `src/test/setup.ts` starts MSW (`onUnhandledRequest: "bypass"`, so existing module-level Axios
  mocks still work) and installs an in-memory `localStorage`/`sessionStorage` polyfill — Node ≥25
  ships a stub global `localStorage` (`{}`) that shadows the DOM env's and breaks `localStorage.clear()`.

### 6. Misc fixes surfaced by the upgrade

- TS 6.0 deprecates `baseUrl`: removed from `tsconfig.app.json` (paths resolve under `bundler`).
- TS 6.0 stricter excess-property/cast checks fixed stale tests: dropped `confirm_password` from
  register test payloads (not in `RegisterPayload` / the API contract), widened a `useMe` mock cast,
  removed an unused `beforeEach` import.

---

## Testing

### Manual verification (all passing)

1. `npm install --legacy-peer-deps` — clean install.
2. `npm run build` (`tsc -b && vite build`) — type-checks and builds; ECharts is in its own lazy chunk.
3. `npm test` — **86 tests pass** (17 files).
4. `npm run lint` — **0 errors** (12 `react-refresh/only-export-components` warnings on TanStack
   Router route files are expected — those files export both `Route` and a component).

### Notes / risks

- **`react-hooks` v7** ships an expanded `recommended` set (e.g. `set-state-in-effect`). We kept the
  classic `rules-of-hooks` + `exhaustive-deps` rules to preserve prior lint behavior; adopting the
  new rules is a separate decision.
- **ECharts is heavy** (~1.1 MB) — always import `EChart` lazily so it stays out of the initial chunk.
- **Node ≥25 `localStorage` stub** — the test polyfill is required; the harmless
  `--localstorage-file ... without a valid path` warning during tests comes from Node, not the repo.
