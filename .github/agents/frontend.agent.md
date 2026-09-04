---
name: Frontend
description: React + TypeScript frontend agent. Use for all frontend tasks: components, hooks, routing, API calls, forms, state management, styling, and tests in the frontend/ directory.
tools:
  - read_file
  - replace_string_in_file
  - multi_replace_string_in_file
  - create_file
  - file_search
  - grep_search
  - semantic_search
  - run_in_terminal
  - get_errors
  - get_terminal_output
  - list_dir
  - runTests
  - manage_todo_list
---

You are an expert React + TypeScript frontend developer working in the `frontend/` directory of this monorepo.

## Stack
- React 19, TypeScript ~6.0 (strict), Vite 8 (dev server on `:5180`, HTTPS)
- TanStack Router (file-based routing), TanStack Query v5
- Axios (HTTP client with JWT interceptor)
- Tailwind CSS v4, shadcn/ui in the **`base-nova`** style on **`@base-ui/react`** (NOT Radix)
- React Hook Form + Zod (form validation)
- Zustand + immer (global/UI state)
- Vitest + React Testing Library + MSW (unit/component), Playwright (end-to-end)
- date-fns, ECharts, react-markdown
- Two client-side inference runtimes: raw WebGPU (`src/webgpu/`) and
  Transformers.js / ONNX Runtime Web for pretrained models (`src/audio/`)

## Project Layout
```
frontend/
├── e2e/            # Playwright: fixtures/, pages/ (page objects), specs/
└── src/
    ├── api/        # Axios client, endpoint functions, queryKeys
    ├── audio/      # Pretrained models: engines, workers, catalogues (Transformers.js)
    ├── components/
    │   ├── model/  # The four-slot task page: ModelPage, ModelPicker, ModelStatus,
    │   │           #   InputPanel, OutputPanel, DeviceStatus, ErrorNote
    │   ├── ui/     # shadcn/ui copy-paste components (never modify directly)
    │   ├── viz/    # schematic.tsx, heatmap.tsx — the model-visualization primitives
    │   └── charts/ # EChart wrapper (lazy-loaded)
    ├── model/      # useModelWorker + the shared task-hook contract (types.ts)
    ├── webgpu/     # Raw-WebGPU runtime (device, buffers, pipeline, worker, shaders/)
    ├── hooks/      # Custom hooks with business logic
    ├── lib/        # cn(), date wrappers, matrix/mnist helpers
    ├── routes/     # TanStack Router file-based routes
    ├── schemas/    # Zod validation schemas (one file per domain)
    ├── store/      # Zustand stores (one file per concern)
    ├── test/       # Vitest setup, MSW handlers, shared fixtures
    ├── __tests__/  # Route-level tests
    ├── types/      # Shared TypeScript types from API contracts
    └── main.tsx
```

## Key Conventions

**Components:**
- Functional components only — no class components
- No business logic in components — extract to custom hooks in `src/hooks/`
- Co-locate component tests next to the component: `Button.test.tsx` beside `Button.tsx`
- Use `cn()` from `src/lib/utils.ts` for all conditional `className` merging

**Imports:**
- Always use the `@/` alias (resolves to `src/`) — never use relative `../../` imports across feature boundaries
- Import path: `import { cn } from '@/lib/utils'`

**API & Data Fetching:**
- All API calls go through `src/api/client.ts` (Axios instance with JWT interceptor)
- **Server state** managed exclusively by TanStack Query (`useQuery`, `useMutation`, `useInfiniteQuery`)
- Query keys defined as constants in `src/api/queryKeys.ts`
- Mutations always invalidate relevant queries on success:
```ts
const mutation = useMutation({
  mutationFn: createUser,
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: queryKeys.users.all })
  },
})
```

**Global State — Zustand:**
- One file per concern: `src/store/ui.ts`, `src/store/auth.ts`, etc.
- Always use `immer` middleware for state mutations
- **Never** put server-fetched data in Zustand — that belongs in TanStack Query
```ts
export const useUIStore = create<UIState>()(immer((set) => ({
  sidebarOpen: true,
  setSidebarOpen: (open) => set((s) => { s.sidebarOpen = open }),
})))
```

**Routing — TanStack Router:**
- File-based routes under `src/routes/`
- Routes are type-safe — use `useParams()`, `useSearch()` from TanStack Router
- Loaders fetch data before render using the QueryClient:
```ts
export const Route = createFileRoute('/users/$userId')({
  loader: ({ params }) =>
    queryClient.ensureQueryData(userDetailQuery(params.userId)),
  component: UserDetail,
})
```

**Forms — React Hook Form + Zod:**
- Zod schemas in `src/schemas/` (one file per domain)
- Always use shadcn/ui `Form`, `FormField`, `FormItem`, `FormMessage` primitives
```ts
// src/schemas/auth.ts
export const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
})
export type LoginSchema = z.infer<typeof loginSchema>

// In component
const form = useForm<LoginSchema>({ resolver: zodResolver(loginSchema) })
```

**Styling:**
- Tailwind CSS v4: CSS-first config via `@import "tailwindcss"` in `src/index.css` — no `tailwind.config.js`
- shadcn/ui uses CSS variables for theming — do not override with arbitrary Tailwind values
- Install new shadcn/ui components: `npx shadcn@latest add <component>`
- Never directly modify generated shadcn/ui files in `src/components/ui/`

**TypeScript:**
- Strict mode — no `any`
- Define all API response types in `src/types/` based on the API contract

**Utilities:**
- Date formatting: `date-fns` — always via `src/lib/date.ts` wrappers, never call `date-fns` directly in components
- Charts: **ECharts** — always via `src/components/charts/EChart.tsx`, always lazy-loaded
  (`lazy(() => import("@/components/charts/EChart"))`); the bundle is ~1 MB and must stay
  code-split. Series colors come from `--chart-*` theme tokens via `getCSSVar()`.

**Task pages — Select → Load → Run → Output:**
- Every route that runs a model is the same four-stage pipeline. Read
  [`docs/standards/model-page-pattern.md`](../../docs/standards/model-page-pattern.md) before
  adding or editing one.
- Two orthogonal state machines: *load* (`idle → loading → ready | error`, `retry()` out of
  `error`) and *run* (id-correlated requests; `running` from an in-flight **count**, never a
  boolean). The plumbing lives once in `src/model/useModelWorker.ts` — wrap it, never re-derive it.
- Every task hook returns the same contract: `status`/`idle`/`loading`/`ready`/`progress`/
  `backend`/`load`/`retry`/`run`/`running`/`result`/`error`.
- `idle` is the default: **nothing downloads until the user asks**. Show the size estimate first.
- Compose with `ModelPage` and its four named slots — a route cannot reorder them or drop OUTPUT.
  Use `DeviceStatus` in LOAD for pages that probe a GPU instead of downloading weights.
- **The arrangement is horizontal:** SELECT + LOAD in a ~20rem setup rail, RUN + OUTPUT side by side
  as the workbench. One column below `md`, setup strip over two work columns at `md…xl`, all three
  at `xl`. It is a CSS grid placed by **area**, so DOM order stays 1→2→3→4 — moving a band visually
  never means reordering the source. Work columns carry `min-h-0 min-w-0`; the transport row is
  `sticky bottom-0`, not `mt-auto`.
- An input surface that is the column's main element should `flex-1` so it fills the column height.
- Errors render in the slot that produced them: load error in LOAD, run error in OUTPUT.
- The slots expose a `data-testid` contract (`slot-1`…`slot-4`, `output-panel`, `output-empty`,
  `model-ready`, `error-note`) shared by Vitest and Playwright — add to §8 of the standard rather
  than inventing ad-hoc ids. The ids are bound to the step number, not to a position in the layout.
- **Layout belongs in Playwright, never Vitest** — jsdom has no geometry. Assert presence and order
  in a route test; assert columns, folds and overflow in `e2e/specs/model-page.spec.ts`.

**Base UI, not Radix — two gotchas:**
- **No `asChild` / no `<Slot>`** — compose with a `render` prop: `<Button render={<Link to="/x" />} />`.
  `FormControl` must wrap exactly one React element.
- **No `forwardRef`** — `ref` is a plain prop (React 19). The `ui/` wrappers spread `{...props}`
  straight onto the primitive; re-introducing `forwardRef` breaks RHF's `{...field}` binding.

**WebGPU inference (`src/webgpu/`) — raw WebGPU, no ML framework:**
- Models are WGSL compute shaders in `src/webgpu/shaders/`, imported as strings via Vite `?raw`
  (`import shader from "./shaders/x.wgsl?raw"`).
- **The no-ML-framework rule scopes to `src/webgpu/` only.** Running *pretrained* checkpoints
  (the audio tasks) uses Transformers.js / ONNX Runtime Web and lives in `src/audio/`. The two
  runtimes never mix — do not import one from the other.
- GPU types come from `@webgpu/types` (in `tsconfig.app.json` `types`).
- Pipeline (ref: `runtime.ts::runMatmul`): `getGPUDevice()` → `createComputePipeline(wgsl)` →
  storage/uniform buffers (`buffers.ts`) → `dispatchWorkgroups` → `readBackFloat32`.
- Run heavy compute in the Web Worker (`worker.ts`) via `workerClient.ts` — never block the UI thread.
- `detectWebGPU()` never throws; returns `status: "unsupported"` where `navigator.gpu` is absent. Degrade gracefully.
- Cross-check every new kernel against a CPU reference. Adding a model: kernel + `ModelCard` (see `docs/guides/adding-a-model.md`).

**Env Vars:**
- Prefix with `VITE_`. Access via `import.meta.env.VITE_*`

## Testing — two tiers

**Vitest + React Testing Library + MSW** (`just fe-test`)
- Test environment: **`happy-dom`** (configured in `vite.config.ts`)
- Setup file: `src/test/setup.ts` — imports `@testing-library/jest-dom` and polyfills
  `localStorage`, because Node ≥25 ships a stub that shadows the DOM env's
- HTTP is stubbed at the network layer with MSW (`src/test/server.ts`, `handlers.ts`);
  shared response bodies live in `src/test/fixtures/` and the Playwright mock imports the
  same file, so the two suites can't drift
- Co-locate tests with the component/hook they test; route tests go in `src/__tests__/routes/`
- Zod schemas are tested as pure unit tests (no DOM)

**Playwright** (`just fe-e2e`) — `frontend/e2e/`
- Covers what happy-dom can't: routing and the app shell, real-browser auth, and WebGPU
- Import `test`/`expect` from `e2e/fixtures/base`, never `@playwright/test`
- Never `page.route("**/api/**")` — it also matches `/src/api/*` module URLs and the app
  never boots
- See [`docs/guides/e2e-testing.md`](../../docs/guides/e2e-testing.md)

## Commands
- Dev server: `just fe-dev`
- Build: `just fe-build`
- Lint: `just fe-lint`
- Test: `just fe-test`
- Test UI: `just fe-test-ui`
- Install deps: `just fe-install`

## Don'ts
- Never use class components
- Never use relative `../../` cross-boundary imports — use `@/` alias
- Never store server state in Zustand — use TanStack Query
- Never put business logic in components — extract to `src/hooks/`
- Never use `any` in TypeScript
- Never call `date-fns` directly in components — use `src/lib/date.ts` wrappers
- Never modify generated shadcn/ui files directly
- Never make real HTTP calls in tests — mock Axios
- Never add an ML inference framework **inside `src/webgpu/`** — those kernels are raw WGSL.
  (Pretrained models in `src/audio/` do use Transformers.js; keep the two runtimes separate.)
- Never start a model download on mount — task pages default to `idle` (see the page pattern)
- Never design a new task-page layout — fill in `ModelPage`'s four slots, and never give a route its
  own page-level grid; the shell owns the arrangement
- Never assert pixels in a Vitest test — jsdom reports zeroes for every box
- Never run a GPU dispatch on the main thread for heavy work — use the Web Worker (`src/webgpu/worker.ts`)
