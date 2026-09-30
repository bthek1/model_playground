import { expect, test } from "../../fixtures/base";

// Phase 0 of #51, kept as a spec so the table in `src/rl/limits.ts` can be
// re-measured rather than trusted: **does an RL step belong on the GPU?**
//
// Logged, not asserted — a timing depends on the machine, and a threshold here
// would be a flaky test about hardware rather than a test about the code. The
// numbers it prints are what `limits.ts` records; `just fe-e2e-rl-phase0` runs
// it alone.
//
// It imports the app's own modules through the dev server (`/src/...`), so it
// measures the code that ships, in the realm that runs it, rather than a
// re-implementation inside `page.evaluate`.

test.describe("@slow @devserver RL Phase 0", () => {
  test("steps per second, CPU against the GPU matmul, and the cost of posting per step", async ({
    page,
  }) => {
    test.setTimeout(600_000);
    await page.goto("/");

    const table = await page.evaluate(async () => {
      type Mod = Record<string, unknown>;
      // Paths the dev server resolves, held in a variable so the spec's own
      // type-check does not try to (it has no `/src` root).
      const load = (path: string) => import(/* @vite-ignore */ path) as Promise<Mod>;
      const net = await load("/src/rl/policyNet.ts");
      const rt = await load("/src/webgpu/runtime.ts");
      const q = await load("/src/rl/qLearning.ts");
      const caps = await load("/src/webgpu/capabilities.ts");

      type Net = { w1: Float32Array; b1: Float32Array; w2: Float32Array; b2: Float32Array; inputs: number; hidden: number; outputs: number };
      const initNet = net.initNet as (s: object, seed: number) => Net;
      const forward = net.forward as (p: Net, x: Float32Array, n: number) => { h: Float32Array };
      const backward = net.backward as (p: Net, c: object, d: Float32Array) => object;
      const transpose = net.transpose as (a: Float32Array, r: number, c: number) => Float32Array;
      const runMatmul = rt.runMatmul as (job: object) => Promise<{ data: Float32Array }>;
      const QLearningRun = q.QLearningRun as new (req: object) => { done: boolean; step(): number | null; steps: number };
      const detect = caps.detectWebGPU as () => Promise<{ status: string; adapter?: { vendor: string; architecture: string } }>;

      const gpu = await detect();
      const hasGpu = gpu.status === "ready";
      const adapter = gpu.adapter ? `${gpu.adapter.vendor} ${gpu.adapter.architecture}` : gpu.status;

      /** Run `fn` repeatedly for ~`ms` and return calls per second. */
      async function rate(fn: () => unknown, ms = 400): Promise<number> {
        for (let i = 0; i < 5; i++) await fn();
        let n = 0;
        const t0 = performance.now();
        while (performance.now() - t0 < ms) {
          await fn();
          n++;
        }
        return (n * 1000) / (performance.now() - t0);
      }

      const gmm = async (a: Float32Array, b: Float32Array, m: number, k: number, n: number) =>
        (await runMatmul({ a, b, m, k, n })).data;

      /** The same two-layer forward, with its matmuls sent to the device. */
      async function gpuForward(p: Net, x: Float32Array, n: number) {
        const h = await gmm(x, p.w1, n, p.inputs, p.hidden);
        for (let i = 0; i < h.length; i++) h[i] = Math.tanh(h[i]);
        const out = await gmm(h, p.w2, n, p.hidden, p.outputs);
        return { h, out };
      }
      async function gpuBackward(p: Net, x: Float32Array, h: Float32Array, d: Float32Array, n: number) {
        await gmm(transpose(h, n, p.hidden), d, p.hidden, n, p.outputs);
        const dh = await gmm(d, transpose(p.w2, p.hidden, p.outputs), n, p.outputs, p.hidden);
        await gmm(transpose(x, n, p.inputs), dh, p.inputs, n, p.hidden);
      }

      const rows: Record<string, string | number>[] = [];

      // The tabular rung: a whole Q-learning run, steps per second.
      {
        const run = new QLearningRun({ algorithm: "q-learning", map: "8x8", slippery: true, alpha: 0.1, gamma: 0.99, epsilon: 1, decay: true, episodes: 20000, seed: 1 });
        const t0 = performance.now();
        while (!run.done) run.step();
        rows.push({ what: "Q-learning step (8x8, table)", cpu: Math.round((run.steps * 1000) / (performance.now() - t0)), gpu: "—" });
      }

      // The minimal device round trip: a 1x1 matmul, write → dispatch → read.
      if (hasGpu) {
        const one = new Float32Array([1]);
        rows.push({ what: "1x1 matmul round trip", cpu: "—", gpu: Math.round(await rate(() => gmm(one, one, 1, 1, 1))) });
      }

      for (const hidden of [32, 128, 512]) {
        const p = initNet({ inputs: 4, hidden, outputs: 2 }, 1);
        const x1 = new Float32Array([0.01, -0.2, 0.03, 0.1]);
        // Per step: pick an action — one forward at batch 1.
        const cpuAct = await rate(() => forward(p, x1, 1));
        const gpuAct = hasGpu ? await rate(() => gpuForward(p, x1, 1)) : NaN;
        rows.push({ what: `act, 4→${hidden}→2, batch 1`, cpu: Math.round(cpuAct), gpu: Math.round(gpuAct) });

        // Per episode: one update over a 200-step episode — the GPU's best case.
        const T = 200;
        const xb = new Float32Array(T * 4).map((_, i) => Math.sin(i));
        const d = new Float32Array(T * 2).map((_, i) => Math.cos(i) * 0.01);
        const cpuUpd = await rate(() => backward(p, forward(p, xb, T), d), 300);
        const gpuUpd = hasGpu
          ? await rate(async () => {
              const f = await gpuForward(p, xb, T);
              await gpuBackward(p, xb, f.h, d, T);
            }, 300)
          : NaN;
        rows.push({ what: `update, 4→${hidden}→2, batch ${T}`, cpu: Math.round(cpuUpd), gpu: Math.round(gpuUpd) });
      }

      // The message cost: one run posting per step against the same run at 60 Hz.
      async function workerRun(renderIntervalMs?: number): Promise<{ ms: number; posts: number; steps: number }> {
        const worker = new Worker("/src/rl/rl.worker.ts", { type: "module" });
        let posts = 0;
        const t0 = performance.now();
        const result = await new Promise<{ steps: number }>((resolve, reject) => {
          worker.onerror = (e) => reject(new Error(e.message));
          worker.onmessage = (e: MessageEvent<{ event?: string; ok?: boolean; result?: { steps: number }; error?: string }>) => {
            if (e.data.event === "progress") posts++;
            else if (e.data.ok && e.data.result) resolve(e.data.result);
            else reject(new Error(e.data.error ?? "worker failed"));
          };
          worker.postMessage({
            type: "train",
            id: 1,
            req: { algorithm: "q-learning", map: "8x8", slippery: true, alpha: 0.1, gamma: 0.99, epsilon: 1, decay: true, episodes: 3000, seed: 1 },
            speed: null,
            renderIntervalMs,
          });
        });
        const ms = performance.now() - t0;
        worker.terminate();
        return { ms, posts, steps: result.steps };
      }
      const throttled = await workerRun();
      const perStep = await workerRun(0);
      rows.push({ what: "worker run, post at 60 Hz", cpu: `${Math.round(throttled.ms)} ms · ${throttled.posts} posts · ${throttled.steps} steps`, gpu: "—" });
      rows.push({ what: "worker run, post per step", cpu: `${Math.round(perStep.ms)} ms · ${perStep.posts} posts · ${perStep.steps} steps`, gpu: "—" });
      rows.push({ what: "per-step posting costs", cpu: `${(perStep.ms / throttled.ms).toFixed(1)}x`, gpu: "—" });

      return { adapter, rows };
    });

    console.log(`\nRL Phase 0 — adapter: ${table.adapter}`);
    console.table(table.rows);
    expect(table.rows.length).toBeGreaterThan(0);
  });
});
