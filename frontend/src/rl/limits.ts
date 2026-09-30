// The caps, and the measurement that set them.
//
// Phase 0 of #51 asked one question before the module was shaped around an
// answer: **does an RL step belong on the GPU?** The roadmap's tier table said
// "WGSL matmul, or plain TypeScript" for the policy networks and handed the
// LOAD band a `DeviceStatus`, as `/tensor` has. Measured in Chromium through
// `e2e/specs/webgpu/rl-phase0.spec.ts` (`just fe-e2e-rl-phase0`), steps per
// second, the GPU column through `linearModel.ts`'s real `runMatmul`:
//
//   what                              CPU            GPU (SwiftShader)
//   Q-learning step, 8×8 table        14 800 000     —
//   1×1 matmul, write→dispatch→read   —              129
//   act, 4→32→2, batch 1              1 190 000      67
//   act, 4→128→2, batch 1             399 000        66
//   act, 4→512→2, batch 1             111 000        66
//   update, 4→32→2, batch 200         4 049          26
//   update, 4→128→2, batch 200        1 141          24
//   update, 4→512→2, batch 200        286            17
//
// The only adapter this machine's browser could reach was SwiftShader, so the
// GPU column is a software rasteriser and must not be quoted as a GPU's speed.
// It does not need to be: **the GPU numbers are flat across a 16× change in
// width**, because at batch 1 they are the round trip, not the arithmetic. A
// real device's round trip is shorter, and it can be granted a generous 0.1 ms
// without changing the answer — acting would then cost two round trips, ≤ 5 000
// steps a second, still 20× behind the CPU's *worst* row. The one place a real
// GPU could plausibly win is the batch-200 update at width 512, which runs once
// per episode; the per-step rollout in front of it still dominates the episode.
//
// So, and this is the decision the issue asked Phase 0 to make:
//
//   1. **`src/rl/` touches no GPU at all.** Not "CPU as a fallback" — the CPU
//      is the right answer, the way it is for `/vad` and the Tabular trees.
//      `policyNet.ts` injects `cpuMatmul` synchronously; see its header.
//   2. **The page has no LOAD band.** It raises no GPU question, so
//      `DeviceStatus` would answer one nobody asked, and there is nothing to
//      download. `/rl` is three bands, as `/time-series-forecasting` is, and
//      says where the fourth would have been.
//
// **The message cost**, on the same 3000-episode run of the slippery 8×8 in the
// real worker: posting the render state at 60 Hz took **32 ms** (2 posts);
// posting it after every one of the 230 956 steps took **1614 ms** — **50×**.
// That is the roadmap's §5 claim quoted as a factor, and `session.test.ts`
// counts posts against steps so it cannot regress.

/** The episode caps per map. The 8×8 needs ~20 000 to find its goal reliably. */
export const MAX_EPISODES = 50_000;

/** Steps per second the speed dial offers; `null` is "as fast as the CPU goes". */
export const SPEEDS: readonly (number | null)[] = [30, 300, 3000, null];

/** Where the page starts: slow enough to watch the agent walk, fast enough to finish. */
export const DEFAULT_SPEED = 3000;
