# Reinforcement Learning in the Browser (CPU, on purpose)

> The **Reinforcement Learning** category of
> [`taskTaxonomy.ts`](../../frontend/src/components/layout/taskTaxonomy.ts): what runs
> client-side, and on what. No Python server in the inference path — and, in this
> category, no checkpoint for most of it.
>
> This file began as issue [#6](https://github.com/bthek1/model_playground/issues/6) and
> moved here when its first route shipped, the same way [`audio.md`](audio.md),
> [`vision.md`](vision.md), [`graph.md`](graph.md) and [`tabular.md`](tabular.md) did: a
> roadmap that documents *shipped* code has to be reviewable alongside the code it
> describes, which an issue body cannot be.

| Roadmap section | Route | Status |
|---|---|---|
| Tabular Q-Learning (§3.1) | [`/rl`](../../frontend/src/routes/rl.tsx) | **Shipped** — FrozenLake, checked against value iteration ([#51](https://github.com/bthek1/model_playground/issues/51)) |
| REINFORCE (§3.2) | `/rl` | Planned — [#52](https://github.com/bthek1/model_playground/issues/52) |
| Actor-Critic (§3.3) | `/rl` | Planned — [#52](https://github.com/bthek1/model_playground/issues/52) |
| Decision Transformer (§3.4) | — | Prose — [#52](https://github.com/bthek1/model_playground/issues/52) |
| RL for Language Models (§3.5) | — | Prose — [#52](https://github.com/bthek1/model_playground/issues/52) |
| Robotics: grounding (§3.6) | [`/robotics`](../../frontend/src/routes/robotics.tsx) | [#53](https://github.com/bthek1/model_playground/issues/53) |
| Robotics: behaviour cloning (§3.6) | `/robotics` | Planned — [#54](https://github.com/bthek1/model_playground/issues/54) |

Reinforcement learning is the best fit for a browser of anything in the taxonomy, and the
reason has nothing to do with model size.

An RL page has something no other page in this app has: **a loop the user can watch**.
Every other task is a single button press with a result. Here the agent acts, the
environment responds, the return goes up, and all of it renders on a canvas while it
happens. A notebook can only show that as a plot after the fact.

The models are also trivially small. A CartPole policy network is two dense layers with a
few thousand parameters; a FrozenLake Q-table is 64 numbers. There is no download, no
large-model warning, and no `dispose()`. What there is instead is a **training loop that
runs in the tab**, which puts these pages in the same family as
[`/training`](../../frontend/src/routes/training.tsx) and the Tabular routes rather than
the pretrained ones.

The constraint that replaces model size is the **environment**. Gymnasium is Python and
MuJoCo is a native physics engine, so neither runs in a browser. The environment has to be
reimplemented in TypeScript, and that is what decides which parts port.

---

## The feasibility bar this file is filtered by

Every row clears the same two tests the other category roadmaps are swept against:

1. **It runs client-side.** Not necessarily on the GPU — §0 is the measurement that says
   where.
2. **Its cheapest usable checkpoint is under ~500 MB**, measured off the Hub.

**This category passes the second vacuously**: nothing here downloads a checkpoint except
§3.6's grounding half, whose two models are already shipped entries. The filter that
removed six models from the NLP roadmap removes nothing here. **The binding constraint is
the environment, not the model** — §2 is that argument, and Decision Transformer and RL
for LLMs are its only casualties.

**There is no pretrained policy worth loading, and the reason is more specific than "none
exist"** (checked against the Hub 2026-09-26). The Hub does publish ONNX policies under
`reinforcement-learning` — dozens — and **every one is a Unity ML-Agents export**
(`ThomasSimonini/MLAgents-Pyramids` ships `Pyramids.onnx`), whose environment is a Unity
build and whose input signature is that build's sensor layout. The checkpoints for
environments we *can* port are pickles, not graphs: `sb3/ppo-CartPole-v1` publishes
`ppo-CartPole-v1.zip` and `policy.pth` and no ONNX at all. So question 1 of
[`adding-a-task-page.md`](../guides/adding-a-task-page.md) §0 fails in both flavours — a
missing **export** for CartPole, a missing **environment** for ML-Agents — and the answer
is the same either way: **this category trains in the tab.**

---

## 0. The measurement that shaped the module: does an RL step belong on the GPU?

The roadmap's tier table (§1) offered "WGSL matmul, or plain TypeScript" for the policy
networks and handed the LOAD band a `DeviceStatus`, as `/tensor` has. That was a guess, so
#51 measured it before `src/rl/` was shaped around an answer —
[`e2e/specs/webgpu/rl-phase0.spec.ts`](../../frontend/e2e/specs/webgpu/rl-phase0.spec.ts)
(`just fe-e2e-rl-phase0`), steps per second in Chromium, the GPU column through
`linearModel.ts`'s real `runMatmul`:

| what | CPU | GPU (SwiftShader) |
|---|---|---|
| Q-learning step, 8×8 table | 14 800 000 | — |
| 1×1 matmul, write → dispatch → read | — | 129 |
| act, 4→32→2, batch 1 | 1 190 000 | 67 |
| act, 4→128→2, batch 1 | 399 000 | 66 |
| act, 4→512→2, batch 1 | 111 000 | 66 |
| update, 4→32→2, batch 200 | 4 049 | 26 |
| update, 4→128→2, batch 200 | 1 141 | 24 |
| update, 4→512→2, batch 200 | 286 | 17 |

The only adapter this machine's browser could reach was SwiftShader, so the GPU column is
a software rasteriser and is **not** a GPU's speed. It does not need to be: **the GPU
numbers are flat across a 16× change in width**, because at batch 1 they measure the round
trip, not the arithmetic. Grant a real device a generous 0.1 ms round trip and acting
costs two of them — ≤ 5 000 steps a second, still 20× behind the CPU's *worst* row. The
one place a real GPU could plausibly win is the batch-200 update at width 512, which runs
once per episode behind a per-step rollout that dominates it.

**Decisions, both recorded in [`rl/limits.ts`](../../frontend/src/rl/limits.ts):**

1. **`src/rl/` touches no GPU at all.** Not "CPU as a fallback": the CPU is the right
   answer, as it is for `/vad` and the Tabular trees. This reverses §1's tier table.
2. **`/rl` has no LOAD band.** It raises no GPU question, so `DeviceStatus` would answer
   one nobody asked; there is nothing to download either. It is three bands, like
   `/time-series-forecasting`, and says where the fourth would have been
   ([`model-page-pattern.md`](../standards/model-page-pattern.md) §7).

**The message cost**, quoted as a factor rather than a claim: the same 3000-episode run of
the slippery 8×8 in the real worker took **32 ms** posting its render state at 60 Hz
(2 posts) and **1614 ms** posting after every one of its 230 956 steps — **50×**.
`session.test.ts` counts posts against steps so it cannot regress.

---

## 1. The core stack

Nothing to install and no ML library. `src/rl/` is plain TypeScript, like `src/tabular/`
and `src/forecast/`:

| Module | What it is |
|---|---|
| [`types.ts`](../../frontend/src/rl/types.ts) | the `Env` interface, the request, the worker protocol |
| [`envs/gridWorld.ts`](../../frontend/src/rl/envs/gridWorld.ts) | FrozenLake, transcribed from Gymnasium |
| [`qLearning.ts`](../../frontend/src/rl/qLearning.ts) | the Q-table, ε-greedy, the update rule, the ε schedule |
| [`valueIteration.ts`](../../frontend/src/rl/valueIteration.ts) | the **exact** Q\* — the learner's CPU reference |
| [`stepper.ts`](../../frontend/src/rl/stepper.ts) | the seam between a learner and the loop that drives it |
| [`session.ts`](../../frontend/src/rl/session.ts) | the pure worker-side handler: pacing, throttled posting, Stop |
| [`rl.worker.ts`](../../frontend/src/rl/rl.worker.ts) + [`client.ts`](../../frontend/src/rl/client.ts) | the thin wrapper and the `{ promise, cancel, control }` handle |
| [`policyNet.ts`](../../frontend/src/rl/policyNet.ts) | a two-layer tanh network, forward and backward by hand (§3.2's foundation) |
| [`limits.ts`](../../frontend/src/rl/limits.ts) | §0's numbers and the caps they set |

Plus [`hooks/useRlTraining.ts`](../../frontend/src/hooks/useRlTraining.ts), shaped on
`useGraphTraining`, and [`components/rl/`](../../frontend/src/components/rl/) for the
canvas and the return chart.

**`/graph`'s trio is the precedent, not `useModelWorker`.** "A long-lived training loop in
a worker, streaming metrics to a canvas" already existed three times over
(`graphSession.ts`, `trainGraphInWorker`, `useGraphTraining`); `useModelWorker` is
request/response over a downloaded model and has nothing this page needs.

**The raw-WebGPU carve-out holds in both directions**: no ML framework in `src/webgpu/`,
and no WGSL in `src/rl/`. `policyNet.ts` reuses `linearModel.ts`'s `cpuMatmul` and
`LinearTrainer.softmaxRows` rather than a second copy of either.

---

## 2. The environment is the work

Port the environment, not the algorithm. The algorithms are short; the environments are
what Gymnasium was providing for free.

- **A grid world** (FrozenLake). Shipped — §3.1.
- **CartPole.** The physics is four state variables and one Euler step; both
  policy-gradient sections use it. §3.2.
- **A 2-D reaching task** for the robotics page. §3.6.

What does not port: **MuJoCo**. Decision Transformer's `hopper-medium-v2` is a native
physics engine with no browser equivalent.

**Transcribe, never invent.** `gridWorld.ts` carries Gymnasium's maps letter for letter,
its action order (LEFT, DOWN, RIGHT, UP), its wall rule, its registered time limits
(100 / 200) and its slip rule — the intended direction a third of the time and each
*perpendicular* a third, **never backwards**. The folk version "slip to any neighbour" is a
different MDP, and every number a user compares against a tutorial would be wrong with
nothing on screen to say so.

**`terminated` and `truncated` are separate**, as in Gymnasium. An episode that ends in a
hole has no future; one that ends because the clock ran out does, so the learner still
bootstraps through a truncation. `qLearning.test.ts` pins both halves.

**The environment and the learner live in one worker**, and the loop yields a
*macrotask* between slices. That second half is the trap: a CPU loop that only awaits
resolved promises never lets the event loop deliver a message, so Stop would be queued
behind the run it was meant to interrupt. `webgpu/worker.ts` gets away without it because
its awaits are real GPU work, which resolves as a task; this loop has no GPU to lean on.
`session.ts` yields through a `MessageChannel`, because `setTimeout(0)` is clamped to 4 ms
after a few nested calls.

---

## 3. Task by task

### 3.1 Tabular Q-Learning — shipped

Part of the taxonomy task **Reinforcement Learning** · [`/rl`](../../frontend/src/routes/rl.tsx) · [#51](https://github.com/bthek1/model_playground/issues/51).

The Q-table drawn **over the grid**: an arrow per cell at the argmax action, the cell
shaded by its value, repainted from the worker at most 60 times a second. The arrows turn
around as the reward the agent finds propagates backward through the table.

**Value iteration is the reference, and it is the most valuable file in the module.** A
wrong Bellman update — no discount, a bootstrap through the terminal state, the max over
the wrong row — still finds the goal on a 4×4 grid and still draws a rising curve. On an
MDP this small the exact answer is twenty lines, so the tests compare against it: on the
deterministic grid, entry for entry to 10⁻³; on the slippery grid, to 0.02 under the
Robbins–Monro step sizes the convergence theorem actually requires (a constant α leaves
noise of order α on every entry, and a tolerance loose enough to pass that would pass a
wrong update too). The page runs the same check on the table in hand — "agrees with value
iteration's exact optimal policy in 11 of 11 cells" — because it is microseconds.

**Found by that check: the agent and the environment must not share a random stream.**
The slippery test first seeded both with 7. Each action draw and the slip that follows it
were then the *same number*, so which way the agent slipped depended on which action it
chose — dynamics the model does not have — and Q-learning converged, confidently and with
a rising curve, to values **four times too high** (Q(0, ·) ≈ 0.29 against Q\* ≈ 0.07).
`QLearningRun` derives the agent's seed as `seed ^ 0x51ed27`, and a test pins the failure.

**Ties go to the first action, as `np.argmax` does, and the page depends on it.** From a
cold table every action is worth 0, so a greedy agent picks LEFT, walks into the wall at
the start, learns that the wall is worth 0 — which it already thought — and does it again
for ever. That is **"set ε to 0 and the agent never finds the goal"**, stated beside the
control and pinned by a unit test and an E2E spec. Break ties at random (as Gymnasium's
own FrozenLake tutorial does) and a cold table explores by accident, which hides the
lesson.

**The same rule makes a small *constant* ε useless, so ε decays by default.** Measured on
the deterministic 4×4 at α = 0.5: ε = 0.2 held for 2000 episodes reached the goal **zero**
times — every exploratory step off the start is undone by the next greedy one. A linear
decay from ε = 1 to 0.05 over the first 80% of the run finds the goal within ~170 episodes
on every seed tried and ends on the exact optimal policy.

**Defaults are per (map, slipperiness), swapped with the environment, never inherited.**
A deterministic grid wants a large α because every sample is exact; a slippery one wants
a small α, or each one-in-three slip drags the estimate around (α = 0.5 on the slippery
4×4 ended with 8–9 of 11 cells optimal; α = 0.1 with 10–11). The 8×8 needs ~20 000
episodes to find its goal at all — and at α = 0.5 one seed in three **never** does, which
the page leaves visible: sparse reward is a real failure mode, and the seed is the most
important control on the page.

**Train is the only button that spends.** The grid, the slipperiness, α, γ, ε, the
episode count and the seed are choices. Two controls legitimately change a run *in
progress* without restarting it — the simulation speed and a live ε that takes over from
the schedule — and they sit in RUN, saying which kind they are. The speed dial is a
**simulation** dial, not a frame rate: at "max" a thousand episodes take milliseconds, so
the page defaults to 3000 steps a second, slow enough to watch the agent walk.

### 3.2 REINFORCE, the policy gradient in its rawest form

Part of the taxonomy task **Reinforcement Learning** · planned in
[#52](https://github.com/bthek1/model_playground/issues/52).

CartPole plus a two-layer policy network, trained in the tab. The forward pass is two
matmuls and a softmax; the backward pass is the log-probability times the return. **The
variance is the thing to render**: the per-episode return without smoothing, next to a
running mean.

### 3.3 Actor-Critic, the same gradient with far less variance

Part of the taxonomy task **Reinforcement Learning** · planned in
[#52](https://github.com/bthek1/model_playground/issues/52).

Same environment, same page, one more network, run against REINFORCE on the same seed.

### 3.4 Decision Transformer — prose

`edbeeching/decision-transformer-gym-hopper-medium` has no ONNX weights and its
environment is MuJoCo. Planned as prose in #52.

### 3.5 RL for Language Models — prose

The policy is an LLM, the reward model is another, and one PPO step is a cluster job.
Planned as prose in #52.

### 3.6 Robotics

Taxonomy task **Robotics**. The perception half — grounding an instruction with OWLv2 and
Depth Anything V2 — is [#53](https://github.com/bthek1/model_playground/issues/53); the
behaviour-cloning multimodality failure on a toy reaching task is
[#54](https://github.com/bthek1/model_playground/issues/54).

---

## 4. Feasibility summary

| Piece | In-browser? | How | Runs on | If not |
|---|---|---|---|---|
| **Tabular Q-Learning** | **Shipped** | a `Float32Array` plus a grid world | CPU, in a worker | - |
| **REINFORCE** | Yes | 2-layer MLP, trained in a worker | CPU (§0) | - |
| **Actor-Critic** | Yes | as above, two networks | CPU (§0) | - |
| **Algorithm head-to-head** | Yes | same seed, two curves | CPU (§0) | - |
| **Decision Transformer** | No | no ONNX weights, and MuJoCo | - | export it, and replace the env |
| **RL for LLMs** | No, and correctly | - | - | prose, plus `/text-generation` |
| **Behaviour cloning** (Robotics) | Toy env only | 2-D reaching task in TypeScript | CPU | `lerobot` datasets to a server |
| **Grounding an instruction** (Robotics) | Yes | OWLv2 plus Depth Anything V2 | WebGPU | - |

Rule of thumb: **the algorithm always ports, the environment usually does not.**

---

## 5. Memory and performance notes

Nothing here downloads weights, so the rules are all about the loop.

- **Never post a message per environment step.** Post the render state on a wall-clock
  interval (≤ 60 Hz). Measured at **50×** in §0.
- **Keep the environment and the learner in one worker.**
- **Decouple the simulation rate from the frame rate.** The loop runs as fast as the dial
  allows; the canvas repaints at most every 16 ms.
- **Yield a macrotask between slices**, or Stop cannot arrive (§2).
- **Transfer a copy, never the buffer the loop is still writing** — and never reference a
  transferred buffer afterwards. The first version of `session.ts` put the last progress
  post's table in the result too; that post had *transferred* it, so the result could not
  be cloned and every run failed on its last line. The unit harness passed because it
  handed objects across by reference; it now does what `postMessage` does
  (`structuredClone` with `transfer`), and fails on the old code.
- **Seed everything, and expose the seed.** A comparison with no fixed seed is not a
  comparison.
- **Nothing to dispose.** Terminate the worker on unmount and that is the whole cleanup
  story. Nothing is persisted and nothing is posted: `createInferenceRun` still has no
  caller.

---

## 6. Reference

- [`/graph`](../../frontend/src/routes/graph.tsx) and its trio — the streaming-training
  precedent.
- [`webgpu/linearModel.ts`](../../frontend/src/webgpu/linearModel.ts) — `cpuMatmul` and
  the softmax the policy networks reuse.
- Gymnasium's [`frozen_lake.py`](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/frozen_lake.py)
  — the source `gridWorld.ts` transcribes.
- Page construction: [`adding-a-task-page.md`](../guides/adding-a-task-page.md).
