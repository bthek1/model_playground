# Reinforcement Learning in the Browser (WebGPU or CPU)

> The **Reinforcement Learning** category of `components/layout/taskTaxonomy.ts`:
> what runs client-side on the user's GPU (WebGPU) or CPU (WebAssembly). No Python
> server in the inference path.

**Neither task is built** — Reinforcement Learning and Robotics both render the
`/tasks/$slug` placeholder. This file is the research a plan gets written from;
the procedure is [`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md).

Reinforcement learning is the best fit for a browser of anything in the taxonomy,
and the reason has nothing to do with model size.

An RL page has something no other page in this app has: **a loop the user can
watch**. Every other task is a single button press with a result. Here the agent
acts, the environment responds, the return goes up, and all of it renders at 60
frames per second on a canvas. A notebook can only show that as a plot after the
fact. A page shows it happening.

The models are also trivially small. A CartPole policy network is two dense
layers with a few thousand parameters, which is kilobytes. There is no download,
no large-model warning, and no `await model.dispose()`. What there is instead is
a **training loop that runs in the tab**, which puts these pages in the same
family as [`/training`](../../../frontend/src/routes/training.tsx) rather than in
the family of the pretrained routes.

The constraint that replaces model size is the environment. Gymnasium is Python
and MuJoCo is a native physics engine, so neither runs in a browser. The
environment has to be reimplemented in TypeScript, and that is what decides which
parts port.

---

## 1. The core stack

Nothing to install. `onnxruntime-web` is already a dependency if you want to
infer a policy trained in Python; otherwise these pages need no library at all,
because the networks are small enough to write by hand.

Three implementation tiers, and the first one covers most of both tasks:

| Tier | What it is | Where the compute goes |
|---|---|---|
| **Tabular** | a Q-table as a `Float32Array` | main thread. It is an array lookup |
| **Small networks, trained in the tab** | 2 dense layers, forward and backward by hand | WGSL matmul in `webgpu/worker.ts`, or plain TypeScript |
| **A policy trained in Python** | exported with `torch.onnx.export`, run in the tab | `onnxruntime-web`, `wasm` provider is plenty |

A two-layer MLP is small enough that hand-written backpropagation is roughly
thirty lines, and writing it out is the point of the page rather than an
obstacle. The existing WGSL matmul, scale and elementwise shaders in
[`webgpu/shaders/`](../../../frontend/src/webgpu/shaders/) already cover the
forward pass, and
[`webgpu/linearModel.ts`](../../../frontend/src/webgpu/linearModel.ts) is the
worked example of a gradient loop on the GPU.

The LOAD slot has no weights to report, so it renders
[`DeviceStatus`](../../../frontend/src/components/model/DeviceStatus.tsx) — is
there a GPU, or nothing to compute on — exactly as `/tensor` does.

---

## 2. The environment is the work

Port the environment, not the algorithm. The algorithms are short; the
environments are what Gymnasium was providing for free.

Three that are worth implementing, in increasing order of effort:

```ts
// src/rl/envs/frozenLake.ts - a grid world, roughly 40 lines.
// Discrete states and actions, so tabular Q-learning has somewhere to live.
export function step(s: number, a: number): { s: number; r: number; done: boolean } { /* ... */ }
```

- **A grid world** (FrozenLake-shaped). Discrete, deterministic or slippery, and
  it is the environment tabular Q-learning needs. Around 40 lines.
- **CartPole.** The classic, and the physics is four state variables and one
  Euler step. Around 60 lines, and it is the environment both policy-gradient
  sections use. It also renders beautifully: a cart, a pole, and an angle.
- **A 2-D reaching task** for the robotics page. Two joints, a target, and
  inverse kinematics you do not need because the policy is learning it.

What does not port: **MuJoCo**. Decision Transformer's `hopper-medium-v2` is a
native physics engine, and there is no browser equivalent that would produce
comparable trajectories. That stays as prose or moves to a server.

Run the environment and the learner in the **same Worker**. An RL step is
environment then policy then update, thousands of times per second, and posting
a message per step would make the message passing dominate the compute. Post the
render state back at 60 Hz, not per step.

```ts
// in the worker
let episode = 0;
while (training) {
  const ret = runEpisode();               // hundreds of steps, no postMessage
  if (++episode % 10 === 0) self.postMessage({ type: "progress", episode, ret });
}
```

---

## 3. Task by task

### 3.1 Tabular Q-Learning, the page that explains what a value is

Part of the taxonomy task **Reinforcement Learning** · not built.

The whole algorithm is one update rule and the whole model is a table, so this
is the cheapest page in the app to build and one of the most instructive.

Render the Q-table **over the grid**: an arrow per cell pointing at the argmax
action, coloured by value. Then run the learning loop and watch the arrows turn
around as the agent discovers the goal. That visual is the reason tabular
methods are still taught, and it is essentially impossible to convey in a
notebook cell.

Controls that belong in the RUN slot, because each one produces a visible
change: the learning rate, the discount, and epsilon. Set epsilon to zero and
watch the agent get stuck; that is the exploration lesson in one slider.

### 3.2 REINFORCE, the policy gradient in its rawest form

Part of the taxonomy task **Reinforcement Learning** · not built.

CartPole plus a two-layer policy network, trained in the tab. The forward pass
is two matmuls and a softmax; the backward pass is the log-probability times the
return.

This is the raw version, with all the variance that implies, and **the variance
is the thing to render**. Plot the per-episode
return without smoothing next to a running mean. The raw trace is violently
noisy, and that noise is the entire motivation for the next section.

### 3.3 Actor-Critic, the same gradient with far less variance

Part of the taxonomy task **Reinforcement Learning** · not built.

Same environment, same page, one more network. Run it against REINFORCE on the
same seed and plot both return curves on one chart — a live head-to-head that
finishes in about thirty seconds of wall-clock time in the tab.

This is the single best argument for the whole RL route: two algorithms, the
same environment, one visible difference, and the user can rerun it with
different seeds until they believe it. Both curves go through the lazy
[`EChart`](../../../frontend/src/components/charts/EChart.tsx) wrapper — `echarts`
is heavy and stays code-split.

### 3.4 Decision Transformer, exportable but not trainable in the tab

Part of the taxonomy task **Reinforcement Learning** · not buildable as it
stands.

`edbeeching/decision-transformer-gym-hopper-medium` has **no ONNX weights**
(checked against the Hub), and its environment is MuJoCo, so neither half ports
as it stands.

The architecture is a small GPT, so exporting it with `torch.onnx.export` would
work and the resulting model would run in `onnxruntime-web`. The blocker is the
environment rather than the network. Treat this as prose on the page, with the
sequence-modelling framing explained against the trajectories the other sections
generate.

### 3.5 RL for Language Models, prose, and correctly so

Part of the taxonomy task **Reinforcement Learning** · prose only.

This is where RL actually became ubiquitous, and it is the one part that should
stay text on the page. Nothing here is a demo: the
policy is an LLM, the reward model is another LLM, and a single PPO step is a
cluster job.

The page can gesture at it honestly with the text-generation route already in
the app: the same prompt through a base model and an instruction-tuned model,
side by side. That is the visible product of the pipeline, even though the
pipeline itself is offscreen.

### 3.6 Robotics, the perception front-end is the browser half

Taxonomy task **Robotics** · not built. Upstream research: behaviour cloning,
multimodality, action chunking.

The buildable half is **grounding an instruction: the perception front-end,
live**, and it uses two models that both have browser exports:

| Upstream (PyTorch) | Browser model | Notes |
|---|---|---|
| `google/owlv2-base-patch16-ensemble` | `Xenova/owlv2-base-patch16-ensemble` | open-vocabulary detection. Point the camera, type "the red block" |
| `depth-anything/Depth-Anything-V2-Small-hf` | `onnx-community/depth-anything-v2-small` | depth, so the target has a distance as well as a pixel |

That page is buildable today and it is the honest half of robot learning
in a browser: **grounding an instruction in what the camera sees.** Both models
come from the vision guide and are already in the catalogue.

The control half does not port. Behaviour cloning needs demonstrations,
`lerobot/pusht` needs a simulator, and action chunking is about real hardware
latency.

There is one control-half page that does work: **the multimodality problem**, on
a toy 2-D reaching task. Train behaviour cloning on
demonstrations that go around an obstacle both ways, then watch the policy
average them and drive straight into the obstacle. That failure is the whole
motivation for action chunking and diffusion policies, it runs in a tab in
seconds, and it is far more convincing seen than described.

---

## 4. Feasibility summary

| Piece | In-browser? | How | Best backend | If not |
|---|---|---|---|---|
| **Tabular Q-Learning** | Yes, excellent | a `Float32Array` plus a grid world | main thread | - |
| **REINFORCE** | Yes | 2-layer MLP, trained in a Worker | WebGPU or plain TS | - |
| **Actor-Critic** | Yes | as above, two networks | WebGPU or plain TS | - |
| **Algorithm head-to-head** | Yes | same seed, two curves | Worker | - |
| **Decision Transformer** | No | no ONNX weights, and MuJoCo | - | export it, and replace the env |
| **RL for LLMs** | No, and correctly | - | - | prose, plus a text-generation route |
| **Behaviour cloning** (Robotics) | Toy env only | 2-D reaching task in TypeScript | Worker | `lerobot` datasets to server |
| **The multimodality failure** (Robotics) | Yes, and worth it | the toy env above | Worker | - |
| **Grounding an instruction** (Robotics) | Yes | OWLv2 plus Depth Anything V2 | WebGPU | - |

Rule of thumb: **the algorithm always ports, the environment usually does not.**
Where you are willing to write the environment in TypeScript, the page is
better than the notebook. Where the environment is MuJoCo or real hardware, it
is not a page.

---

## 5. Memory and performance notes

Nothing here downloads weights, so the rules are all about the loop.

- **Never post a message per environment step.** Batch the render state and post
  at 60 Hz. This is the mistake that makes an RL page slower than the notebook.
- **Keep the environment and the learner in one Worker.** Splitting them across
  a message boundary costs more than the arithmetic they perform.
- **Decouple the simulation rate from the frame rate.** Training should run as
  fast as it can; rendering should run at 60 Hz. A page that renders every step
  is throttled by the display.
- **Seed everything, and expose the seed.** Two RL runs differ enormously, so a
  comparison with no fixed seed is not a comparison. This is the most important
  control on the page.
- **Offer a stop button that actually stops.** A training loop in a Worker will
  happily run forever, and `useModelWorker`'s `cancel()` already exists for it —
  it tears the worker down and returns to `idle`.
- **Nothing to dispose.** No GPU session outlives an episode. Terminate the
  Worker on unmount and that is the whole cleanup story.

---

## 6. Reference

- **[`src/webgpu/`](../../../frontend/src/webgpu/)**: the WGSL matmul,
  elementwise, scale and transpose shaders plus `linearModel.ts` are what the
  policy networks need. [`/training`](../../../frontend/src/routes/training.tsx)
  is the closest existing route and the right one to read first — including
  [`../../standards/model-page-pattern.md`](../../standards/model-page-pattern.md)
  §7 on why it is allowed its own layout, and why a new page probably is not.
- **onnxruntime-web** if a policy is trained in Python and exported.
- **Vision models** for the robotics perception page:
  [`Computer_Vision_Models_in_React_WebGPU_and_CPU.md`](Computer_Vision_Models_in_React_WebGPU_and_CPU.md).
  That page is the one piece of this category that downloads weights, so it needs
  the full four-slot treatment rather than the training-loop shape.
- **Page construction**: [`HOW_TO_ADD_A_TASK_PAGE.md`](HOW_TO_ADD_A_TASK_PAGE.md).
- Method and model recommendations here come from a companion collection of
  Python notebooks, which is a separate project and not a dependency of this
  repo. Only the robotics perception page needs a checkpoint at all.
