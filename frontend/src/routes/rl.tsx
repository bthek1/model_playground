// Reinforcement Learning — tabular Q-learning on FrozenLake, and REINFORCE
// against Actor-Critic on CartPole. The first page in the app whose subject is a
// loop the user watches.
//
// Every other route is a press and a result. Here the agent acts, the
// environment answers, and the Q-table's arrows turn around on screen while it
// learns. The model is a `Float32Array`: nothing is downloaded, nothing is
// uploaded, and there is nothing to dispose.
//
// **Three bands, not four**, and `rl/limits.ts` carries the measurement that
// decided it. An RL step is thousands of tiny matmuls a second, and at batch 1
// the round trip to a GPU costs more than the arithmetic by orders of
// magnitude — so this page raises no GPU question, `DeviceStatus` would answer
// one nobody asked, and there is no download to report either. The page says
// where the band would have been (`/time-series-forecasting` is the precedent,
// model-page-pattern.md §7).
//
// **Train is the only button that spends.** The grid, the slipperiness, α, γ,
// ε, the episode count and the seed are all choices and land in SELECT. The
// tempting failure here is "drag ε and watch it retrain" — the
// five-samples-five-inferences bug with a slider in front of it. What ε *may*
// do live is change the behaviour of a run already in progress, which is not a
// re-run; that control sits in RUN, and says which kind it is.
//
// **SELECT spans (environment, algorithm)**, and the invalid pairs are disabled
// with the reason on the row — `ModelPicker`'s behaviour, transplanted. There
// is one taxonomy row, so there is one route: the grid with Q-learning (#51) and
// CartPole with the two policy gradients (#52) are two rows of one control.
// "Run both" is the head-to-head: both policy gradients, one seed, in sequence.

import { createFileRoute } from "@tanstack/react-router";
import { Gamepad2, GitCompare, Loader2, Play, Square } from "lucide-react";
import { useMemo, useState } from "react";

import { ErrorNote } from "@/components/model/ErrorNote";
import { ModelSlot } from "@/components/model/ModelPage";
import { OutputPanel } from "@/components/model/OutputPanel";
import { CartPoleCanvas } from "@/components/rl/CartPoleCanvas";
import { GridCanvas } from "@/components/rl/GridCanvas";
import { ReturnChart } from "@/components/rl/ReturnChart";
import { Button } from "@/components/ui/button";
import { type RlRunRecord, runKey, useRlTraining } from "@/hooks/useRlTraining";
import { cn } from "@/lib/utils";
import { ACTION_NAMES, GridWorld, MAPS } from "@/rl/envs/gridWorld";
import { DEFAULT_SPEED, MAX_EPISODES, SPEEDS } from "@/rl/limits";
import { EPSILON_FLOOR, greedyRollout, QTable } from "@/rl/qLearning";
import { PG_DEFAULTS } from "@/rl/policyGradient";
import type {
  GridMapId,
  PolicyGradientRequest,
  QLearningRequest,
  RlAlgorithm,
  RlTrainRequest,
} from "@/rl/types";
import { optimalActions, valueIteration } from "@/rl/valueIteration";

export const Route = createFileRoute("/rl")({
  component: RlPage,
});

/**
 * Defaults per (map, slipperiness), **swapped with the environment, never
 * inherited** — measured on the grids themselves. A deterministic grid wants a
 * large α because every sample is exact; a slippery one wants a small α, or
 * each one-in-three slip drags the estimate around (α = 0.5 on the slippery 4×4
 * ended with the optimal policy in 8–9 of 11 cells; α = 0.1 in 10–11). The 8×8
 * needs ~20 000 episodes to find its goal at all.
 */
export const DEFAULTS: Record<`${GridMapId}-${"det" | "slip"}`, { alpha: number; episodes: number }> = {
  "4x4-det": { alpha: 0.5, episodes: 1000 },
  "4x4-slip": { alpha: 0.1, episodes: 3000 },
  "8x8-det": { alpha: 0.5, episodes: 20000 },
  "8x8-slip": { alpha: 0.1, episodes: 20000 },
};

const defaultsFor = (map: GridMapId, slippery: boolean) =>
  DEFAULTS[`${map}-${slippery ? "slip" : "det"}`];

/** The success rate is averaged over this many episodes. */
const WINDOW = 100;
/** CartPole's returns are averaged over fewer: a run is only ~400 episodes. */
const PG_WINDOW = 20;

export type EnvId = "frozenlake" | "cartpole";

/** The algorithms this page offers; behaviour cloning lives on /robotics. */
type PageAlgorithm = Exclude<RlAlgorithm, "behaviour-cloning">;

/**
 * Which algorithms each environment pairs with, and why the others do not.
 * The reason is rendered on the disabled row, because the constraint is the
 * lesson: a table needs states it can index.
 */
export const PAIRINGS: Record<EnvId, Partial<Record<PageAlgorithm, string>>> = {
  frozenlake: {
    reinforce:
      "Not paired here: a random policy reaches FrozenLake's goal about once in seventy episodes, so almost every REINFORCE gradient is exactly zero.",
    "actor-critic":
      "Not paired here, for the same reason: the reward is too sparse for a policy gradient to find from a random start.",
  },
  cartpole: {
    "q-learning":
      "Tabular methods need discrete states. CartPole's state is four real numbers — there is no row of the table to look up.",
  },
};

const ALGORITHMS: { id: PageAlgorithm; label: string; note: string }[] = [
  { id: "q-learning", label: "Tabular Q-learning", note: "a table of action values, one update rule" },
  { id: "reinforce", label: "REINFORCE", note: "the policy gradient, weighted by the raw return" },
  { id: "actor-critic", label: "Actor-Critic", note: "the same gradient, weighted by a learned advantage" },
];


function RlPage() {
  const rl = useRlTraining();

  const [env, setEnv] = useState<EnvId>("frozenlake");
  const [algorithm, setAlgorithm] = useState<PageAlgorithm>("q-learning");
  const [pg, setPg] = useState(() => pgDefaults("reinforce"));

  const [map, setMap] = useState<GridMapId>("4x4");
  const [slippery, setSlippery] = useState(false);
  const [alpha, setAlpha] = useState(defaultsFor("4x4", false).alpha);
  const [gamma, setGamma] = useState(0.95);
  const [epsilon, setEpsilon] = useState(1);
  const [decay, setDecay] = useState(true);
  const [episodes, setEpisodes] = useState(defaultsFor("4x4", false).episodes);
  const [seed, setSeed] = useState(1);
  const [speed, setSpeed] = useState<number | null>(DEFAULT_SPEED);
  const [liveEpsilon, setLiveEpsilon] = useState<number | null>(null);

  /** Switching the environment swaps in its own defaults rather than keeping the last one's. */
  function chooseEnv(nextMap: GridMapId, nextSlippery: boolean) {
    setMap(nextMap);
    setSlippery(nextSlippery);
    const d = defaultsFor(nextMap, nextSlippery);
    setAlpha(d.alpha);
    setEpisodes(d.episodes);
  }

  /** A new environment picks its first valid algorithm, with that algorithm's defaults. */
  function chooseEnvironment(next: EnvId) {
    setEnv(next);
    const first = ALGORITHMS.find((a) => !PAIRINGS[next][a.id])!.id;
    chooseAlgorithm(first);
  }

  /** Each algorithm brings its own measured defaults — never the previous one's. */
  function chooseAlgorithm(next: PageAlgorithm) {
    setAlgorithm(next);
    if (next !== "q-learning") setPg(pgDefaults(next));
  }

  const qRequest: QLearningRequest = {
    algorithm: "q-learning",
    map,
    slippery,
    alpha,
    gamma,
    epsilon,
    decay,
    episodes,
    seed,
  };
  const pgRequest = (alg: "reinforce" | "actor-critic"): PolicyGradientRequest => ({
    algorithm: alg,
    env: "cartpole",
    ...pg,
    // "Run both" gives each algorithm its own rates — REINFORCE collapses at
    // Actor-Critic's (policyGradient.ts) — and shares everything else.
    ...(alg === algorithm ? {} : pgRates(alg)),
    seed,
  });
  const request: RlTrainRequest = algorithm === "q-learning" ? qRequest : pgRequest(algorithm);

  const train = () => {
    setLiveEpsilon(null);
    rl.start(request, speed);
  };

  const runBoth = () => rl.startAll([pgRequest("reinforce"), pgRequest("actor-critic")], speed);

  const changeSpeed = (next: number | null) => {
    setSpeed(next);
    if (rl.training) rl.control({ speed: next });
  };

  const changeLiveEpsilon = (next: number) => {
    setLiveEpsilon(next);
    rl.control({ epsilon: next });
  };

  // The render state to draw: the run in progress, or the one that finished.
  const render = rl.render;
  const lastRequest = rl.history[rl.history.length - 1]?.request;
  const lastQ = lastRequest?.algorithm === "q-learning" ? lastRequest : undefined;
  const shownMap = render?.kind === "grid" ? render.map : map;

  return (
    <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-6 md:h-full md:min-h-0">
      <header className="min-w-0">
        <h1 className="mb-1 flex items-center gap-2 text-2xl font-semibold">
          <Gamepad2 className="size-6" /> Reinforcement Learning
        </h1>
        <p className="text-sm text-muted-foreground">
          Agents trained in front of you, on environments transcribed from
          Gymnasium: tabular Q-learning on FrozenLake, where the whole model is
          a table and you watch its arrows turn toward the goal — and REINFORCE
          against Actor-Critic on CartPole, where the difference between them is
          the noise in the curve. Nothing is downloaded and nothing leaves this
          tab.
        </p>
      </header>

      {/* Three bands, not four — see the note at the top of this file and
          model-page-pattern.md §7. */}
      <div
        className={cn(
          "grid min-h-0 grid-cols-1 gap-6 md:flex-1",
          "md:grid-cols-2 md:grid-rows-[auto_minmax(0,1fr)]",
          "md:[grid-template-areas:'setup_setup'_'work-a_work-b']",
          "xl:grid-cols-[minmax(17rem,20rem)_minmax(0,1fr)_minmax(0,1fr)]",
          "xl:grid-rows-[minmax(0,1fr)]",
          "xl:[grid-template-areas:'setup_work-a_work-b']",
        )}
      >
        <div className="flex min-w-0 flex-col gap-5 rounded-lg border bg-muted/30 p-4 md:[grid-area:setup] md:flex-row md:gap-8 xl:max-h-full xl:flex-col xl:gap-5 xl:self-start xl:overflow-y-auto">
          <ModelSlot step={1} label="Agent & environment" dense className="min-w-0 md:flex-1 xl:flex-none">
            <div className="space-y-4">
              <AlgorithmPicker
                env={env}
                algorithm={algorithm}
                onEnv={chooseEnvironment}
                onAlgorithm={chooseAlgorithm}
                disabled={rl.training}
              />
              {env === "cartpole" ? (
                <PgHyperparameters
                  algorithm={algorithm as "reinforce" | "actor-critic"}
                  value={pg}
                  onChange={setPg}
                  seed={seed}
                  onSeed={setSeed}
                  disabled={rl.training}
                />
              ) : (
                <>
              <EnvPicker map={map} slippery={slippery} onChoose={chooseEnv} disabled={rl.training} />
              <Hyperparameters
                alpha={alpha}
                onAlpha={setAlpha}
                gamma={gamma}
                onGamma={setGamma}
                epsilon={epsilon}
                onEpsilon={setEpsilon}
                decay={decay}
                onDecay={setDecay}
                episodes={episodes}
                onEpisodes={setEpisodes}
                seed={seed}
                onSeed={setSeed}
                disabled={rl.training}
              />
                </>
              )}
            </div>
          </ModelSlot>

          <div className="min-w-0 space-y-1.5 md:w-80 md:shrink-0 xl:w-auto" data-testid="no-load-band">
            <p className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
              Nothing to load
            </p>
            <p className="text-xs leading-snug text-muted-foreground">
              Every other model page has a band here for downloading weights or
              probing a GPU. This one has neither: the model is a table that
              starts at zero, and it runs on <strong>the CPU, on purpose</strong>.
              A Q-learning step is an array lookup — about 15 million a second
              in a worker — and at the sizes RL uses, the round trip to a GPU
              costs more than the arithmetic it would do.
            </p>
            <p className="text-xs leading-snug text-muted-foreground">
              No pretrained policy exists to load, either: the Hub&apos;s ONNX
              policies are Unity ML-Agents exports for environments that cannot
              run here, and the ones for environments that can are PyTorch
              pickles with no ONNX export.
            </p>
          </div>
        </div>

        <div className="flex min-h-0 min-w-0 flex-col md:overflow-y-auto md:[grid-area:work-a]">
          <ModelSlot
            step={2}
            label="Train"
            className="flex min-h-0 flex-1 flex-col [&>*:last-child]:min-h-0 [&>*:last-child]:flex-1"
          >
            <div className="flex min-h-0 flex-1 flex-col gap-4">
              <SpeedDial speed={speed} onSpeed={changeSpeed} />
              {algorithm === "q-learning" && (
                <LiveEpsilon
                  training={rl.training}
                  value={liveEpsilon ?? rl.progress?.epsilon ?? epsilon}
                  onChange={changeLiveEpsilon}
                />
              )}
              {env === "cartpole" && (
                <p className="text-xs leading-snug text-muted-foreground" data-testid="run-both-note">
                  <strong>Run both</strong> trains REINFORCE and then
                  Actor-Critic at the same seed ({seed}) and draws them on one
                  chart. Two seeds of one algorithm differ more than two
                  algorithms at one seed, so the seed is fixed and on screen —
                  change it and run both again until you believe the result.
                </p>
              )}
              <p className="text-xs leading-snug text-muted-foreground">
                Everything in band 1 is a <strong>choice</strong>: changing it
                trains nothing until you press Train. The two controls above are
                different — they change the run <em>in progress</em> without
                restarting it.
              </p>

              <ErrorNote message={rl.error} />

              <div className="sticky bottom-0 mt-2 flex flex-wrap items-center gap-2 border-t bg-background/80 pt-3 backdrop-blur">
                <Button onClick={train} disabled={rl.training} data-testid="train-button">
                  {rl.training ? (
                    <>
                      <Loader2 className="size-4 animate-spin" /> Training…
                    </>
                  ) : (
                    <>
                      <Play className="size-4" /> Train
                    </>
                  )}
                </Button>
                {env === "cartpole" && (
                  <Button variant="secondary" onClick={runBoth} disabled={rl.training} data-testid="run-both">
                    <GitCompare className="size-4" /> Run both
                  </Button>
                )}
                <Button variant="outline" onClick={rl.stop} disabled={!rl.training}>
                  <Square className="size-4" /> Stop
                </Button>
                {rl.progress && (
                  <span
                    className="font-mono text-xs text-muted-foreground tabular-nums"
                    data-testid="rl-progress"
                  >
                    episode {rl.progress.episode.toLocaleString()}/
                    {rl.progress.totalEpisodes.toLocaleString()}
                    {rl.progress.epsilon != null && ` · ε ${rl.progress.epsilon.toFixed(2)}`}
                  </span>
                )}
              </div>
            </div>
          </ModelSlot>
        </div>

        <div className="flex min-h-0 min-w-0 flex-col md:overflow-y-auto md:[grid-area:work-b]">
          <ModelSlot
            step={3}
            label={env === "cartpole" ? "The policy" : "The table"}
            className="flex min-h-0 flex-1 flex-col [&>*:last-child]:min-h-0 [&>*:last-child]:flex-1"
          >
            <OutputPanel
              title={env === "cartpole" ? "CartPole" : "Q-table over the grid"}
              meta={
                rl.result
                  ? `${rl.result.episodes.toLocaleString()} episodes · ${rl.result.steps.toLocaleString()} steps · ${Math.round(rl.result.elapsedMs)} ms${rl.result.stopped ? " · stopped" : ""}`
                  : undefined
              }
              description={
                <>
                  {env === "cartpole"
                    ? "A policy network, 4 → hidden → 2, samples left or right from the cart's state. Each episode's return is the number of steps the pole stayed up (500 is the cap, and a truncation, not a win). The raw trace is drawn unsmoothed on purpose: REINFORCE's noise is the argument for Actor-Critic."
                    : "Each arrow is the action the agent would take from that cell if it stopped exploring; the shading is how much that cell is worth. Faint arrows are ties — every action still worth the same, so the arrow is only the tie-break rule, not something learned."}
                  {/* Read before a result exists — a caveat that arrives once you
                      already believe the answer comes too late — and then moved
                      below the result, where it no longer squeezes it. */}
                  {render == null && <RlNotes />}
                </>
              }
              running={rl.training && !render}
              runningLabel="Starting…"
              empty={
                <span data-testid="rl-empty">
                  {env === "cartpole"
                    ? "Press Train, or Run both for the head-to-head. The policy starts near a coin flip, and the pole falls within a few dozen steps until it learns."
                    : "Press Train. The agent starts knowing nothing — every cell worth zero — and the arrows turn toward the goal as the reward it finds propagates backward through the table."}
                </span>
              }
            >
              {/* One child, not two conditionals: OutputPanel reads an array as a
                  result and would never show its empty state. */}
              {render == null ? null : render.kind === "grid" ? (
                <div className="space-y-4">
                  <GridCanvas map={render.map} q={render.q} agent={render.agent} />
                  <Scoreboard
                    returns={rl.returns}
                    q={render.q}
                    request={rl.training && algorithm === "q-learning" ? qRequest : (lastQ ?? qRequest)}
                    map={shownMap}
                    final={!rl.training}
                  />
                  <ReturnChart
                    returns={rl.returns}
                    window={WINDOW}
                    yLabel="reached the goal"
                    comparisons={comparisonsFor(rl.history, rl.training && algorithm === "q-learning" ? qRequest : (lastQ ?? qRequest))}
                  />
                  <p className="text-xs leading-snug text-muted-foreground">
                    Each episode&apos;s return is 1 if the agent reached the goal
                    and 0 otherwise, drawn raw — the running mean over it is the
                    success rate <em>while exploring</em>, which is why it tops
                    out below 100%. Dashed lines are earlier runs from this visit.
                  </p>
                  <PolicyTable q={render.q} map={render.map} />
                  <History history={rl.history.filter((r) => r.request.algorithm === "q-learning")} onClear={rl.clearHistory} />
                  <RlNotes />
                </div>
              ) : render.kind === "cartpole" ? (
                <div className="space-y-4">
                  <CartPoleCanvas state={render.state} />
                  <PgScoreboard
                    returns={rl.returns}
                    diagnostics={render.diagnostics}
                    algorithm={(rl.training ? request : lastRequest ?? request).algorithm}
                  />
                  <ReturnChart
                    returns={rl.returns}
                    window={PG_WINDOW}
                    yLabel="steps balanced"
                    label={ALGORITHMS.find((a) => a.id === (rl.training ? request : lastRequest ?? request).algorithm)?.label}
                    comparisons={pgComparisons(rl.history, rl.training ? request : (lastRequest ?? request))}
                  />
                  <p className="text-xs leading-snug text-muted-foreground">
                    Faint lines are every episode, raw; solid and dashed lines are
                    the mean of the last {PG_WINDOW}. The other algorithm&apos;s
                    run at the same seed, if there is one, is drawn beside this
                    one — raw and smoothed alike, so neither is flattered.
                  </p>
                  <SeedSpread history={rl.history} />
                  <PgHistory history={rl.history} onClear={rl.clearHistory} />
                  <RlNotes />
                </div>
              ) : null}
            </OutputPanel>
          </ModelSlot>
        </div>
      </div>
    </div>
  );
}

interface PgSettings {
  hidden: number;
  gamma: number;
  episodes: number;
  normalise: boolean;
  lr: number;
  criticLr: number;
  lambda: number;
}

/** An algorithm's measured defaults (policyGradient.ts), never the last one's. */
function pgDefaults(alg: "reinforce" | "actor-critic"): PgSettings {
  return {
    hidden: PG_DEFAULTS.hidden,
    gamma: PG_DEFAULTS.gamma,
    episodes: PG_DEFAULTS.episodes,
    normalise: false,
    ...pgRates(alg),
  };
}

function pgRates(alg: "reinforce" | "actor-critic"): Pick<PgSettings, "lr" | "criticLr" | "lambda"> {
  const d = PG_DEFAULTS[alg];
  return { lr: d.lr, criticLr: d.criticLr, lambda: d.lambda };
}


function pgLabel(r: RlTrainRequest): string {
  return ALGORITHMS.find((a) => a.id === r.algorithm)?.label ?? r.algorithm;
}

function pgComparisons(history: RlRunRecord[], shown: RlTrainRequest) {
  // The head-to-head: the *other* algorithm at the *same* seed, raw trace and
  // all. Other seeds are summarised by the spread below, not drawn — a chart of
  // eight noisy curves shows nothing.
  if (shown.algorithm === "q-learning") return [];
  return history
    .filter(
      (r) =>
        r.request.algorithm !== "q-learning" &&
        r.request.algorithm !== shown.algorithm &&
        r.request.seed === shown.seed,
    )
    .slice(-1)
    .map((r) => ({ label: `${pgLabel(r.request)}, seed ${r.request.seed}`, returns: r.returns, raw: true }));
}

function runLabel(r: QLearningRequest): string {
  return `seed ${r.seed} · ε ${r.epsilon}${r.decay ? "↘" : ""} · α ${r.alpha} · ${r.map}${r.slippery ? " slippery" : ""}`;
}

function comparisonsFor(history: RlRunRecord[], shown: QLearningRequest) {
  const q = history.filter(
    (r): r is RlRunRecord & { request: QLearningRequest } => r.request.algorithm === "q-learning",
  );
  // Only runs on the same grid: a 4×4 curve over 1 000 episodes drawn against
  // an 8×8 one over 20 000 compares two different problems. And never the run
  // on screen itself, which is the newest entry once it finishes.
  return q
    .filter(
      (r) =>
        r.key !== runKey(shown) &&
        r.request.map === shown.map &&
        r.request.slippery === shown.slippery,
    )
    .slice(-3)
    .map((r) => ({ label: runLabel(r.request), returns: r.returns }));
}

function EnvPicker({
  map,
  slippery,
  onChoose,
  disabled,
}: {
  map: GridMapId;
  slippery: boolean;
  onChoose: (map: GridMapId, slippery: boolean) => void;
  disabled: boolean;
}) {
  return (
    <div className="space-y-2">
      <p className="text-xs font-medium">FrozenLake · tabular Q-learning</p>
      <div className="flex flex-wrap gap-1.5">
        {(["4x4", "8x8"] as GridMapId[]).map((m) => (
          <Button
            key={m}
            size="sm"
            variant={m === map ? "default" : "outline"}
            aria-pressed={m === map}
            disabled={disabled}
            onClick={() => onChoose(m, slippery)}
          >
            {m.replace("x", " × ")}
          </Button>
        ))}
        <Button
          size="sm"
          variant={slippery ? "secondary" : "outline"}
          aria-pressed={slippery}
          disabled={disabled}
          onClick={() => onChoose(map, !slippery)}
        >
          {slippery ? "Slippery" : "Not slippery"}
        </Button>
      </div>
      <p className="text-xs leading-snug text-muted-foreground">
        Transcribed from Gymnasium&apos;s <code>FrozenLake-v1</code>: the same
        maps, the same action order, and the same slip rule — on ice the agent
        moves the way it meant to a third of the time, and sideways otherwise,
        never backwards. Gymnasium&apos;s default is slippery; this page starts
        on dry ground so the arrows settle where you can check them.
      </p>
    </div>
  );
}

function Hyperparameters(props: {
  alpha: number;
  onAlpha: (v: number) => void;
  gamma: number;
  onGamma: (v: number) => void;
  epsilon: number;
  onEpsilon: (v: number) => void;
  decay: boolean;
  onDecay: (v: boolean) => void;
  episodes: number;
  onEpisodes: (v: number) => void;
  seed: number;
  onSeed: (v: number) => void;
  disabled: boolean;
}) {
  const { disabled } = props;
  return (
    <div className="space-y-3 border-t pt-3">
      <Slider id="rl-alpha" label="Learning rate α" value={props.alpha} min={0.01} max={1} step={0.01} onChange={props.onAlpha} disabled={disabled} />
      <Slider id="rl-gamma" label="Discount γ" value={props.gamma} min={0.5} max={0.99} step={0.01} onChange={props.onGamma} disabled={disabled} />
      <Slider id="rl-epsilon" label="Exploration ε" value={props.epsilon} min={0} max={1} step={0.05} onChange={props.onEpsilon} disabled={disabled} />
      <label className="flex items-center gap-2 text-xs">
        <input
          type="checkbox"
          checked={props.decay}
          disabled={disabled}
          onChange={(e) => props.onDecay(e.target.checked)}
        />
        Decay ε to {EPSILON_FLOOR} over the first 80% of episodes
      </label>
      <p className="text-xs leading-snug text-muted-foreground" data-testid="epsilon-lesson">
        <strong>Set ε to 0 and the agent never finds the goal.</strong> Every
        cell starts worth zero, ties go to the first action — left, into the
        wall — and a greedy agent learns nothing from a wall it already valued
        at zero. That is the exploration lesson in one control. A small
        <em> constant</em> ε barely does better here, which is why the decay is
        on by default.
      </p>
      <NumberField id="rl-episodes" label="Episodes" value={props.episodes} min={10} max={MAX_EPISODES} onChange={props.onEpisodes} disabled={disabled} />
      <NumberField id="rl-seed" label="Seed" value={props.seed} min={0} max={2 ** 31 - 1} onChange={props.onSeed} disabled={disabled} />
      <p className="text-xs leading-snug text-muted-foreground">
        Two runs at different seeds differ more than you would guess — the seed
        is the most important control on an RL page. Train, change it, train
        again: each run is kept below for comparison.
      </p>
    </div>
  );
}

function SpeedDial({ speed, onSpeed }: { speed: number | null; onSpeed: (s: number | null) => void }) {
  return (
    <div className="space-y-1.5">
      <p className="text-sm font-medium">Simulation speed</p>
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Simulation speed">
        {SPEEDS.map((s) => (
          <Button
            key={String(s)}
            size="sm"
            variant={s === speed ? "secondary" : "outline"}
            aria-pressed={s === speed}
            className="h-7 px-2 text-xs"
            onClick={() => onSpeed(s)}
          >
            {s == null ? "max" : `${s.toLocaleString()} steps/s`}
          </Button>
        ))}
      </div>
      <p className="text-xs leading-snug text-muted-foreground">
        A <strong>simulation</strong> dial, not a frame rate: the canvas is
        repainted at most 60 times a second however fast the loop runs. At
        &ldquo;max&rdquo; a thousand episodes take milliseconds — slow it down to
        watch the agent walk. It takes effect on a run in progress.
      </p>
    </div>
  );
}

function LiveEpsilon({
  training,
  value,
  onChange,
}: {
  training: boolean;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1">
      <label htmlFor="rl-live-epsilon" className="flex items-baseline justify-between text-sm font-medium">
        Live ε
        <span className="font-mono text-xs text-muted-foreground tabular-nums">{value.toFixed(2)}</span>
      </label>
      <input
        id="rl-live-epsilon"
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={value}
        disabled={!training}
        onChange={(e) => onChange(Number(e.target.value))}
        className="block w-full"
      />
      <p className="text-xs leading-snug text-muted-foreground">
        {training
          ? "Changes how the agent behaves from now on, and takes over from the decay schedule. It does not restart the run."
          : "Live while a run is in progress — it changes the behaviour of that run, which is not a re-run."}
      </p>
    </div>
  );
}

function Scoreboard({
  returns,
  q,
  request,
  map,
  final,
}: {
  returns: readonly number[];
  q: Float32Array;
  request: QLearningRequest;
  map: GridMapId;
  final: boolean;
}) {
  // Derived on the main thread from the table in hand: a 64-cell value
  // iteration and a hundred greedy walks are microseconds, and neither is a
  // re-run of anything.
  const check = useMemo(() => {
    if (!final) return null;
    const env = new GridWorld({ map, slippery: request.slippery });
    const table = new QTable(env.nStates, env.nActions);
    table.q.set(q);
    const vi = valueIteration(env, request.gamma);
    const optimal = optimalActions(vi.q, env.nStates, env.nActions, 1e-6);
    let agree = 0;
    let cells = 0;
    for (let s = 0; s < env.nStates; s++) {
      if (env.isTerminal(s)) continue;
      cells++;
      if (optimal[s].includes(table.greedy(s))) agree++;
    }
    let reached = 0;
    const trials = 100;
    for (let k = 0; k < trials; k++) reached += greedyRollout(env, table, k === 0 ? 12345 : undefined).return;
    return { agree, cells, greedy: reached / trials, optimalValue: vi.v[env.start] };
  }, [final, q, map, request.slippery, request.gamma]);

  const recent = returns.slice(-WINDOW);
  const rate = recent.length > 0 ? recent.reduce((a, b) => a + b, 0) / recent.length : null;

  return (
    <dl className="grid grid-cols-1 gap-x-4 gap-y-1 text-xs sm:grid-cols-2" data-testid="rl-scoreboard">
      <div className="flex justify-between gap-2">
        <dt className="text-muted-foreground">Success, last {Math.min(WINDOW, recent.length)} episodes</dt>
        <dd className="font-mono tabular-nums" data-testid="success-rate">
          {rate == null ? "—" : `${Math.round(rate * 100)}%`}
        </dd>
      </div>
      {check && (
        <>
          <div className="flex justify-between gap-2">
            <dt className="text-muted-foreground">Greedy policy, 100 fresh tries</dt>
            <dd className="font-mono tabular-nums" data-testid="greedy-rate">
              {Math.round(check.greedy * 100)}%
            </dd>
          </div>
          <div className="flex justify-between gap-2 sm:col-span-2">
            <dt className="text-muted-foreground">
              Agrees with value iteration&apos;s exact optimal policy
            </dt>
            <dd className="font-mono tabular-nums" data-testid="policy-agreement">
              {check.agree} of {check.cells} cells
            </dd>
          </div>
          <p className="text-xs leading-snug text-muted-foreground sm:col-span-2">
            Value iteration solves this MDP exactly — it is how the module&apos;s
            own tests check the learner, because a wrong update rule still draws
            a rising curve. Under the best possible policy the start is worth{" "}
            {check.optimalValue.toFixed(3)} at γ = {request.gamma}.
          </p>
        </>
      )}
    </dl>
  );
}

/** The greedy action per cell as text — for a screen reader, and for a test. */
function PolicyTable({ q, map }: { q: Float32Array; map: GridMapId }) {
  const desc = MAPS[map];
  const cols = desc[0].length;
  const glyph = ["←", "↓", "→", "↑"];
  return (
    <details className="text-xs">
      <summary className="cursor-pointer text-muted-foreground">The policy as text</summary>
      <table className="mt-2 font-mono" data-testid="policy-grid">
        <caption className="sr-only">Greedy action per cell</caption>
        <tbody>
          {desc.map((row, r) => (
            <tr key={r}>
              {row.split("").map((letter, c) => {
                const s = r * cols + c;
                const base = s * 4;
                let a = 0;
                for (let k = 1; k < 4; k++) if (q[base + k] > q[base + a]) a = k;
                const terminal = letter === "H" || letter === "G";
                return (
                  <td
                    key={c}
                    className="size-6 text-center"
                    data-state={s}
                    data-action={terminal ? undefined : ACTION_NAMES[a]}
                  >
                    {terminal ? letter : glyph[a]}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

function History({ history, onClear }: { history: RlRunRecord[]; onClear: () => void }) {
  if (history.length === 0) return null;
  return (
    <div className="space-y-1.5 border-t pt-3" data-testid="rl-history">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium">Runs this visit</p>
        <Button size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={onClear}>
          Clear
        </Button>
      </div>
      <ul className="space-y-0.5 text-xs">
        {history.map((r) => {
          const tail = r.returns.slice(-WINDOW);
          const rate = tail.length ? tail.reduce((a, b) => a + b, 0) / tail.length : 0;
          return (
            <li key={r.key} className="flex justify-between gap-2 font-mono tabular-nums">
              <span className="truncate">
                {runLabel(r.request as QLearningRequest)}
                {r.result.epsilonChanged ? " · ε changed live" : ""}
                {r.result.stopped ? " · stopped" : ""}
              </span>
              <span>{Math.round(rate * 100)}%</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Slider({
  id,
  label,
  value,
  min,
  max,
  step,
  onChange,
  disabled,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (n: number) => void;
  disabled: boolean;
}) {
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="flex items-baseline justify-between text-xs font-medium">
        {label}
        <span className="font-mono text-xs text-muted-foreground tabular-nums">{value}</span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="block w-full"
      />
    </div>
  );
}

function NumberField({
  id,
  label,
  value,
  min,
  max,
  onChange,
  disabled,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (n: number) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-2">
      <label htmlFor={id} className="text-xs font-medium">
        {label}
      </label>
      <input
        id={id}
        type="number"
        min={min}
        max={max}
        value={value}
        disabled={disabled}
        onChange={(e) => {
          const n = Math.round(Number(e.target.value));
          if (Number.isFinite(n)) onChange(Math.min(max, Math.max(min, n)));
        }}
        className="w-28 rounded-md border bg-background px-2 py-1 text-right font-mono text-xs tabular-nums"
      />
    </div>
  );
}

function AlgorithmPicker({
  env,
  algorithm,
  onEnv,
  onAlgorithm,
  disabled,
}: {
  env: EnvId;
  algorithm: PageAlgorithm;
  onEnv: (env: EnvId) => void;
  onAlgorithm: (alg: PageAlgorithm) => void;
  disabled: boolean;
}) {
  return (
    <div className="space-y-2" data-testid="algorithm-picker">
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Environment">
        {(
          [
            ["frozenlake", "FrozenLake"],
            ["cartpole", "CartPole"],
          ] as [EnvId, string][]
        ).map(([id, label]) => (
          <Button
            key={id}
            size="sm"
            variant={id === env ? "default" : "outline"}
            aria-pressed={id === env}
            disabled={disabled}
            onClick={() => onEnv(id)}
          >
            {label}
          </Button>
        ))}
      </div>
      <div className="flex flex-col gap-1.5">
        {ALGORITHMS.map((a) => {
          const reason = PAIRINGS[env][a.id];
          const selected = a.id === algorithm;
          return (
            <Button
              key={a.id}
              variant={selected ? "default" : "outline"}
              aria-pressed={selected}
              disabled={disabled || reason != null}
              onClick={() => onAlgorithm(a.id)}
              className="h-auto w-full flex-col items-start gap-0.5 px-3 py-2 text-left whitespace-normal"
            >
              <span className="text-sm font-medium">{a.label}</span>
              <span
                className={cn(
                  "text-xs leading-snug font-normal",
                  selected ? "text-primary-foreground/75" : "text-muted-foreground",
                )}
                data-testid={reason ? `pairing-reason-${a.id}` : undefined}
              >
                {reason ?? a.note}
              </span>
            </Button>
          );
        })}
      </div>
    </div>
  );
}

function PgHyperparameters({
  algorithm,
  value,
  onChange,
  seed,
  onSeed,
  disabled,
}: {
  algorithm: "reinforce" | "actor-critic";
  value: PgSettings;
  onChange: (v: PgSettings) => void;
  seed: number;
  onSeed: (v: number) => void;
  disabled: boolean;
}) {
  const set = <K extends keyof PgSettings>(k: K) => (v: PgSettings[K]) => onChange({ ...value, [k]: v });
  return (
    <div className="space-y-3 border-t pt-3">
      <div className="space-y-1">
        <p className="text-xs font-medium">Hidden width</p>
        <div className="flex gap-1.5">
          {[16, 32, 64, 128].map((h) => (
            <Button
              key={h}
              size="sm"
              variant={h === value.hidden ? "secondary" : "outline"}
              aria-pressed={h === value.hidden}
              className="h-7 px-2 text-xs"
              disabled={disabled}
              onClick={() => set("hidden")(h)}
            >
              {h}
            </Button>
          ))}
        </div>
      </div>
      <Slider id="pg-lr" label="Policy learning rate" value={value.lr} min={0.0005} max={0.03} step={0.0005} onChange={set("lr")} disabled={disabled} />
      {algorithm === "actor-critic" && (
        <>
          <Slider id="pg-critic-lr" label="Critic learning rate" value={value.criticLr} min={0.001} max={0.1} step={0.001} onChange={set("criticLr")} disabled={disabled} />
          <Slider id="pg-lambda" label="GAE λ" value={value.lambda} min={0} max={1} step={0.05} onChange={set("lambda")} disabled={disabled} />
        </>
      )}
      <Slider id="pg-gamma" label="Discount γ" value={value.gamma} min={0.9} max={0.999} step={0.001} onChange={set("gamma")} disabled={disabled} />
      {algorithm === "reinforce" && (
        <div className="space-y-1">
          <label className="flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={value.normalise}
              disabled={disabled}
              onChange={(e) => set("normalise")(e.target.checked)}
            />
            Normalise returns per episode
          </label>
          <p className="text-xs leading-snug text-muted-foreground" data-testid="normalise-note">
            Off by default, and that is a choice: standardising the returns is a
            crude baseline, and it removes most of the variance this comparison
            is about — measured, it lifts REINFORCE from about 260 to about 430
            steps. Turn it on to see how much of Actor-Critic&apos;s advantage a
            one-line trick recovers.
          </p>
        </div>
      )}
      <NumberField id="pg-episodes" label="Episodes" value={value.episodes} min={10} max={5000} onChange={set("episodes")} disabled={disabled} />
      <NumberField id="rl-seed" label="Seed" value={seed} min={0} max={2 ** 31 - 1} onChange={onSeed} disabled={disabled} />
      <p className="text-xs leading-snug text-muted-foreground">
        Each algorithm opens with its own measured learning rate. Given
        Actor-Critic&apos;s, REINFORCE collapses on some seeds — a comparison at
        a rate that suits only one of them would say nothing about either.
      </p>
    </div>
  );
}

/** §3.4 and §3.5 of the roadmap: the two things this page does not do, and why. */
function RlNotes() {
  return (
    <span className="mt-2 block space-y-1.5 text-xs text-muted-foreground" data-testid="rl-notes">
      <span className="block">
        <strong>Not here: Decision Transformer.</strong> It treats RL as sequence
        modelling — a small GPT reads (return-to-go, state, action) triples and
        predicts the next action, the way the trajectories above could be read.
        It is unavailable for two separate reasons: the checkpoint publishes no
        ONNX export (<code>edbeeching/decision-transformer-gym-hopper-medium</code>{" "}
        ships only <code>pytorch_model.bin</code>, checked 2026-09-26), and its
        environment is MuJoCo, which has no browser equivalent.
      </span>
      <span className="block">
        <strong>Not here: RL for language models.</strong> There the policy is an
        LLM, the reward model is another, and one optimisation step is a cluster
        job. Its product is visible on{" "}
        <a className="underline" href="/text-generation">
          /text-generation
        </a>
        : GPT-2 has no instruction tuning at all, and SmolLM2-360M-Instruct was
        tuned with supervised fine-tuning and then <strong>DPO</strong> on
        UltraFeedback — a preference method, not PPO-style RLHF, which is what
        its model card says.
      </span>
    </span>
  );
}

function tailMean(returns: readonly number[], n: number): number | null {
  const tail = returns.slice(-n);
  return tail.length ? tail.reduce((a, b) => a + b, 0) / tail.length : null;
}

function PgScoreboard({
  returns,
  diagnostics,
  algorithm,
}: {
  returns: readonly number[];
  diagnostics: { meanAbsWeight: number; criticLoss: number | null } | null;
  algorithm: RlAlgorithm;
}) {
  const mean = tailMean(returns, PG_WINDOW);
  return (
    <dl className="grid grid-cols-1 gap-x-4 gap-y-1 text-xs sm:grid-cols-2" data-testid="pg-scoreboard">
      <div className="flex justify-between gap-2">
        <dt className="text-muted-foreground">Mean return, last {PG_WINDOW}</dt>
        <dd className="font-mono tabular-nums" data-testid="pg-mean-return">
          {mean == null ? "—" : mean.toFixed(0)}
        </dd>
      </div>
      <div className="flex justify-between gap-2">
        <dt className="text-muted-foreground">Episodes</dt>
        <dd className="font-mono tabular-nums">{returns.length}</dd>
      </div>
      {diagnostics && (
        <>
          <div className="flex justify-between gap-2">
            <dt className="text-muted-foreground">
              Weight on ∇log π, mean {algorithm === "actor-critic" ? "|Aₜ|" : "|Gₜ|"}
            </dt>
            <dd className="font-mono tabular-nums" data-testid="pg-weight">
              {diagnostics.meanAbsWeight.toFixed(2)}
            </dd>
          </div>
          {diagnostics.criticLoss != null && (
            <div className="flex justify-between gap-2">
              <dt className="text-muted-foreground">Critic loss</dt>
              <dd className="font-mono tabular-nums">{diagnostics.criticLoss.toFixed(3)}</dd>
            </div>
          )}
          <p className="text-xs leading-snug text-muted-foreground sm:col-span-2">
            Where the variance went: REINFORCE scales each step&apos;s gradient
            by the whole return that followed it, which is large and different
            every episode. Actor-Critic scales it by how much better the step
            went than the critic expected — near zero once the critic is good.
          </p>
        </>
      )}
    </dl>
  );
}

/** Final returns per algorithm across the seeds trained this visit — the claim is a spread. */
function SeedSpread({ history }: { history: RlRunRecord[] }) {
  const rows = (["reinforce", "actor-critic"] as const)
    .map((alg) => {
      const finals = history
        .filter((r) => r.request.algorithm === alg && !r.result.stopped)
        .map((r) => ({ seed: r.request.seed, value: tailMean(r.returns, PG_WINDOW) ?? 0 }));
      if (finals.length === 0) return null;
      const values = finals.map((f) => f.value);
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
      return { alg, finals, mean, sd, min: Math.min(...values), max: Math.max(...values) };
    })
    .filter((r) => r != null);
  if (rows.length === 0) return null;
  return (
    <div className="space-y-1.5 border-t pt-3" data-testid="seed-spread">
      <p className="text-xs font-medium">Across seeds</p>
      <table className="w-full text-xs">
        <caption className="sr-only">Final mean return per algorithm, across the seeds trained</caption>
        <thead>
          <tr className="text-left text-muted-foreground">
            <th className="font-normal">Algorithm</th>
            <th className="font-normal">Seeds</th>
            <th className="font-normal">Mean</th>
            <th className="font-normal">Spread (sd · range)</th>
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">
          {rows.map((r) => (
            <tr key={r.alg} data-testid={`spread-${r.alg}`}>
              <td className="font-sans">{pgLabel({ algorithm: r.alg } as RlTrainRequest)}</td>
              <td>{r.finals.map((f) => f.seed).join(", ")}</td>
              <td>{r.mean.toFixed(0)}</td>
              <td>
                {r.finals.length > 1 ? `${r.sd.toFixed(0)} · ${r.min.toFixed(0)}–${r.max.toFixed(0)}` : "one seed"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs leading-snug text-muted-foreground">
        The claim this page makes is about the <strong>spread</strong>, not the
        winner of any one seed. Actor-Critic beating REINFORCE on a given seed is
        not something RL promises; a narrower spread over several seeds is what
        the baseline buys. Run both at a few seeds to fill this in.
      </p>
    </div>
  );
}

function PgHistory({ history, onClear }: { history: RlRunRecord[]; onClear: () => void }) {
  const runs = history.filter((r) => r.request.algorithm !== "q-learning");
  if (runs.length === 0) return null;
  return (
    <div className="space-y-1.5 border-t pt-3" data-testid="pg-history">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium">Runs this visit</p>
        <Button size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={onClear}>
          Clear
        </Button>
      </div>
      <ul className="space-y-0.5 text-xs">
        {runs.map((r) => (
          <li key={r.key} className="flex justify-between gap-2 font-mono tabular-nums">
            <span className="truncate">
              {pgLabel(r.request)} · seed {r.request.seed}
              {r.result.stopped ? " · stopped" : ""}
            </span>
            <span>{(tailMean(r.returns, PG_WINDOW) ?? 0).toFixed(0)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
