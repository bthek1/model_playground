// Reinforcement Learning — tabular Q-learning over a FrozenLake grid, and the
// first page in the app whose subject is a loop the user watches.
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

import { createFileRoute } from "@tanstack/react-router";
import { Gamepad2, Loader2, Play, Square } from "lucide-react";
import { useMemo, useState } from "react";

import { ErrorNote } from "@/components/model/ErrorNote";
import { ModelSlot } from "@/components/model/ModelPage";
import { OutputPanel } from "@/components/model/OutputPanel";
import { GridCanvas } from "@/components/rl/GridCanvas";
import { ReturnChart } from "@/components/rl/ReturnChart";
import { Button } from "@/components/ui/button";
import { type RlRunRecord, runKey, useRlTraining } from "@/hooks/useRlTraining";
import { cn } from "@/lib/utils";
import { ACTION_NAMES, GridWorld, MAPS } from "@/rl/envs/gridWorld";
import { DEFAULT_SPEED, MAX_EPISODES, SPEEDS } from "@/rl/limits";
import { EPSILON_FLOOR, greedyRollout, QTable } from "@/rl/qLearning";
import type { GridMapId, QLearningRequest } from "@/rl/types";
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

function RlPage() {
  const rl = useRlTraining();

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

  const request: QLearningRequest = {
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

  const train = () => {
    setLiveEpsilon(null);
    rl.start(request, speed);
  };

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
  const shownMap = render?.kind === "grid" ? render.map : map;

  return (
    <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-6 md:h-full md:min-h-0">
      <header className="min-w-0">
        <h1 className="mb-1 flex items-center gap-2 text-2xl font-semibold">
          <Gamepad2 className="size-6" /> Reinforcement Learning
        </h1>
        <p className="text-sm text-muted-foreground">
          Tabular Q-learning on Gymnasium&apos;s FrozenLake, trained in front of
          you. The whole model is a table of {MAPS[map].length ** 2 * 4} numbers
          and the whole algorithm is one update rule — watch the arrows turn
          around as the agent discovers the goal. Nothing is downloaded and
          nothing leaves this tab.
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
              <LiveEpsilon
                training={rl.training}
                value={liveEpsilon ?? rl.progress?.epsilon ?? epsilon}
                onChange={changeLiveEpsilon}
              />
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
                <Button variant="outline" onClick={rl.stop} disabled={!rl.training}>
                  <Square className="size-4" /> Stop
                </Button>
                {rl.progress && (
                  <span
                    className="font-mono text-xs text-muted-foreground tabular-nums"
                    data-testid="rl-progress"
                  >
                    episode {rl.progress.episode.toLocaleString()}/
                    {rl.progress.totalEpisodes.toLocaleString()} · ε{" "}
                    {rl.progress.epsilon.toFixed(2)}
                  </span>
                )}
              </div>
            </div>
          </ModelSlot>
        </div>

        <div className="flex min-h-0 min-w-0 flex-col md:overflow-y-auto md:[grid-area:work-b]">
          <ModelSlot
            step={3}
            label="The table"
            className="flex min-h-0 flex-1 flex-col [&>*:last-child]:min-h-0 [&>*:last-child]:flex-1"
          >
            <OutputPanel
              title="Q-table over the grid"
              meta={
                rl.result
                  ? `${rl.result.episodes.toLocaleString()} episodes · ${rl.result.steps.toLocaleString()} steps · ${Math.round(rl.result.elapsedMs)} ms${rl.result.stopped ? " · stopped" : ""}`
                  : undefined
              }
              description="Each arrow is the action the agent would take from that cell if it stopped exploring; the shading is how much that cell is worth. Faint arrows are ties — every action still worth the same, so the arrow is only the tie-break rule, not something learned."
              running={rl.training && !render}
              runningLabel="Starting…"
              empty={
                <span data-testid="rl-empty">
                  Press Train. The agent starts knowing nothing — every cell worth
                  zero — and the arrows turn toward the goal as the reward it finds
                  propagates backward through the table.
                </span>
              }
            >
              {render?.kind === "grid" && (
                <div className="space-y-4">
                  <GridCanvas map={render.map} q={render.q} agent={render.agent} />
                  <Scoreboard
                    returns={rl.returns}
                    q={render.q}
                    request={rl.training ? request : (lastRequest ?? request)}
                    map={shownMap}
                    final={!rl.training}
                  />
                  <ReturnChart
                    returns={rl.returns}
                    window={WINDOW}
                    yLabel="reached the goal"
                    comparisons={comparisonsFor(rl.history, rl.training ? request : (lastRequest ?? request))}
                  />
                  <p className="text-xs leading-snug text-muted-foreground">
                    Each episode&apos;s return is 1 if the agent reached the goal
                    and 0 otherwise, drawn raw — the running mean over it is the
                    success rate <em>while exploring</em>, which is why it tops
                    out below 100%. Dashed lines are earlier runs from this visit.
                  </p>
                  <PolicyTable q={render.q} map={render.map} />
                  <History history={rl.history} onClear={rl.clearHistory} />
                </div>
              )}
            </OutputPanel>
          </ModelSlot>
        </div>
      </div>
    </div>
  );
}

function runLabel(r: QLearningRequest): string {
  return `seed ${r.seed} · ε ${r.epsilon}${r.decay ? "↘" : ""} · α ${r.alpha} · ${r.map}${r.slippery ? " slippery" : ""}`;
}

function comparisonsFor(history: RlRunRecord[], shown: QLearningRequest) {
  // Only runs on the same grid: a 4×4 curve over 1 000 episodes drawn against
  // an 8×8 one over 20 000 compares two different problems. And never the run
  // on screen itself, which is the newest entry once it finishes.
  return history
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
        A <strong>simulation</strong> dial, not a frame rate: the grid is
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
