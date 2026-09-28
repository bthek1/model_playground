// `/robotics`'s behaviour-cloning entry: the multimodality failure, with the
// control that makes it evidence.
//
// The same four bands as the grounding entry beside it in the picker. SELECT is
// the entry plus the demonstration set (mix, count, obstacle), the network and
// the seed — all choices. LOAD resolves at once: there is nothing to download,
// and the band says so rather than disappearing, so the route keeps one rhythm
// across its two entries. RUN is the workspace with the demonstrations drawn
// **live** — generating them is pure and cheap, so the preview updates as the
// choices move — and Train/Stop. Training does not update live; only Train
// spends.
//
// OUTPUT is the rollout over the demonstrations, the action field, and a
// verdict that distinguishes reached / collided / stalled. Once both mixes have
// been trained at one seed they sit side by side, because the failure alone is
// indistinguishable from a bug in this page's own code.

import { Bot, Loader2, Play, Square } from "lucide-react";
import { type ReactNode, useMemo, useState } from "react";

import { ErrorNote } from "@/components/model/ErrorNote";
import { ModelPage } from "@/components/model/ModelPage";
import { OutputPanel } from "@/components/model/OutputPanel";
import { Button } from "@/components/ui/button";
import { type RlRunRecord, useRlTraining } from "@/hooks/useRlTraining";
import {
  actionField,
  type ClonedPolicy,
  rollout,
  type Rollout,
  unpackPolicy,
} from "@/rl/behaviourCloning";
import { type DemoMix, demonstrations } from "@/rl/demos";
import { makeScene } from "@/rl/envs/reacher2d";
import type { CloningRequest } from "@/rl/types";

import { CLONING_DEFAULTS, SIDE_COLORS, VERDICT } from "./constants";
import { Workspace } from "./Workspace";

function policyOf(record: RlRunRecord): ClonedPolicy | null {
  const r = record.result.render;
  return r?.kind === "cloning" ? unpackPolicy(r.policy, r.hidden) : null;
}

export function CloningPage({ select }: { select: ReactNode }) {
  const rl = useRlTraining();
  const [mix, setMix] = useState<DemoMix>("both");
  const [demos, setDemos] = useState(CLONING_DEFAULTS.demos);
  const [obstacle, setObstacle] = useState(CLONING_DEFAULTS.obstacle);
  const [hidden, setHidden] = useState(CLONING_DEFAULTS.hidden);
  const [seed, setSeed] = useState(CLONING_DEFAULTS.seed);

  const request: CloningRequest = {
    algorithm: "behaviour-cloning",
    mix,
    demos,
    obstacle,
    hidden,
    epochs: CLONING_DEFAULTS.epochs,
    lr: CLONING_DEFAULTS.lr,
    seed,
  };

  // The preview: the same pure generator, the same seed, as the worker's.
  const scene = useMemo(() => makeScene(obstacle), [obstacle]);
  const preview = useMemo(() => demonstrations(scene, mix, demos, seed), [scene, mix, demos, seed]);

  // The run on screen: live while training, or the last one's final policy.
  const last = rl.history[rl.history.length - 1];
  const shownRequest = (rl.training ? request : (last?.request as CloningRequest | undefined)) ?? null;
  const shownScene = useMemo(
    () => (shownRequest ? makeScene(shownRequest.obstacle) : scene),
    [shownRequest?.obstacle, scene], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const shownDemos = useMemo(
    () =>
      shownRequest
        ? demonstrations(shownScene, shownRequest.mix, shownRequest.demos, shownRequest.seed)
        : preview,
    [shownRequest?.mix, shownRequest?.demos, shownRequest?.seed, shownScene, preview], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const render = rl.render?.kind === "cloning" ? rl.render : null;
  const policy = useMemo(() => (render ? unpackPolicy(render.policy, render.hidden) : null), [render]);
  const shownRollout = useMemo(() => (policy ? rollout(policy, shownScene) : null), [policy, shownScene]);
  const field = useMemo(() => (policy ? actionField(policy, shownScene) : null), [policy, shownScene]);

  // Both mixes at one seed and one scene: the failure beside its control.
  const pair = useMemo(() => {
    if (!shownRequest) return null;
    const find = (m: DemoMix) =>
      rl.history.find((r) => {
        const q = r.request as CloningRequest;
        return (
          q.algorithm === "behaviour-cloning" &&
          q.mix === m &&
          q.seed === shownRequest.seed &&
          q.obstacle === shownRequest.obstacle &&
          q.demos === shownRequest.demos &&
          q.hidden === shownRequest.hidden &&
          !r.result.stopped
        );
      });
    const out: Partial<Record<DemoMix, Rollout>> = {};
    for (const m of ["both", "one"] as const) {
      const rec = find(m);
      const p = rec && policyOf(rec);
      if (p) out[m] = rollout(p, makeScene(shownRequest.obstacle));
    }
    return out;
  }, [rl.history, shownRequest?.seed, shownRequest?.obstacle, shownRequest?.demos, shownRequest?.hidden]); // eslint-disable-line react-hooks/exhaustive-deps

  const lastLoss = rl.returns.length ? rl.returns[rl.returns.length - 1] : null;

  return (
    <ModelPage
      icon={Bot}
      title="Robotics"
      description={
        <>
          Behaviour cloning on a toy two-link arm: copy a demonstrator that goes
          round an obstacle, and watch what copying does when the demonstrator
          went both ways. Trained in this tab — nothing is downloaded.
        </>
      }
      labels={{ select: "Entry & demonstrations", load: "Load", run: "Demonstrations", output: "The cloned policy" }}
      select={
        <div className="space-y-3">
          {select}
          <CloningSettings
            mix={mix}
            onMix={setMix}
            demos={demos}
            onDemos={setDemos}
            obstacle={obstacle}
            onObstacle={setObstacle}
            hidden={hidden}
            onHidden={setHidden}
            seed={seed}
            onSeed={setSeed}
            disabled={rl.training}
          />
        </div>
      }
      load={
        <p className="text-xs text-muted-foreground" data-testid="model-ready">
          Nothing to download — the policy is a {hidden}-unit network that
          starts from random weights and is trained on this machine&apos;s CPU
          when you press Train. Ready.
        </p>
      }
      run={
        <div className="flex min-h-0 flex-1 flex-col gap-3">
          <Workspace scene={scene} demos={preview} testId="demo-preview" />
          <p className="flex flex-wrap gap-x-3 text-xs text-muted-foreground" data-testid="demo-summary">
            <span>
              <span className="inline-block size-2 rounded-full" style={{ background: SIDE_COLORS.above }} />{" "}
              {preview.filter((d) => d.side === "above").length} above
            </span>
            <span>
              <span className="inline-block size-2 rounded-full" style={{ background: SIDE_COLORS.below }} />{" "}
              {preview.filter((d) => d.side === "below").length} below
            </span>
            <span>{preview.reduce((s, d) => s + d.actions.length / 2, 0)} (state, action) pairs</span>
          </p>
          <p className="text-xs leading-snug text-muted-foreground">
            The demonstrator is a <strong>script</strong>, not a learned expert:
            it heads for a waypoint just clear of the obstacle on its side, then
            for the target, and turns that into joint speeds with the arm&apos;s
            Jacobian. The drawing updates as you change the set; nothing trains
            until you press Train. Only the hand collides — the links pass
            through the obstacle, a simplification of this toy.
          </p>
          <ErrorNote message={rl.error} />
          <div className="sticky bottom-0 mt-auto flex flex-wrap items-center gap-2 border-t bg-background/80 pt-3 backdrop-blur">
            <Button onClick={() => rl.start(request, null)} disabled={rl.training} data-testid="train-button">
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
              <span className="font-mono text-xs text-muted-foreground tabular-nums" data-testid="rl-progress">
                epoch {rl.progress.episode}/{rl.progress.totalEpisodes}
                {lastLoss != null && ` · loss ${lastLoss.toFixed(3)}`}
              </span>
            )}
          </div>
        </div>
      }
      output={
        <OutputPanel
          title="The cloned policy"
          description={
            <span data-testid="cloning-fix">
              Trained on both ways round, a mean-squared-error policy learns the{" "}
              <strong>average</strong> of the two — and the average of
              &ldquo;over&rdquo; and &ldquo;under&rdquo; is &ldquo;straight
              through&rdquo;. The fixes model the whole distribution of actions
              instead of its mean: <strong>action chunking</strong> commits to a
              whole short sequence at once, and <strong>diffusion policies</strong>{" "}
              sample one mode rather than blending them. This page names them and
              does not demonstrate either.
            </span>
          }
          running={rl.training && !render}
          runningLabel="Starting…"
          empty="Train on the demonstrations shown. Then switch the mix and train again at the same seed: the failure only means something beside the run that succeeds."
        >
          {render && shownRollout && (
            <div className="space-y-3">
              <Workspace scene={shownScene} demos={shownDemos} rollout={shownRollout} field={field} testId="cloning-result" />
              <p className="text-sm" data-testid="cloning-verdict" data-outcome={shownRollout.outcome}>
                Trained on{" "}
                {shownRequest?.mix === "both" ? "both ways round" : "one way round"}, the
                policy <strong>{VERDICT[shownRollout.outcome]}</strong>
                {rl.training ? " (so far)" : ""}.
              </p>
              <p className="text-xs leading-snug text-muted-foreground">
                The small arrows are the policy&apos;s action field — where it
                would move the hand from each point, with the target fixed.
                Between the two modes, a policy that saw both points straight at
                the obstacle: that is the averaging, drawn.
              </p>
              {pair && (pair.both || pair.one) && (
                <div className="grid grid-cols-2 gap-2 border-t pt-3 text-xs" data-testid="cloning-pair">
                  {(["both", "one"] as const).map((m) => (
                    <div key={m} className="rounded-md border px-2 py-1.5" data-testid={`verdict-${m}`} data-outcome={pair[m]?.outcome ?? ""}>
                      <p className="font-medium">{m === "both" ? "Both ways round" : "One way round"}</p>
                      <p className="text-muted-foreground">
                        {pair[m] ? VERDICT[pair[m]!.outcome] : "not trained yet at this seed"}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </OutputPanel>
      }
    />
  );
}

function CloningSettings(props: {
  mix: DemoMix;
  onMix: (m: DemoMix) => void;
  demos: number;
  onDemos: (n: number) => void;
  obstacle: number;
  onObstacle: (n: number) => void;
  hidden: number;
  onHidden: (n: number) => void;
  seed: number;
  onSeed: (n: number) => void;
  disabled: boolean;
}) {
  const { disabled } = props;
  return (
    <div className="space-y-3 border-t pt-3">
      <div className="space-y-1">
        <p className="text-xs font-medium">Demonstrations go</p>
        <div className="flex gap-1.5" role="group" aria-label="Demonstration mix">
          {(
            [
              ["both", "Both ways round"],
              ["one", "One way round"],
            ] as [DemoMix, string][]
          ).map(([m, label]) => (
            <Button
              key={m}
              size="sm"
              variant={props.mix === m ? "default" : "outline"}
              aria-pressed={props.mix === m}
              disabled={disabled}
              onClick={() => props.onMix(m)}
            >
              {label}
            </Button>
          ))}
        </div>
      </div>
      <Range id="bc-demos" label="Demonstrations" value={props.demos} min={2} max={60} step={2} onChange={props.onDemos} disabled={disabled} />
      <Range id="bc-obstacle" label="Obstacle offset (m)" value={props.obstacle} min={-0.25} max={0.25} step={0.05} onChange={props.onObstacle} disabled={disabled} />
      <p className="text-xs leading-snug text-muted-foreground">
        At 0 the two ways round are equally long. Move it toward either end and
        one way becomes nearly straight — the near-degenerate case.
      </p>
      <div className="space-y-1">
        <p className="text-xs font-medium">Hidden width</p>
        <div className="flex gap-1.5">
          {[16, 32, 64, 128].map((h) => (
            <Button
              key={h}
              size="sm"
              variant={props.hidden === h ? "secondary" : "outline"}
              aria-pressed={props.hidden === h}
              className="h-7 px-2 text-xs"
              disabled={disabled}
              onClick={() => props.onHidden(h)}
            >
              {h}
            </Button>
          ))}
        </div>
      </div>
      <div className="flex items-center justify-between gap-2">
        <label htmlFor="bc-seed" className="text-xs font-medium">
          Seed
        </label>
        <input
          id="bc-seed"
          type="number"
          min={0}
          value={props.seed}
          disabled={disabled}
          onChange={(e) => {
            const n = Math.round(Number(e.target.value));
            if (Number.isFinite(n) && n >= 0) props.onSeed(n);
          }}
          className="w-24 rounded-md border bg-background px-2 py-1 text-right font-mono text-xs tabular-nums"
        />
      </div>
    </div>
  );
}

function Range({
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
