// Discrete Maths — walks, paths, and the k-hop neighbourhood.
//
// Theory's last row, and one topic done properly rather than a syllabus: the
// fact that `(Aᵏ)[s][v]` counts the walks of length k from s to v. It is the
// piece of discrete maths the three Graph ML pages stand on — a k-layer GCN
// multiplies by `Âᵏ`, so node s can only have heard from the nodes where row s
// of that power is nonzero — and it is stated there without being shown.
//
// The page puts two computations of the same set side by side. BFS, stepped
// one layer per press, grows the ball of nodes within k hops; the matrix power
// beside it is nonzero on exactly that ball when the walk may pause (A + I),
// and on a *different* set when it may not (A alone): on a bipartite graph a
// walk of length k can only end at k's parity, so the source cannot even hear
// itself at odd k. That is why GCN adds the self-loop, and `/graph`'s kernel
// adds it rather than storing it.
//
// **Three bands, like `/rl` and `/time-series-forecasting`** (model-page-pattern
// §7): there is nothing to download or fit, so the LOAD band is replaced by a
// note saying so. No worker either — BFS and a 34×34 matrix power are
// microseconds, and the route test stubs `Worker` to keep it that way.
//
// Choosing a graph or a source runs nothing and clears the result. **Step** is
// the one control that advances the traversal; the A / A + I switch in OUTPUT
// re-reads the same k and re-derives, spending nothing.

import { useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { ChevronsRight, RotateCcw, StepForward, Waypoints } from "lucide-react";

import { ModelSlot } from "@/components/model/ModelPage";
import { OutputPanel } from "@/components/model/OutputPanel";
import { GraphDiagram } from "@/components/theory/GraphDiagram";
import { WalkMatrix } from "@/components/theory/WalkMatrix";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { bfs } from "@/theory/bfs";
import {
  DEFAULT_GRAPH,
  SMALL_GRAPHS,
  graphById,
  neighbours,
  type SmallGraph,
} from "@/theory/graphs";
import { adjacencyPower, row, support } from "@/theory/walks";

export const Route = createFileRoute("/discrete-maths")({
  component: DiscreteMathsPage,
});

function DiscreteMathsPage() {
  const [graphId, setGraphId] = useState(DEFAULT_GRAPH.id);
  const [source, setSource] = useState(0);
  // Null until the first Step: the OUTPUT band's empty state is a real state.
  const [k, setK] = useState<number | null>(null);
  const [selfLoops, setSelfLoops] = useState(true);

  const graph = graphById(graphId) ?? DEFAULT_GRAPH;
  const traversal = useMemo(() => bfs(neighbours(graph), source), [graph, source]);
  const ecc = traversal.eccentricity;

  const chooseGraph = (id: string) => {
    setGraphId(id);
    setSource(0);
    setK(null);
  };
  const chooseSource = (s: number) => {
    setSource(s);
    setK(null);
  };

  return (
    <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-6 md:h-full md:min-h-0">
      <header className="min-w-0">
        <h1 className="mb-1 flex items-center gap-2 text-2xl font-semibold">
          <Waypoints className="size-6" /> Discrete Maths
        </h1>
        <p className="text-sm text-muted-foreground">
          Walks, paths and the k-hop neighbourhood: step a breadth-first search
          one layer at a time, and watch the power of the adjacency matrix beside
          it pick out the same nodes — or, without the self-loop, not quite the
          same ones. It is the arithmetic a graph neural network&apos;s receptive
          field is made of.
        </p>
      </header>

      {/* Three bands — see the note at the top of this file. */}
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
          <ModelSlot step={1} label="Graph" dense className="min-w-0 md:flex-1 xl:flex-none">
            <div className="space-y-3">
              <div className="flex flex-wrap gap-1.5">
                {SMALL_GRAPHS.map((g) => (
                  <Button
                    key={g.id}
                    size="sm"
                    variant={g.id === graph.id ? "default" : "outline"}
                    aria-pressed={g.id === graph.id}
                    onClick={() => chooseGraph(g.id)}
                  >
                    {g.label}
                  </Button>
                ))}
              </div>
              <p className="text-xs text-muted-foreground tabular-nums" data-testid="graph-facts">
                {graph.n} nodes · {graph.edges.length} edges ·{" "}
                <strong>{graph.bipartite ? "bipartite" : "not bipartite"}</strong>
              </p>
              <p className="text-xs leading-snug text-muted-foreground">{graph.note}</p>
              {graph.source && (
                <p className="text-xs leading-snug text-muted-foreground">
                  Data: {graph.source}.
                </p>
              )}
            </div>
          </ModelSlot>

          <div
            className="min-w-0 space-y-1.5 md:w-80 md:shrink-0 xl:w-auto"
            data-testid="no-load-band"
          >
            <p className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
              Nothing to load
            </p>
            <p className="text-xs leading-snug text-muted-foreground">
              Every other page here has a band for downloading weights or
              fitting a model. This one has nothing to put in it: a breadth-first
              search and a 34×34 matrix power are exact integer arithmetic that
              finishes in microseconds, on the main thread, with no model, no
              worker and no GPU.
            </p>
          </div>
        </div>

        <div className="flex min-h-0 min-w-0 flex-col md:overflow-y-auto md:[grid-area:work-a]">
          <ModelSlot
            step={2}
            label="Traversal"
            className="flex min-h-0 flex-1 flex-col [&>*:last-child]:min-h-0 [&>*:last-child]:flex-1"
          >
            <div className="space-y-4">
              <div className="space-y-1">
                <label htmlFor="bfs-source" className="text-xs font-medium">
                  Source node
                </label>
                <select
                  id="bfs-source"
                  value={source}
                  onChange={(e) => chooseSource(Number(e.target.value))}
                  className="block w-40 rounded-md border bg-background px-2 py-1 text-sm tabular-nums"
                >
                  {Array.from({ length: graph.n }, (_, v) => (
                    <option key={v} value={v}>
                      Node {v} (degree {neighbours(graph)[v].length})
                    </option>
                  ))}
                </select>
                <p className="text-xs leading-snug text-muted-foreground">
                  Choosing a graph or a source runs nothing — it starts a fresh
                  search at step 0.
                </p>
              </div>

              <GraphDiagram graph={graph} source={source} dist={null} k={null} />

              <div className="sticky bottom-0 flex flex-wrap items-center gap-2 border-t bg-background/95 pt-3">
                <Button
                  onClick={() => setK((prev) => Math.min((prev ?? 0) + 1, ecc))}
                  disabled={k != null && k >= ecc}
                >
                  <StepForward className="size-4" /> Step
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setK(ecc)}
                  disabled={k != null && k >= ecc}
                >
                  <ChevronsRight className="size-4" /> Run to end
                </Button>
                <Button variant="ghost" onClick={() => setK(null)} disabled={k == null}>
                  <RotateCcw className="size-4" /> Reset
                </Button>
                <span className="text-xs text-muted-foreground tabular-nums" data-testid="step-count">
                  {k == null ? "step 0" : `step ${k}`} of {ecc}
                </span>
              </div>
            </div>
          </ModelSlot>
        </div>

        <div className="flex min-h-0 min-w-0 flex-col md:overflow-y-auto md:[grid-area:work-b]">
          <ModelSlot
            step={3}
            label="Result"
            className="flex min-h-0 flex-1 flex-col [&>*:last-child]:min-h-0 [&>*:last-child]:flex-1"
          >
            <OutputPanel
              title="The k-hop neighbourhood, two ways"
              description={
                k == null ? undefined : `${graph.label} · from node ${source} · k = ${k}`
              }
              running={false}
              empty={
                <span>
                  Press <strong>Step</strong> to take one breadth-first layer. Each
                  step shows which nodes are within k hops of the source, and the
                  matching power of the adjacency matrix — whose row for the source
                  counts the walks of length k to every node.
                </span>
              }
            >
              {k != null && (
                <Result
                  graph={graph}
                  source={source}
                  k={k}
                  dist={traversal.dist}
                  layers={traversal.layers}
                  eccentricity={ecc}
                  selfLoops={selfLoops}
                  onSelfLoops={setSelfLoops}
                />
              )}
            </OutputPanel>
          </ModelSlot>
        </div>
      </div>
    </div>
  );
}

function Result({
  graph,
  source,
  k,
  dist,
  layers,
  eccentricity,
  selfLoops,
  onSelfLoops,
}: {
  graph: SmallGraph;
  source: number;
  k: number;
  dist: Int32Array;
  layers: number[][];
  eccentricity: number;
  selfLoops: boolean;
  onSelfLoops: (on: boolean) => void;
}) {
  const matrix = useMemo(() => adjacencyPower(graph, k, selfLoops), [graph, k, selfLoops]);
  const walks = row(matrix, graph.n, source);
  const nonzero = support(walks);
  const ball = layers.slice(0, k + 1).flat().sort((a, b) => a - b);
  const frontier = layers[k] ?? [];
  const missing = ball.filter((v) => walks[v] === 0);
  const total = walks.reduce((a, b) => a + b, 0);
  const name = selfLoops ? "(A + I)" : "A";

  return (
    <div className="space-y-4">
      <GraphDiagram graph={graph} source={source} dist={dist} k={k} walks={walks} />
      <p className="text-xs text-muted-foreground">
        The number on a node is its distance from node {source}; the dashed ring
        marks the frontier this step added; the small grey number is the node&apos;s
        id.
      </p>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-4" data-testid="bfs-stats">
        <Stat label="Within k hops" value={`${ball.length} of ${graph.n}`} testId="stat-ball" />
        <Stat label="Frontier" value={String(frontier.length)} testId="stat-frontier" />
        <Stat label={`Walks of length ${k}`} value={total.toLocaleString()} testId="stat-walks" />
        <Stat label="Eccentricity" value={String(eccentricity)} testId="stat-ecc" />
      </dl>

      {k === eccentricity && (
        <p className="text-xs text-muted-foreground" data-testid="bfs-done">
          BFS is finished: every node is within {eccentricity} hops of node {source},
          so a {eccentricity}-layer GNN at this node reads the whole graph.
        </p>
      )}

      <div className="space-y-2 border-t pt-3">
        <div className="flex flex-wrap items-center gap-2">
          <p className="text-xs font-medium">
            Row {source} of {name}
            <sup>{k}</sup>
          </p>
          <div className="flex gap-1" role="group" aria-label="Matrix">
            {[true, false].map((on) => (
              <Button
                key={String(on)}
                size="sm"
                variant={selfLoops === on ? "default" : "outline"}
                aria-pressed={selfLoops === on}
                className="h-6 px-2 text-xs"
                onClick={() => onSelfLoops(on)}
              >
                {on ? "A + I (a walk may pause)" : "A (a walk must move)"}
              </Button>
            ))}
          </div>
        </div>
        <p className="text-xs leading-snug text-muted-foreground">
          Switching the matrix re-reads the same k — nothing re-runs.
        </p>

        <div className="flex flex-col gap-3 sm:flex-row sm:items-start">
          <WalkMatrix
            className="w-full max-w-56 shrink-0 sm:w-48"
            matrix={matrix}
            n={graph.n}
            source={source}
            k={k}
            selfLoops={selfLoops}
          />
          <div className="min-w-0 space-y-2 text-xs">
            <p data-testid="walk-support" data-count={nonzero.length}>
              Nonzero at {nonzero.length} {nonzero.length === 1 ? "node" : "nodes"}:{" "}
              <span className="font-mono tabular-nums">{nonzero.join(", ")}</span>
            </p>
            <Verdict
              selfLoops={selfLoops}
              bipartite={graph.bipartite}
              k={k}
              source={source}
              ballSize={ball.length}
              missing={missing}
              hearsItself={walks[source] > 0}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, testId }: { label: string; value: string; testId: string }) {
  return (
    <div>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-mono tabular-nums" data-testid={testId}>
        {value}
      </dd>
    </div>
  );
}

/**
 * The comparison, said in words. Computed, never assumed: the ✓ is shown only
 * when the row's support really equals the BFS ball, so a bug in either half
 * surfaces here rather than being narrated over.
 */
function Verdict({
  selfLoops,
  bipartite,
  k,
  source,
  ballSize,
  missing,
  hearsItself,
}: {
  selfLoops: boolean;
  bipartite: boolean;
  k: number;
  source: number;
  ballSize: number;
  missing: number[];
  hearsItself: boolean;
}) {
  if (missing.length === 0) {
    return (
      <p data-testid="verdict" data-match="true">
        <strong>✓ The same {ballSize} nodes as the BFS ball.</strong>{" "}
        {selfLoops
          ? `Letting a walk pause makes “a walk of length ${k}” mean “within ${k} hops”, so a ${k}-layer GCN at node ${source} reads exactly these nodes. That is why GCN uses A + I.`
          : `Even without the self-loop, walks of length ${k} happen to reach every node within ${k} hops here — an odd cycle lets a walk fix its parity.`}
      </p>
    );
  }
  return (
    <p data-testid="verdict" data-match="false">
      <strong>
        ✗ {missing.length} of the {ballSize} nodes within {k} {k === 1 ? "hop" : "hops"}{" "}
        {missing.length === 1 ? "has" : "have"} no walk of length exactly {k}
      </strong>{" "}
      <span className="font-mono tabular-nums">({missing.join(", ")})</span>.{" "}
      {bipartite
        ? `This graph is bipartite, so a walk that must move every step can only end at a node whose distance has the same parity as ${k}.`
        : `A walk that must move every step cannot always arrive in exactly ${k} steps.`}{" "}
      {!hearsItself &&
        `Node ${source} cannot even hear itself: a GNN layer without the self-loop drops a node's own features at this depth.`}
    </p>
  );
}
