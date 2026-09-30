// The node-link drawing for /discrete-maths: which nodes BFS has reached after
// k steps, with the hop count written on each.
//
// SVG rather than the canvas `GraphCanvas` uses, and the visualization
// standard's "one canvas, not a node per node" rule is why that is fine here:
// the rule is about 2708 nodes repainted every epoch, and these graphs are at
// most 34 nodes redrawn once per button press. SVG buys a real element per
// node, so the distance and the walk count are in the DOM for a screen reader
// and for a spec, instead of only in pixels.
//
// Nothing is encoded in colour alone (model-visualization §4): the distance is
// printed on every reached node and the frontier is ringed as well as shaded.

import { useMemo } from "react";

import type { SmallGraph } from "@/theory/graphs";

import { fitToBox, reachedOpacity } from "./graphGeometry";

const WIDTH = 360;
const HEIGHT = 300;
const RADIUS = 11;

export function GraphDiagram({
  graph,
  source,
  dist,
  k,
  walks,
}: {
  graph: SmallGraph;
  source: number;
  /** BFS distances from `source`; null draws the bare graph. */
  dist: Int32Array | null;
  /** Steps taken; nodes with `dist ≤ k` are reached. */
  k: number | null;
  /** Row `source` of the matrix power on screen, for each node's tooltip. */
  walks?: ArrayLike<number> | null;
}) {
  const pos = useMemo(
    () => fitToBox(graph.layout(), { width: WIDTH, height: HEIGHT, pad: RADIUS + 12 }),
    [graph],
  );
  const reached = (v: number) => dist != null && k != null && dist[v] >= 0 && dist[v] <= k;
  const reachedCount = dist && k != null ? Array.from(dist).filter((d) => d >= 0 && d <= k).length : 0;

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="h-auto w-full max-w-[36rem]"
      role="img"
      aria-label={
        k == null
          ? `${graph.label}: ${graph.n} nodes, ${graph.edges.length} edges. Source node ${source}.`
          : `${graph.label} after ${k} BFS ${k === 1 ? "step" : "steps"} from node ${source}: ${reachedCount} of ${graph.n} nodes reached.`
      }
      data-testid="graph-diagram"
    >
      <g>
        {graph.edges.map(([u, v]) => (
          <line
            key={`${u}-${v}`}
            x1={pos.x[u]}
            y1={pos.y[u]}
            x2={pos.x[v]}
            y2={pos.y[v]}
            stroke={reached(u) && reached(v) ? "var(--muted-foreground)" : "var(--border)"}
            strokeWidth={reached(u) && reached(v) ? 1.6 : 1}
          />
        ))}
      </g>
      <g>
        {Array.from({ length: graph.n }, (_, v) => {
          const d = dist?.[v] ?? -1;
          const on = reached(v);
          const frontier = on && k != null && d === k && k > 0;
          return (
            <g
              key={v}
              data-testid={`node-${v}`}
              data-reached={on ? "true" : "false"}
              data-dist={on ? d : undefined}
              data-walks={walks ? walks[v] : undefined}
            >
              <title>
                {`Node ${v}`}
                {on ? ` — ${d} ${d === 1 ? "hop" : "hops"} from ${source}` : ""}
                {walks ? ` — ${walks[v]} ${walks[v] === 1 ? "walk" : "walks"}` : ""}
              </title>
              <circle
                cx={pos.x[v]}
                cy={pos.y[v]}
                r={RADIUS}
                fill={on ? "var(--primary)" : "var(--muted)"}
                fillOpacity={on ? reachedOpacity(d, k ?? 0) : 1}
                stroke={v === source || frontier ? "var(--primary)" : "var(--border)"}
                strokeWidth={v === source ? 3 : frontier ? 2.2 : 1}
                strokeDasharray={frontier ? "3 2" : undefined}
              />
              {on && (
                <text
                  x={pos.x[v]}
                  y={pos.y[v]}
                  textAnchor="middle"
                  dominantBaseline="central"
                  className="fill-foreground text-[10px] font-semibold tabular-nums"
                >
                  {d}
                </text>
              )}
              <text
                x={pos.x[v] + RADIUS * 0.8}
                y={pos.y[v] - RADIUS * 0.9}
                className="fill-muted-foreground text-[8px] tabular-nums"
              >
                {v}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}
