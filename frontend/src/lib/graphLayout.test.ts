import { describe, expect, it } from "vitest";

import { forceLayout, type GraphLayout } from "./graphLayout";

/** Build CSR from an undirected edge list, storing each edge in both directions. */
function csr(nNodes: number, edges: [number, number][]) {
  const degree = new Uint32Array(nNodes);
  for (const [u, v] of edges) {
    degree[u]++;
    degree[v]++;
  }
  const rowPtr = new Uint32Array(nNodes + 1);
  for (let i = 0; i < nNodes; i++) rowPtr[i + 1] = rowPtr[i] + degree[i];
  const colIdx = new Uint32Array(rowPtr[nNodes]);
  const cursor = rowPtr.slice(0, nNodes);
  for (const [u, v] of edges) {
    colIdx[cursor[u]++] = v;
    colIdx[cursor[v]++] = u;
  }
  return { rowPtr, colIdx };
}

/** Every pair within one clique, as an edge list offset by `base`. */
function clique(base: number, size: number): [number, number][] {
  const edges: [number, number][] = [];
  for (let i = 0; i < size; i++) {
    for (let j = i + 1; j < size; j++) edges.push([base + i, base + j]);
  }
  return edges;
}

const meanDistance = (
  l: GraphLayout,
  a: number[],
  b: number[],
  sameSet: boolean,
) => {
  let sum = 0;
  let n = 0;
  for (const i of a) {
    for (const j of b) {
      if (sameSet && j <= i) continue;
      sum += Math.hypot(l.x[i] - l.x[j], l.y[i] - l.y[j]);
      n++;
    }
  }
  return sum / n;
};

describe("forceLayout", () => {
  it("is reproducible for a seed, and different for another", () => {
    const { rowPtr, colIdx } = csr(12, clique(0, 12));
    const a = forceLayout(rowPtr, colIdx, 12, { seed: 3, iterations: 60 });
    const b = forceLayout(rowPtr, colIdx, 12, { seed: 3, iterations: 60 });
    const c = forceLayout(rowPtr, colIdx, 12, { seed: 4, iterations: 60 });

    expect([...a.x]).toEqual([...b.x]);
    expect([...a.y]).toEqual([...b.y]);
    expect([...a.x]).not.toEqual([...c.x]);
  });

  it("produces finite coordinates inside the unit square", () => {
    const edges: [number, number][] = [];
    for (let i = 1; i < 200; i++) edges.push([i, Math.floor(i / 2)]); // a binary tree
    const { rowPtr, colIdx } = csr(200, edges);
    const l = forceLayout(rowPtr, colIdx, 200, { iterations: 120 });

    for (let i = 0; i < 200; i++) {
      expect(Number.isFinite(l.x[i])).toBe(true);
      expect(Number.isFinite(l.y[i])).toBe(true);
      expect(l.x[i]).toBeGreaterThanOrEqual(0);
      expect(l.x[i]).toBeLessThanOrEqual(1);
      expect(l.y[i]).toBeGreaterThanOrEqual(0);
      expect(l.y[i]).toBeLessThanOrEqual(1);
    }
  });

  it("does not collapse the drawing to a point", () => {
    const { rowPtr, colIdx } = csr(60, clique(0, 60));
    const l = forceLayout(rowPtr, colIdx, 60, { iterations: 150 });
    const span = Math.max(...l.x) - Math.min(...l.x);
    expect(span).toBeGreaterThan(0.5); // normalise() fills one axis
  });

  it("separates two disconnected components", () => {
    // Two cliques, no edge between them — the structure the layout has to show.
    const edges = [...clique(0, 25), ...clique(25, 25)];
    const { rowPtr, colIdx } = csr(50, edges);
    const l = forceLayout(rowPtr, colIdx, 50, { iterations: 400 });

    const left = Array.from({ length: 25 }, (_, i) => i);
    const right = Array.from({ length: 25 }, (_, i) => i + 25);
    const within =
      (meanDistance(l, left, left, true) + meanDistance(l, right, right, true)) / 2;
    const between = meanDistance(l, left, right, false);

    expect(between).toBeGreaterThan(within);
  });

  it("handles the degenerate graphs without dividing by zero", () => {
    expect(forceLayout(new Uint32Array(1), new Uint32Array(0), 0).x).toHaveLength(0);
    const one = forceLayout(Uint32Array.from([0, 0]), new Uint32Array(0), 1);
    expect(Number.isFinite(one.x[0])).toBe(true);

    // Isolated nodes only: no attraction at all, so nothing bounds the drawing
    // except gravity. It must still normalise rather than return NaN.
    const isolated = forceLayout(new Uint32Array(6), new Uint32Array(0), 5, {
      iterations: 50,
    });
    for (let i = 0; i < 5; i++) expect(Number.isFinite(isolated.x[i])).toBe(true);
  });
});
