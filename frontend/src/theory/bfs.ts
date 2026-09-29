// Breadth-first search, kept in layers because the layers are what the page
// shows: step k adds exactly the nodes at distance k, and the union of the
// first k layers is the k-hop neighbourhood a k-layer GNN reads from.

export interface BfsResult {
  /** Hops from the source, or −1 where unreachable. */
  dist: Int32Array;
  /** `layers[d]` is every node at distance `d`, ascending. `layers[0]` is the source. */
  layers: number[][];
  /** The largest finite distance — how many steps until BFS has nothing left to add. */
  eccentricity: number;
}

export function bfs(neighbours: readonly (readonly number[])[], source: number): BfsResult {
  const n = neighbours.length;
  if (source < 0 || source >= n) throw new RangeError(`source ${source} is not a node of a ${n}-node graph`);

  const dist = new Int32Array(n).fill(-1);
  dist[source] = 0;
  const layers: number[][] = [[source]];

  let frontier = [source];
  while (frontier.length) {
    const next: number[] = [];
    for (const u of frontier) {
      for (const v of neighbours[u]) {
        if (dist[v] < 0) {
          dist[v] = dist[u] + 1;
          next.push(v);
        }
      }
    }
    if (next.length) layers.push(next.sort((a, b) => a - b));
    frontier = next;
  }

  return { dist, layers, eccentricity: layers.length - 1 };
}
