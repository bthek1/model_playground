import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Step a breadth-first search one layer at a time, and count walks with powers of the adjacency matrix.",
  about: [
    "Two pieces of graph theory meet on this page. A breadth-first search (BFS) finds every node within k hops of a source — its k-hop neighbourhood. The adjacency matrix A has a 1 wherever two nodes share an edge, and the entry at row s, column v of its k-th power counts the walks of length k from s to v. Pick a graph and a source node, press Step, and the search grows by one layer while the matching row of the matrix power appears beside it.",
    "With a self-loop added, the nonzero entries in that row of (A + I) to the k are exactly the nodes the search has reached. With A alone they need not be: on a bipartite graph a walk that must move at every step can only end at a node whose distance has the same parity as k, so at odd k the source cannot even reach itself. The page computes the match or mismatch from both halves rather than asserting it. This is why a graph convolutional network adds the self-loop, and why a k-layer GNN's receptive field is the k-hop ball.",
    "The graphs are small and fixed: a path, two cycles (one even, one odd), a 4×4 grid, the Petersen graph and Zachary's karate club with its 34 members and 78 edges. Each node is drawn with its distance from the source and its walk count.",
    "Everything is exact integer arithmetic on the main thread, finished in microseconds, with no model, no Web Worker, no GPU and nothing to download. It covers this one topic only — there are no weighted shortest paths, and you cannot load a graph of your own.",
  ],
  aliases: [
    "breadth-first search",
    "adjacency matrix",
    "counting walks",
    "k-hop neighbourhood",
    "graph theory",
    "Petersen graph",
    "Zachary's karate club",
  ],
};
