import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Train a graph neural network to tell enzymes from non-enzymes in the PROTEINS dataset, in your browser.",
  about: [
    "Graph classification gives one label to a whole graph rather than one to each node. The PROTEINS dataset holds 1113 protein structures, each a small graph of its parts, labelled enzyme or not. This page trains GCN, GraphSAGE, GIN or GAT on them in the tab, then pools each molecule's node outputs into a single answer. The graph pooling is your choice of mean or sum: mean ignores the size of the protein, sum does not, and on PROTEINS size itself carries signal.",
    "It is the only page in the graph category that fetches anything: the dataset, about 2 MB, comes from the Hugging Face Hub on first use and is kept in IndexedDB afterwards. There are no model weights to download. All 1113 graphs are joined into one disconnected graph of 43 471 nodes and trained full-batch through the same WGSL kernel as node classification, on your GPU with WebGPU or on the CPU without it.",
    "Every accuracy is shown beside its majority baseline, and that comparison is the point. PROTEINS has 663 enzymes to 450 non-enzymes, so a classifier that ignores the molecules and always answers enzyme scores 0.598, while published GNNs reach roughly 0.73 to 0.76. A score without that reference cannot be told apart from a model that only learned the class balance. A gallery of 48 held-out proteins, outlined by whether the model got them right, shows which shapes it finds hard.",
  ],
  aliases: [
    "graph classification",
    "PROTEINS dataset",
    "enzymes",
    "graph pooling",
    "majority baseline",
    "GNN",
  ],
};
