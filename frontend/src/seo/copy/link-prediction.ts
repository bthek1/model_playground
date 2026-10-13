import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Predict missing citations in the Cora graph with a GNN trained in your browser, scored by AUC.",
  about: [
    "Link prediction asks which pairs of nodes in a graph should be connected but are not. Here the graph is Cora, 2708 papers and their citations: a share of the citations (15% by default, adjustable from 5% to 30%) is taken out before training, a graph neural network learns an embedding for every paper from what is left, and each pair is scored by the dot product of its two embeddings through a sigmoid. That is the graph auto-encoder recipe, with GCN, GraphSAGE, GIN or GAT as the encoder.",
    "The result is reported as test AUC, the probability that a real held-out citation outscores a random unlinked pair, with average precision beside it. With the shipped settings it measures about 0.925 on Cora, against 0.910 published for the original graph auto-encoder. The page then draws the pairs it rates most likely among those that are not linked, and you can pick any two papers to score them directly.",
    "The failure this page guards against is leakage, and it makes the number better rather than worse. If the encoder may still pass messages along a citation it is later tested on, it has already mixed the two papers together and answers from memory, and test AUC climbs to around 0.99. So the held-out citations leave the graph itself, both directions of each one, the node degrees are recomputed without them, and negative pairs are checked against the full graph. Training runs as WGSL on your GPU through WebGPU, or the same arithmetic on the CPU without it; there is no pretrained model to fetch.",
  ],
  aliases: [
    "link prediction",
    "missing citations",
    "graph auto-encoder",
    "AUC",
    "Cora",
    "GNN",
  ],
};
