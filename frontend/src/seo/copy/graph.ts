import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline:
    "Train a graph neural network to classify the papers of the Cora citation graph, and watch it learn.",
  about: [
    "Node classification gives every node in a graph a label, using its own features and those of its neighbours. On this page a graph neural network learns to sort the 2708 machine-learning papers of Cora into seven topics from their 5278 citations, starting from 20 labelled papers per topic. The whole graph is drawn while it trains, so you can see the predictions spread along the citation links epoch by epoch.",
    "Four architectures are on offer: GCN, GraphSAGE, GIN and GAT. The first three differ only in how a node averages or sums its neighbourhood, and share one hand-written WGSL compute shader that runs on your GPU through WebGPU. GAT learns an attention weight per edge, so its gather runs in TypeScript while its projections stay on the GPU. A machine without WebGPU runs the same arithmetic on the CPU, which is also the reference the shader is checked against. There is no pretrained checkpoint: Cora ships with the page as a 161 KB sparse file and the model starts from random weights.",
    "The depth slider is the page's second lesson, oversmoothing. Each extra layer mixes every node with a wider neighbourhood until neighbouring papers look alike: measured with GCN, test accuracy falls from about 0.78 at two layers to about 0.52 at eight while the similarity between adjacent nodes climbs towards 1. GIN fails differently, because its unnormalised sum grows until ReLU switches units off, and the page reports that separately rather than calling it oversmoothing.",
  ],
  aliases: [
    "graph neural network",
    "node classification",
    "Cora citation graph",
    "GCN",
    "GraphSAGE",
    "oversmoothing",
  ],
};
