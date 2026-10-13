import type { PageCopy } from "./types";

export const copy: PageCopy = {
  tagline: "The whole retrieval stack, in your browser: keyword search, dense embeddings, a hybrid fusion of the two, and a cross-encoder rerank.",
  about: [
    "Text ranking orders a set of passages by how well each answers a query, the retrieval step behind semantic search. Paste a corpus or use the sample, type a query, and this page ranks it four ways in four columns: BM25 keyword search, dense retrieval with sentence embeddings, hybrid search by reciprocal rank fusion (RRF), and reranking with a cross-encoder. The rank movement between the columns is the result, because the stages disagree and a single fused list would hide it. The sample corpus is built so that BM25 misses an answer the other stages find.",
    "BM25 and RRF are plain arithmetic with no model, so BM25 ranks before anything is downloaded and changing k1, b or RRF's k re-scores instantly. Embedding the corpus costs one forward pass per passage, once; each query after that is a single pass plus dot products. A cross-encoder reads the query and a passage together, so its scores cannot be precomputed, and it reranks only the top 20 candidates, one pass each. The page states both counts before you press.",
    "Each choice is a pair of models loaded together, one embedder and one reranker. The default, all-MiniLM-L6-v2 with the ms-marco MiniLM reranker, is 91 MB on WebGPU or 46 MB on the CPU. BGE base with the same reranker is 264 MB and 133 MB. MiniLM with the mxbai reranker is 133 MB and 110 MB, since that reranker publishes no fp16 build and runs at 8-bit on both. The corpus and the query stay in your browser.",
  ],
  aliases: [
    "BM25",
    "dense retrieval",
    "reranking",
    "cross-encoder",
    "hybrid search",
    "reciprocal rank fusion",
    "semantic search",
  ],
};
