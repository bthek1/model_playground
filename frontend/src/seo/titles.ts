/**
 * Each task page's `<title>`, minus the " · Model Playground" suffix (#71).
 *
 * Written in the words people search with, not the Hugging Face task names
 * the sidebar keeps: nobody searches "Image to Image" for an upscaler,
 * "Keypoint Detection" for pose estimation or "Audio to Audio" for noise
 * removal. The heading on the page stays the Hub label, so the sidebar, the
 * breadcrumb and the `<h1>` still agree.
 *
 * At most 51 characters, so the full title stays within the 70 a result shows
 * (`pages.test.ts`). Lives apart from `seo/copy/`, because the tab title is
 * set by the entry chunk (`useDocumentHead`) and the page copy must not be.
 */
export const TASK_TITLES: Readonly<Record<string, string>> = {
  // Audio
  "/text-to-speech": "Text to Speech in your browser, free and private",
  "/asr": "Speech to Text with Whisper, in your browser",
  "/audio-to-audio": "Remove Background Noise from Speech, in browser",
  "/audio-classification": "Audio Classification: tag sounds in your browser",
  "/vad": "Voice Activity Detection with Silero VAD",
  // Computer Vision
  "/depth": "Depth Estimation: a depth map from one photo",
  "/image-classification": "Image Classification in your browser",
  "/object-detection": "Object Detection in your browser, on your GPU",
  "/segmentation": "Image Segmentation in your browser",
  "/super-resolution": "AI Image Upscaler (Swin2SR) in your browser",
  "/video-classification": "Video Classification in your browser",
  "/zero-shot-image-classification": "Zero-Shot Image Classification with CLIP",
  "/mask-generation": "Segment Anything (SAM) in your browser",
  "/zero-shot-object-detection": "Detect Objects by Name with OWL-ViT, in browser",
  "/image-to-3d": "Photo to 3D Point Cloud, in your browser",
  "/image-features": "Image Embeddings with CLIP and DINOv2",
  "/pose": "Human Pose Estimation in your browser",
  "/background-removal": "Remove Image Backgrounds in your browser",
  // Multimodal
  "/image-text-to-text": "Ask Questions About an Image with SmolVLM",
  "/visual-question-answering": "Visual Question Answering in your browser",
  "/video-text-to-text": "Ask Questions About a Video with SmolVLM2",
  // Natural Language Processing
  "/text-classification": "Sentiment and Text Classification in browser",
  "/token-classification": "Named Entity Recognition (NER) in your browser",
  "/question-answering": "Extractive Question Answering in your browser",
  "/zero-shot-classification": "Zero-Shot Text Classification in your browser",
  "/translation": "Machine Translation in your browser, privately",
  "/summarization": "Text Summarizer in your browser: T5 and DistilBART",
  "/text-features": "Sentence Embeddings in your browser",
  "/text-generation": "Run a Small LLM in your browser: text generation",
  "/fill-mask": "Fill-Mask with BERT and RoBERTa, in your browser",
  "/sentence-similarity": "Sentence Similarity in your browser",
  "/text-ranking": "BM25 vs Dense Retrieval vs Reranking, in browser",
  // Graph Machine Learning
  "/graph": "Train a Graph Neural Network in your browser",
  "/link-prediction": "Link Prediction with a GNN, in your browser",
  "/graph-classification": "Graph Classification with GNNs, in your browser",
  // Reinforcement Learning
  "/rl": "Reinforcement Learning in your browser: Q-learning",
  "/robotics": "Robot Perception and Behaviour Cloning, in browser",
  // Tabular
  "/tabular-classification": "Train a Classifier on your CSV, in your browser",
  "/tabular-regression": "Train a Regression Model on your CSV, in browser",
  "/time-series-forecasting": "Time Series Forecasting Baselines, in browser",
  // Theory
  "/training": "Train a Classifier on MNIST with WebGPU",
  "/discrete-maths": "BFS and Counting Walks with Adjacency Matrix Powers",
  "/tensor": "Tensor Arithmetic on your GPU with WebGPU",
  "/playground": "WebGPU Inference Playground",
};
