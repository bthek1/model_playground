import type { LucideIcon } from "lucide-react";
import {
  AudioLines,
  Eye,
  Gamepad2,
  Layers,
  Shapes,
  Sigma,
  Table,
  Type,
} from "lucide-react";

/**
 * The app's navigation taxonomy — a two-level tree of task categories and the
 * tasks within them, modelled on the Hugging Face pipeline taxonomy.
 *
 * The sidebar is data-driven: to add a task, add an entry here. A task links to
 * a real route when one exists (`to`), otherwise it falls back to the generic
 * `/tasks/$slug` landing page (see `routes/tasks.$slug.tsx`).
 */

export interface TaskItem {
  label: string;
  /** Kebab-case identifier derived from `label`; used in `/tasks/$slug`. */
  slug: string;
  /** Route this task links to. Real route if implemented, else `/tasks/$slug`. */
  to: string;
}

export interface TaskCategory {
  label: string;
  icon: LucideIcon;
  tasks: TaskItem[];
}

function slugify(label: string): string {
  return label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

/**
 * Tasks that map to a real, implemented route. Everything else routes to the
 * generic `/tasks/$slug` placeholder.
 */
const REAL_ROUTES: Record<string, string> = {
  "text-generation": "/playground",
  "linear-model-training": "/training",
  "tensor-arithmetic": "/tensor",
  "automatic-speech-recognition": "/asr",
  "audio-classification": "/audio-classification",
};

function task(label: string): TaskItem {
  const slug = slugify(label);
  return { label, slug, to: REAL_ROUTES[slug] ?? `/tasks/${slug}` };
}

function tasks(labels: string[]): TaskItem[] {
  return labels.map(task);
}

export const taskCategories: TaskCategory[] = [
  {
    label: "Audio",
    icon: AudioLines,
    tasks: tasks([
      "Text to Speech",
      "Text to Audio",
      "Automatic Speech Recognition",
      "Audio to Audio",
      "Audio Classification",
      "Voice Activity Detection",
    ]),
  },
  {
    label: "Computer Vision",
    icon: Eye,
    tasks: tasks([
      "Depth Estimation",
      "Image Classification",
      "Object Detection",
      "Image Segmentation",
      "Text to Image",
      "Image to Text",
      "Image to Image",
      "Image to Video",
      "Unconditional Image Generation",
      "Video Classification",
      "Text to Video",
      "Zero Shot Image Classification",
      "Mask Generation",
      "Zero Shot Object Detection",
      "Text to 3D",
      "Image to 3D",
      "Image Feature Extraction",
      "Keypoint Detection",
      "Video to Video",
    ]),
  },
  {
    label: "Multimodal",
    icon: Layers,
    tasks: tasks([
      "Audio Text to Text",
      "Image Text to Text",
      "Image Text to Image",
      "Image Text to Video",
      "Visual Question Answering",
      "Document Question Answering",
      "Video Text to Text",
      "Visual Document Retrieval",
      "Any to Any",
    ]),
  },
  {
    label: "Natural Language Processing",
    icon: Type,
    tasks: tasks([
      "Text Classification",
      "Token Classification",
      "Table Question Answering",
      "Question Answering",
      "Zero Shot Classification",
      "Translation",
      "Summarization",
      "Feature Extraction",
      "Text Generation",
      "Fill Mask",
      "Sentence Similarity",
      "Text Ranking",
    ]),
  },
  {
    label: "Other",
    icon: Shapes,
    tasks: tasks(["Graph Machine Learning"]),
  },
  {
    label: "Reinforcement Learning",
    icon: Gamepad2,
    tasks: tasks(["Reinforcement Learning", "Robotics"]),
  },
  {
    label: "Tabular",
    icon: Table,
    tasks: tasks([
      "Tabular Classification",
      "Tabular Regression",
      "Time Series Forecasting",
    ]),
  },
  {
    label: "Theory",
    icon: Sigma,
    tasks: tasks([
      "Linear Model Training",
      "Discrete Maths",
      "Tensor Arithmetic",
    ]),
  },
];

/** Flat lookup of every task by slug — used by the `/tasks/$slug` route. */
export const tasksBySlug: Record<string, TaskItem> = Object.fromEntries(
  taskCategories.flatMap((c) => c.tasks.map((t) => [t.slug, t])),
);

/** The category label that owns the given pathname, if any. */
export function categoryForPath(pathname: string): string | undefined {
  for (const category of taskCategories) {
    if (category.tasks.some((t) => t.to === pathname)) return category.label;
  }
  return undefined;
}
