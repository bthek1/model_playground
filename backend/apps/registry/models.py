import uuid

from django.conf import settings
from django.db import models


class ModelTask(models.TextChoices):
    """The kind of inference a model card performs in the browser."""

    LLM = "llm", "Language model"
    VISION = "vision", "Computer vision"
    EMBEDDING = "embedding", "Embedding"
    AUDIO = "audio", "Audio"
    CUSTOM = "custom", "Custom"


class ModelCard(models.Model):
    """Catalog entry describing a model that runs in-browser via WebGPU.

    The backend never runs inference — it only stores the metadata a browser
    needs to fetch weights and build the WebGPU pipeline (`config`), plus
    discovery fields (task, license, size) for the playground UI.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    slug = models.SlugField(unique=True, max_length=120)
    name = models.CharField(max_length=200)
    task = models.CharField(
        max_length=20, choices=ModelTask.choices, default=ModelTask.CUSTOM
    )
    description = models.TextField(blank=True)

    # Where the browser fetches the weights from (safetensors/gguf/bin/onnx…).
    weights_url = models.URLField(max_length=500, blank=True)

    # Free-form config the browser runtime consumes: tensor shapes, quantization,
    # WGSL entry points, tokenizer refs, preprocessing params, etc.
    config = models.JSONField(default=dict, blank=True)

    size_bytes = models.BigIntegerField(null=True, blank=True)
    license = models.CharField(max_length=100, blank=True)
    is_public = models.BooleanField(default=True)

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="model_cards",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        indexes = [
            models.Index(fields=["task"]),
            models.Index(fields=["is_public"]),
        ]

    def __str__(self):
        return self.name


class InferenceRun(models.Model):
    """A record of an in-browser inference run, reported by the client.

    Inference happens entirely on the user's GPU; the client POSTs the outcome
    (params + metrics such as latency and throughput, plus the GPU adapter it
    used) so the playground can show history and benchmarks.
    """

    class Status(models.TextChoices):
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(ModelCard, on_delete=models.CASCADE, related_name="runs")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="inference_runs",
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.COMPLETED
    )

    # Inputs used for the run (prompt, generation params, image ref, …).
    params = models.JSONField(default=dict, blank=True)
    # Client-reported metrics: latency_ms, tokens_per_sec, adapter info, …
    metrics = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["-created_at"]),
        ]

    def __str__(self):
        return f"{self.model.slug} @ {self.created_at:%Y-%m-%d %H:%M}"
