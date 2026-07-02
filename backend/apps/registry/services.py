from .models import InferenceRun, ModelCard


def record_inference_run(
    *,
    model: ModelCard,
    user=None,
    status: str = InferenceRun.Status.COMPLETED,
    params: dict | None = None,
    metrics: dict | None = None,
) -> InferenceRun:
    """Persist the outcome of an in-browser inference run.

    Inference itself runs on the client's GPU; this only stores what the
    client reports so the playground can show history and benchmarks.
    """
    return InferenceRun.objects.create(
        model=model,
        user=user if (user and user.is_authenticated) else None,
        status=status,
        params=params or {},
        metrics=metrics or {},
    )
