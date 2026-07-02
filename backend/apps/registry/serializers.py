from rest_framework import serializers

from .models import InferenceRun, ModelCard


class ModelCardSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelCard
        fields = (
            "id",
            "slug",
            "name",
            "task",
            "description",
            "weights_url",
            "config",
            "size_bytes",
            "license",
            "is_public",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("id", "created_at", "updated_at")


class InferenceRunSerializer(serializers.ModelSerializer):
    model_slug = serializers.SlugRelatedField(
        source="model", slug_field="slug", read_only=True
    )

    class Meta:
        model = InferenceRun
        fields = (
            "id",
            "model",
            "model_slug",
            "status",
            "params",
            "metrics",
            "created_at",
        )
        read_only_fields = ("id", "model_slug", "created_at")
