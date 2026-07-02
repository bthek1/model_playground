from django.db.models import Q
from rest_framework import mixins, viewsets
from rest_framework.permissions import IsAuthenticated, IsAuthenticatedOrReadOnly

from .models import InferenceRun, ModelCard
from .serializers import InferenceRunSerializer, ModelCardSerializer
from .services import record_inference_run


class ModelCardViewSet(viewsets.ModelViewSet):
    """CRUD for model catalog entries.

    Anyone may read public models; authenticated users additionally see their
    own private models and may create/update/delete their own entries.
    """

    serializer_class = ModelCardSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    lookup_field = "slug"

    def get_queryset(self):
        qs = ModelCard.objects.all()
        user = self.request.user
        if user.is_authenticated:
            return qs.filter(Q(is_public=True) | Q(created_by=user))
        return qs.filter(is_public=True)

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)


class InferenceRunViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    viewsets.GenericViewSet,
):
    """Create and list the current user's in-browser inference runs."""

    serializer_class = InferenceRunSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return InferenceRun.objects.filter(user=self.request.user).select_related(
            "model"
        )

    def perform_create(self, serializer):
        run = record_inference_run(
            model=serializer.validated_data["model"],
            user=self.request.user,
            status=serializer.validated_data.get(
                "status", InferenceRun.Status.COMPLETED
            ),
            params=serializer.validated_data.get("params"),
            metrics=serializer.validated_data.get("metrics"),
        )
        serializer.instance = run
