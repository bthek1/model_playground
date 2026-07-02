from rest_framework.routers import DefaultRouter

from .views import InferenceRunViewSet, ModelCardViewSet

router = DefaultRouter()
router.register("models", ModelCardViewSet, basename="modelcard")
router.register("runs", InferenceRunViewSet, basename="inferencerun")

urlpatterns = router.urls
