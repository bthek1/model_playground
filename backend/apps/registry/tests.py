import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from apps.registry.models import InferenceRun, ModelCard

User = get_user_model()


@pytest.fixture
def api():
    return APIClient()


@pytest.fixture
def user(db):
    return User.objects.create_user(email="dev@example.com", password="pw12345678")


@pytest.fixture
def public_model(db):
    return ModelCard.objects.create(
        slug="tiny-matmul",
        name="Tiny MatMul",
        task="custom",
        weights_url="https://example.com/tiny.safetensors",
        config={"entry_point": "main", "workgroup_size": [16, 16, 1]},
        is_public=True,
    )


@pytest.mark.django_db
class TestModelCardEndpoints:
    def test_anonymous_can_list_public_models(self, api, public_model):
        res = api.get("/api/registry/models/")
        assert res.status_code == 200
        slugs = [m["slug"] for m in res.json()]
        assert "tiny-matmul" in slugs

    def test_anonymous_cannot_see_private_models(self, api, db):
        ModelCard.objects.create(slug="secret", name="Secret", is_public=False)
        res = api.get("/api/registry/models/")
        assert res.status_code == 200
        assert all(m["slug"] != "secret" for m in res.json())

    def test_anonymous_cannot_create(self, api):
        res = api.post("/api/registry/models/", {"slug": "x", "name": "X"})
        assert res.status_code == 401

    def test_authenticated_user_can_create(self, api, user):
        api.force_authenticate(user)
        res = api.post(
            "/api/registry/models/",
            {"slug": "my-net", "name": "My Net", "task": "vision"},
            format="json",
        )
        assert res.status_code == 201
        card = ModelCard.objects.get(slug="my-net")
        assert card.created_by == user

    def test_retrieve_by_slug(self, api, public_model):
        res = api.get("/api/registry/models/tiny-matmul/")
        assert res.status_code == 200
        assert res.json()["config"]["entry_point"] == "main"


@pytest.mark.django_db
class TestInferenceRunEndpoints:
    def test_auth_required_to_record_run(self, api, public_model):
        res = api.post(
            "/api/registry/runs/", {"model": str(public_model.id)}, format="json"
        )
        assert res.status_code == 401

    def test_record_run_persists_metrics(self, api, user, public_model):
        api.force_authenticate(user)
        res = api.post(
            "/api/registry/runs/",
            {
                "model": str(public_model.id),
                "status": "completed",
                "metrics": {"latency_ms": 12.5, "gflops": 340.0},
            },
            format="json",
        )
        assert res.status_code == 201
        run = InferenceRun.objects.get()
        assert run.user == user
        assert run.metrics["gflops"] == 340.0

    def test_user_only_sees_own_runs(self, api, user, public_model):
        other = User.objects.create_user(
            email="other@example.com", password="pw12345678"
        )
        InferenceRun.objects.create(model=public_model, user=other)
        api.force_authenticate(user)
        res = api.get("/api/registry/runs/")
        assert res.status_code == 200
        assert res.json() == []
