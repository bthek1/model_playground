from django.contrib import admin

from .models import InferenceRun, ModelCard


@admin.register(ModelCard)
class ModelCardAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "task", "is_public", "created_at")
    list_filter = ("task", "is_public")
    search_fields = ("name", "slug", "description")
    prepopulated_fields = {"slug": ("name",)}
    readonly_fields = ("created_at", "updated_at")


@admin.register(InferenceRun)
class InferenceRunAdmin(admin.ModelAdmin):
    list_display = ("model", "user", "status", "created_at")
    list_filter = ("status",)
    search_fields = ("model__slug", "user__email")
    readonly_fields = ("created_at",)
