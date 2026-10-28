from django.urls import path

from apps.core import views

app_name = "core"

urlpatterns = [
    path("", views.home, name="home"),
    path("model/performance/", views.model_performance, name="model_performance"),
    path(
        "model/performance/calibration.png",
        views.calibration_curve_png,
        name="calibration_curve",
    ),
]