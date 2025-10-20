from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("/dashboard", views.dashboard, name="dashboard"),
    path("/about", views.about, name="about"),
    path("/analyze", views.analyze_view, name="analyze"),
    path("/validate-data", views.validate_data_view, name="validate_data"),
    path("/download-pdf/<str:analysis_id>/", views.download_pdf, name="download_pdf"),
]