from django.urls import path, include

from . import views

urlpatterns = [
    path("", views.landing_view, name="landing_view"),
    path("diagnosis/", views.diagnosis, name="diagnosis"),
    path("user/", views.user, name="user"),
    path("diagnosis/results/", views.results_view, name="results_url"),
    
    path("login/", views.login_view, name="login_url"),
    path("register/", views.register_view, name="register_url"),
    path("logout/", views.logout_view, name="logout_url"),

    # Admin URLs
    

    # Expert URLs
    path("pred/expert/reuse-data/<int:submission_id>/", views.expert_reuse_data, name="expert_reuse_data"),
    # path("pred/expert/reuse-data/<int:submission_id>/", views.expert_reuse_data, name="expert_reuse_data"),

    # API URLs
    # path("pred/api/", include("pred.api_urls")),
]