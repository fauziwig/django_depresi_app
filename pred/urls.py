from django.urls import path, include

from . import views

urlpatterns = [
    path("", views.landing_view, name="landing_view"),
    path("pred/form/", views.my_view, name="my_view"),
    path("pred/success/", views.success_view, name="success_url"),
    path("pred/results/", views.results_view, name="results_url"),
    path("pred/history/", views.history_view, name="history_url"),
    path("pred/login/", views.login_view, name="login_url"),
    path("pred/register/", views.register_view, name="register_url"),
    path("pred/logout/", views.logout_view, name="logout_url"),

    # Admin URLs
    path("pred/admin/dashboard/", views.admin_dashboard, name="admin_dashboard"),
    path("pred/admin/all-submissions/", views.admin_all_submissions, name="admin_all_submissions"),
    path("pred/admin/users/", views.admin_users, name="admin_users"),
    path("pred/admin/delete-submission/<int:submission_id>/", views.admin_delete_submission, name="admin_delete_submission"),

    # Expert URLs
    path("pred/expert/reuse-data/<int:submission_id>/", views.expert_reuse_data, name="expert_reuse_data"),

    # API URLs
    path("pred/api/", include("pred.api_urls")),
]