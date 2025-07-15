from django.urls import path

from . import views

urlpatterns = [
    path("", views.my_view, name="my_view"),
    path("success/", views.success_view, name="success_url"),
    path("results/", views.results_view, name="results_url"),
    path("history/", views.history_view, name="history_url"),
    path("login/", views.login_view, name="login_url"),
    path("register/", views.register_view, name="register_url"),
    path("logout/", views.logout_view, name="logout_url"),

    # Admin URLs
    path("admin/dashboard/", views.admin_dashboard, name="admin_dashboard"),
    path("admin/all-submissions/", views.admin_all_submissions, name="admin_all_submissions"),
    path("admin/users/", views.admin_users, name="admin_users"),
    path("admin/delete-submission/<int:submission_id>/", views.admin_delete_submission, name="admin_delete_submission"),

    # Expert URLs
    path("expert/reuse-data/<int:submission_id>/", views.expert_reuse_data, name="expert_reuse_data"),
]