from django.contrib import admin
from django.http import HttpResponse
from .models import FormSubmission

# Register your models here.
@admin.register(FormSubmission)
class FormSubmissionAdmin(admin.ModelAdmin):
    list_display = ('id', 'gender', 'age', 'prediction_result', 'prediction_probability', 'similarity_score', 'is_reused_in_dataset', 'submitted_at')
    list_filter = ('gender', 'work_pressure', 'job_satisfaction', 'financial_stress', 'sleep_duration', 'dietary_habits', 'suicidal_thoughts', 'family_history_of_mental_illness', 'prediction_result', 'is_reused_in_dataset', 'submitted_at')
    search_fields = ('prediction_result',)
    readonly_fields = ('submitted_at', 'prediction_result', 'prediction_probability', 'prediction_message', 'similarity_score', 'similar_case_id', 'is_reused_in_dataset', 'reused_at', 'reused_by')
    ordering = ('-submitted_at',)

    # Add date hierarchy for easy filtering by date
    date_hierarchy = 'submitted_at'

    # Show 25 items per page
    list_per_page = 25

    # Group fields in the admin form
    fieldsets = (
        ('Basic Information', {
            'fields': ('gender', 'age')
        }),
        ('Assessment Data', {
            'fields': ('work_pressure', 'job_satisfaction', 'financial_stress', 'sleep_duration', 'dietary_habits', 'suicidal_thoughts', 'work_hours', 'family_history_of_mental_illness')
        }),
        ('Prediction Results', {
            'fields': ('prediction_result', 'prediction_probability', 'prediction_message'),
            'classes': ('collapse',)
        }),
        ('Similarity Analysis', {
            'fields': ('similarity_score', 'similar_case_id'),
            'classes': ('collapse',)
        }),
        ('Dataset Reuse Tracking', {
            'fields': ('is_reused_in_dataset', 'reused_at', 'reused_by'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('submitted_at',)
        })
    )
