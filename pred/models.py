from django.db import models
from django.utils import timezone

# Create your models here.
class FormSubmission(models.Model):
    gender = models.CharField(max_length=10, verbose_name="Gender")
    age = models.IntegerField(verbose_name="Age")
    work_pressure = models.CharField(max_length=10, verbose_name="Work Pressure")
    job_satisfaction = models.CharField(max_length=10, verbose_name="Job Satisfaction")
    financial_stress = models.CharField(max_length=10, verbose_name="Financial Stress")
    sleep_duration = models.CharField(max_length=10, verbose_name="Sleep Duration")
    dietary_habits = models.CharField(max_length=10, verbose_name="Dietary Habits")
    suicidal_thoughts = models.CharField(max_length=10, verbose_name="Suicidal Thoughts")
    work_hours = models.IntegerField(verbose_name="Work Hours")
    family_history_of_mental_illness = models.CharField(max_length=10, verbose_name="Family History of Mental Illness")

    # Prediction results
    prediction_result = models.CharField(max_length=20, verbose_name="Prediction Result", blank=True, null=True)
    prediction_probability = models.FloatField(verbose_name="Prediction Probability", blank=True, null=True)
    prediction_message = models.TextField(verbose_name="Prediction Message", blank=True, null=True)

    # Similarity results
    similarity_score = models.FloatField(verbose_name="Similarity Score", blank=True, null=True)
    similar_case_id = models.IntegerField(verbose_name="Similar Case ID", blank=True, null=True)

    # Dataset reuse tracking
    is_reused_in_dataset = models.BooleanField(verbose_name="Reused in Dataset", default=False)
    reused_at = models.DateTimeField(verbose_name="Reused At", blank=True, null=True)
    reused_by = models.ForeignKey('auth.User', verbose_name="Reused By", blank=True, null=True, on_delete=models.SET_NULL, related_name='reused_submissions')

    submitted_at = models.DateTimeField(default=timezone.now, verbose_name="Waktu Submit")

    class Meta:
        ordering = ['-submitted_at']  # Latest submissions first
        verbose_name = "Form Submission"
        verbose_name_plural = "Form Submissions"

    def __str__(self):
        return f"Submission #{self.id} - {self.submitted_at.strftime('%Y-%m-%d %H:%M')}"
