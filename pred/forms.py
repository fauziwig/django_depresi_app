from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class MyForm(forms.Form):

    # Radio button choices (removed empty default options)
    GENDER_CHOICES = [
        ('1', 'Laki-laki'),
        ('0', 'Perempuan'),
    ]

    PRESSURE_CHOICES = [
        ('1', 'Sangat Santai'),
        ('2', 'Santai'),
        ('3', 'Cukup Sibuk'),
        ('4', 'Tertekan'),
        ('5', 'Sangat Tertekan'),
    ]

    SATISFACTION_CHOICES = [
        ('1', 'Sangat Tidak Puas'),
        ('2', 'Tidak Puas'),
        ('3', 'Cukup Puas'),
        ('4', 'Puas'),
        ('5', 'Sangat Puas'),
    ]

    STRESS_FINANCIAL_CHOICES = [
        ('1', 'Sangat Aman'),
        ('2', 'Aman'),
        ('3', 'Netral'),
        ('4', 'Khawatir'),
        ('5', 'Sangat Khawatir'),
    ]

    SLEEP_DURATION_CHOICES = [
        ('0', '5-6 Jam'),
        ('1', '7-8 Jam'),
        ('2', 'Kurang dari 5 Jam'),
        ('3', 'Lebih dari 8 Jam'),
    ]

    DIETARY_HABITS_CHOICES = [
        ('0', 'Sehat'),
        ('1', 'Sedang'),
        ('2', 'Tidak Sehat'),
    ]   

    YES_NO_CHOICES = [
        ('1', 'Ya'),
        ('0', 'Tidak'),
    ]

    # Form fields
    gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        label="Jenis Kelamin",
        required=True,
        widget=forms.RadioSelect(attrs={'class': 'radio-group'})
    )

    age = forms.IntegerField(
        label="Usia",
        required=True,
        widget=forms.NumberInput(attrs={
            'class': 'mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm',
            'placeholder': 'Masukkan usia Anda'
        })
    )

    work_pressure = forms.ChoiceField(
        choices=PRESSURE_CHOICES,
        label="Tekanan Kerja",
        required=True,
        widget=forms.RadioSelect(attrs={'class': 'radio-group'})
    )

    job_satisfaction = forms.ChoiceField(
        choices=SATISFACTION_CHOICES,
        label="Kepuasan Kerja",
        required=True,
        widget=forms.RadioSelect(attrs={'class': 'radio-group'})
    )

    financial_stress = forms.ChoiceField(
        choices=STRESS_FINANCIAL_CHOICES,
        label="Stres Keuangan",
        required=True,
        widget=forms.RadioSelect(attrs={'class': 'radio-group'})
    )

    sleep_duration = forms.ChoiceField(
        choices=SLEEP_DURATION_CHOICES,
        label="Durasi Tidur",
        required=True,
        widget=forms.RadioSelect(attrs={'class': 'radio-group'})
    )

    dietary_habits = forms.ChoiceField(
        choices=DIETARY_HABITS_CHOICES,
        label="Kebiasaan Makan",
        required=True,
        widget=forms.RadioSelect(attrs={'class': 'radio-group'})
    )

    suicidal_thoughts = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        label="Apakah Anda pernah memiliki pikiran untuk bunuh diri?",
        required=True,
        widget=forms.RadioSelect(attrs={'class': 'radio-group'})
    )

    work_hours = forms.IntegerField(
        label="Jam Kerja",
        required=True,
        widget=forms.NumberInput(attrs={
            'class': 'mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm',
            'placeholder': 'Masukkan jam kerja per hari'
        })
    )

    family_history_of_mental_illness = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        label="Riwayat Keluarga Penyakit Mental",
        required=True,
        widget=forms.RadioSelect(attrs={'class': 'radio-group'})
    )


class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True, label="Email")
    first_name = forms.CharField(max_length=30, required=True, label="Nama Depan")
    last_name = forms.CharField(max_length=30, required=True, label="Nama Belakang")

    class Meta:
        model = User
        fields = ("username", "first_name", "last_name", "email", "password1", "password2")

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        user.first_name = self.cleaned_data["first_name"]
        user.last_name = self.cleaned_data["last_name"]
        if commit:
            user.save()
        return user