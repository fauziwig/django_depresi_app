from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class MyForm(forms.Form):

    # Dropdown choices
    GENDER_CHOICES = [
        ('', 'Pilih Gender'),
        ('1', 'Laki Laki'),
        ('0', 'Perempuan'),
    ]

    PRESSURE_CHOICES = [
        ('', 'Pilih Opsi'),
        ('1', 'Sangat Rendah'),
        ('2', 'Rendah'),
        ('3', 'Medium'),
        ('4', 'Tinggi'),
        ('5', 'Sangat Tinggi'),
    ]

    SLEEP_DURATION_CHOICES = [
        ('', 'Pilih Tipe'),
        ('0', 'Kurang dari 5 Jam'),
        ('1', '5-6 Jam'),
        ('2', '7-8 Jam'),
        ('3', 'Lebih dari 8 Jam'),
    ]

    DIETARY_HABITS_CHOICES = [
        ('', 'Pilih Tipe'),
        ('0', 'Sehat'),
        ('1', 'Sedang / Medium'),
        ('2', 'Tidak Sehat'),
    ]

    YES_NO_CHOICES = [
        ('', 'Pilih Status'),
        ('1', 'Yes'),
        ('0', 'No'),
    ]

    # Form fields
    gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        label="Jenis Kelamin",
        required=True,
        widget=forms.Select(attrs={'class': 'mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm'})
    )

    age = forms.IntegerField(
        label="Usia (Age)",
        required=True,
        widget=forms.NumberInput(attrs={
            'class': 'mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm',
            'placeholder': 'Masukkan Usia'
        })
    )

    work_pressure = forms.ChoiceField(
        choices=PRESSURE_CHOICES,
        label="Tekanan Kerja",
        required=True,
        widget=forms.Select(attrs={'class': 'mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm'})
    )

    job_satisfaction = forms.ChoiceField(
        choices=PRESSURE_CHOICES,
        label="Kepuasan Kerja",
        required=True,
        widget=forms.Select(attrs={'class': 'mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm'})
    )

    financial_stress = forms.ChoiceField(
        choices=PRESSURE_CHOICES,
        label="Tekanan Finansial",
        required=True,
        widget=forms.Select(attrs={'class': 'mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm'})
    )

    sleep_duration = forms.ChoiceField(
        choices=SLEEP_DURATION_CHOICES,
        label="Durasi Tidur Per Hari",
        required=True,
        widget=forms.Select(attrs={'class': 'mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm'})
    )

    dietary_habits = forms.ChoiceField(
        choices=DIETARY_HABITS_CHOICES,
        label="Kebiasaan Makan",
        required=True,
        widget=forms.Select(attrs={'class': 'mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm'})
    )

    suicidal_thoughts = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        label="Pernahkah Anda memiliki pikiran untuk bunuh diri??",
        required=True,
        widget=forms.Select(attrs={'class': 'mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm'})
    )

    work_hours = forms.IntegerField(
        label="Jam Kerja Per Hari",
        required=True,
        widget=forms.NumberInput(attrs={
            'class': 'mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm',
            'placeholder': 'Masukkan jam kerja per hari'
        })
    )

    family_history_of_mental_illness = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        label="Riwayat Penyakit Mental Keluarga",
        required=True,
        widget=forms.Select(attrs={'class': 'mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm'})
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