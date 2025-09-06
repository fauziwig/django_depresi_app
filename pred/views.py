from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect
from .forms import MyForm, CustomUserCreationForm
from django.template import loader
from .models import FormSubmission
from django.utils import timezone
from django.contrib.auth import authenticate, login, logout
from datetime import datetime
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from django.contrib import messages
import joblib
import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings
from django.views.decorators.http import require_POST
from django.urls import reverse
from django.contrib.auth.forms import UserCreationForm 
from django.contrib.auth import login # Tambahkan ini jika Anda ingin auto-login

# Helper function to check if user is admin
def is_admin(user):
    return user.is_authenticated and (user.is_superuser or user.is_staff)

# Helper function to check if user is expert
def is_expert(user):
    return user.is_authenticated and hasattr(user, 'groups') and user.groups.filter(name='Expert').exists()

# Landing page view
def landing_view(request):
    """
    Display the landing page with system information and call-to-action.
    This is purely informational and doesn't affect any core system functionality.
    """
    return render(request, 'landing.html')

def get_data_summary_for_submission(submission):
    """
    Mengembalikan list of dict berisi label pertanyaan dan jawaban human-readable
    untuk satu diagnosis submission, sesuai forms.py.
    """
    # Ambil data jawaban (misal field JSON/dict di model)
    # Misal: submission.form_data adalah dict hasil pengisian form
    # Ambil data jawaban (misal field JSON/dict di model)
    form_data = submission.form_data if hasattr(submission, 'form_data') and submission.form_data else {}

    # PATCH: Jika form_data kosong, ambil dari field model (untuk data lama)
    if not form_data:
        form_data = {
            'gender': getattr(submission, 'gender', ''),
            'age': getattr(submission, 'age', ''),
            'work_hours': getattr(submission, 'work_hours', ''),
            'sleep_duration': getattr(submission, 'sleep_duration', ''),
            'work_pressure': getattr(submission, 'work_pressure', ''),
            'job_satisfaction': getattr(submission, 'job_satisfaction', ''),
            'financial_stress': getattr(submission, 'financial_stress', ''),
            'dietary_habits': getattr(submission, 'dietary_habits', ''),
            'suicidal_thoughts': getattr(submission, 'suicidal_thoughts', ''),
            'family_history_of_mental_illness': getattr(submission, 'family_history_of_mental_illness', ''),
        }

    print("DEBUG form_data:", form_data)
    
    # Label pertanyaan sesuai forms.py
    field_labels = {
        'gender': 'Jenis Kelamin',
        'age': 'Usia',
        'work_hours': 'Rata-rata, berapa jam kerja yang Anda habiskan dalam sehari?',
        'sleep_duration': 'Rata-rata, berapa jam tidur Anda dalam semalam?',
        'work_pressure': 'Bagaimana Anda menilai tingkat tekanan dalam pekerjaan Anda akhir-akhir ini?',
        'job_satisfaction': 'Seberapa puas Anda dengan pekerjaan Anda saat ini?',
        'financial_stress': 'Bagaimana perasaan Anda mengenai kondisi keuangan Anda saat ini?',
        'dietary_habits': 'Bagaimana Anda menggambarkan pola makan Anda sehari-hari?',
        'suicidal_thoughts': 'Apakah Anda pernah memiliki pikiran untuk bunuh diri?',
        'family_history_of_mental_illness': 'Apakah ada anggota keluarga Anda yang memiliki riwayat gangguan kesehatan mental?',
    }

    # Helper display sesuai forms.py
    from .views import (
        get_gender_display, get_pressure_display, get_satisfaction_display,
        get_financial_stress_display, get_sleep_duration_display,
        get_dietary_habits_display, get_pernah_tidak_display, get_ada_tidak_display
    )

    summary = []
    for field in [
        'gender', 'age', 'work_hours', 'sleep_duration', 'work_pressure',
        'job_satisfaction', 'financial_stress', 'dietary_habits',
        'suicidal_thoughts', 'family_history_of_mental_illness'
    ]:
        label = field_labels.get(field, field)
        value = form_data.get(field, '')
        # Mapping ke display
        if field == 'gender':
            display = get_gender_display(value)
        elif field == 'work_pressure':
            display = get_pressure_display(value)
        elif field == 'job_satisfaction':
            display = get_satisfaction_display(value)
        elif field == 'financial_stress':
            display = get_financial_stress_display(value)
        elif field == 'sleep_duration':
            display = get_sleep_duration_display(value)
        elif field == 'dietary_habits':
            display = get_dietary_habits_display(value)
        elif field == 'suicidal_thoughts':
            display = get_pernah_tidak_display(value)
        elif field == 'family_history_of_mental_illness':
            display = get_ada_tidak_display(value)
        else:
            display = value
        summary.append({'label': label, 'value': display})
    return summary

def get_result_text_for_submission(submission):
    """
    Mengembalikan string hasil diagnosis untuk satu submission,
    misal: 'Terindikasi Mengalami Depresi (Diagnosis : 1)' atau 'Tidak Terindikasi Mengalami Depresi (Diagnosis : 0)'
    """
    # Asumsi: submission.prediction_result berisi 'Positif' atau 'Negatif'
    if hasattr(submission, 'prediction_result'):
        if str(submission.prediction_result).lower() == 'positif':
            return 'Terindikasi Mengalami Depresi (Diagnosis : 1)'
        elif str(submission.prediction_result).lower() == 'negatif':
            return 'Tidak Terindikasi Mengalami Depresi (Diagnosis : 0)'
        else:
            return str(submission.prediction_result)
    return 'Tidak Diketahui'

@login_required
def user(request):
    user_is_admin_or_expert = is_admin(request.user) or is_expert(request.user)

    if user_is_admin_or_expert:
        # Admin/Expert melihat semua riwayat
        diagnosis_history = FormSubmission.objects.all().order_by('-submitted_at')
    else:
        # Pengguna biasa hanya melihat riwayatnya sendiri
        diagnosis_history = FormSubmission.objects.filter(user=request.user).order_by('-submitted_at')

    for diag in diagnosis_history:
        diag.data_summary = get_data_summary_for_submission(diag)
        diag.result_text = get_result_text_for_submission(diag)
    context = {
        'user': request.user,
        'user_is_admin_or_expert': user_is_admin_or_expert,
        'diagnosis_history': diagnosis_history,
    }
    return render(request, 'user.html', context)


# Helper function to check if user has admin-level access (admin or expert)
def has_admin_access(user):
    return is_admin(user) or is_expert(user)

# Data transformation helper function
def transform_form_data_for_model(form_data):
    """
    Transform form data from user input format to the format expected by the ML model

    Input format (from user):
    {'gender': '1', 'age': 20, 'work_pressure': '2', ...}

    Output format (for model):
    {'age': 20, 'work_pressure': 2, ..., 'gender_Female': 0, 'gender_Male': 1}
    """
    # Convert gender to the format expected by the model
    gender_value = int(form_data['gender'])
    gender_male = 1 if gender_value == 1 else 0
    gender_female = 1 if gender_value == 0 else 0

    # Create the transformed data dictionary
    transformed_data = {
        'age': int(form_data['age']),
        'work_pressure': int(form_data['work_pressure']),
        'job_satisfaction': int(form_data['job_satisfaction']),
        'sleep_duration': int(form_data['sleep_duration']),
        'dietary_habits': int(form_data['dietary_habits']),
        'suicidal_thoughts': int(form_data['suicidal_thoughts']),
        'work_hours': int(form_data['work_hours']),
        'financial_stress': int(form_data['financial_stress']),
        'family_history_of_mental_illness': int(form_data['family_history_of_mental_illness']),
        'gender_Female': gender_female,
        'gender_Male': gender_male
    }



    return transformed_data

# Display helper functions to convert database values to human-readable text
def get_gender_display(value):
    gender_choices = {
        '1': 'Laki-laki',
        '0': 'Perempuan'
    }
    return gender_choices.get(str(value), 'Tidak Diketahui')

def get_pressure_display(value):
    pressure_choices = {
        '1': 'Sangat Santai',
        '2': 'Santai',
        '3': 'Cukup Sibuk',
        '4': 'Tertekan',
        '5': 'Sangat Tertekan'
    }
    return pressure_choices.get(str(value), 'Tidak Diketahui')

def get_satisfaction_display(value):
    satisfaction_choices = {
        '1': 'Sangat Tidak Puas',
        '2': 'Tidak Puas',
        '3': 'Cukup Puas',
        '4': 'Puas',
        '5': 'Sangat Puas'
    }
    return satisfaction_choices.get(str(value), 'Tidak Diketahui')

def get_financial_stress_display(value):
    financial_stress_choices = {
        '1': 'Sangat Aman',
        '2': 'Aman',
        '3': 'Netral',
        '4': 'Khawatir',
        '5': 'Sangat Khawatir'
    }
    return financial_stress_choices.get(str(value), 'Tidak Diketahui')

def get_sleep_duration_display(value):
    sleep_choices = {
        '0': '5-6 Jam',
        '1': '7-8 Jam',
        '2': 'Kurang dari 5 Jam',
        '3': 'Lebih dari 8 Jam'
    }
    return sleep_choices.get(str(value), 'Tidak Diketahui')

def get_dietary_habits_display(value):
    dietary_choices = {
        '0': 'Sehat',
        '1': 'Sedang',
        '2': 'Tidak Sehat'
    }
    return dietary_choices.get(str(value), 'Tidak Diketahui')

def get_pernah_tidak_display(value):
    pernah_tidak_choices = {
        '1': 'Pernah',
        '0': 'Tidak Pernah'
    }
    return pernah_tidak_choices.get(str(value), 'Tidak Diketahui')

def get_ada_tidak_display(value):
    ada_tidak_choices = {
        '1': 'Ada',
        '0': 'Tidak Ada'
    }
    return ada_tidak_choices.get(str(value), 'Tidak Diketahui')
# Load the ML model
def load_depression_model():
    """Load the depression prediction model"""
    try:
        model_path = os.path.join(settings.BASE_DIR, 'svm_model.joblib')
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_depression_scaler():
    """Load the depression prediction scaler"""
    try:
        scaler_path = os.path.join(settings.BASE_DIR, 'load_scaler.joblib')
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

# Prediction function
def predict_depression(form_data):
    """
    Predict depression based on form data
    Returns: dict with prediction result and probability
    """
    
    try:
        model = load_depression_model()
        scaler = load_depression_scaler()

        if model is None:
            return {
                'prediction': 'Error',
                'probability': 0,
                'message': 'Model could not be loaded'
            }

        if scaler is None:
            return {
                'prediction': 'Error',
                'probability': 0,
                'message': 'Scaler could not be loaded'
            }
        

        # Transform form data to the format expected by the ML model
        transformed_data = transform_form_data_for_model(form_data)
    
        # Create DataFrame with the transformed data
        input_data = pd.DataFrame([transformed_data])
        
        input_data_scaled = scaler.transform(input_data)

        # Log the transformed data for debugging
        print("\n🔄 Data Transformation for ML Model:")
        print(f"  Original gender: {form_data['gender']} ({'Male' if int(form_data['gender']) == 1 else 'Female'})")
        print(f"  Transformed to: gender_Male={transformed_data['gender_Male']}, gender_Female={transformed_data['gender_Female']}")
        print("\n📊 Final DataFrame for Model:")
        for col, val in transformed_data.items():
            print(f"  {col}: {val}")
        print()

        # Make prediction
        print('input data : ',type(transformed_data))
        prediction = model.predict(input_data_scaled)
        print('prediction: ',prediction[0])

        # Get prediction probability if available
        try:
            prediction_proba = model.predict_proba(input_data_scaled)[0]
            probability = max(prediction_proba) * 100  # Convert to percentage
        except:
            probability = 0

        # Interpret the result
        if prediction == 1:
            result = {
                'prediction': 'Positif',
                'probability': probability,
                'message': 'Model menunjukkan kemungkinan depresi yang lebih tinggi. Silakan berkonsultasi dengan profesional kesehatan.',
                'class': 'text-danger',
                'color': '#dc3545'
            }
        else:
            result = {
                'prediction': 'Negatif',
                'probability': probability,
                'message': 'Model menunjukkan kemungkinan depresi yang lebih rendah. Namun, ini bukan diagnosis medis.',
                'class': 'text-success',
                'color': '#28a745'
            }

        return result

    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            'prediction': 'Error',
            'probability': 0,
            'message': f'Terjadi kesalahan saat prediksi: {str(e)}',
            'class': 'text-warning',
            'color': '#ffc107'
        }

# Cosine similarity function
def find_similar_cases(form_data):
    """
    Find the most similar cases from the dataset using cosine similarity
    Returns: dict with similarity results
    """
    try:
        # Load the dataset
        dataset_path = os.path.join(settings.BASE_DIR, 'dataset_processed.csv')
        if not os.path.exists(dataset_path):
            return {
                'success': False,
                'error': 'Dataset file not found',
                'message': 'File dataset tidak ditemukan'
            }

        df = pd.read_csv(dataset_path)
        print(f"📊 Dataset loaded: {len(df)} rows, columns: {list(df.columns)}")

        # Transform user data to match the format expected by the model
        transformed_data = transform_form_data_for_model(form_data)

        # Prepare user input vector (excluding depression column if it exists)
        # Use the same order as the ML model expects
        user_features = [
            transformed_data['age'],
            transformed_data['work_pressure'],
            transformed_data['job_satisfaction'],
            transformed_data['financial_stress'],
            transformed_data['sleep_duration'],
            transformed_data['dietary_habits'],
            transformed_data['suicidal_thoughts'],
            transformed_data['work_hours'],
            transformed_data['family_history_of_mental_illness'],
            transformed_data['gender_Female'],
            transformed_data['gender_Male']
        ]

        user_vector = np.array([user_features])
        print(f"🔍 User vector shape: {user_vector.shape}")
        print(f"🔍 User features: {user_features}")

        # Check if dataset has the new format (gender_Male, gender_Female) or old format (gender)
        if 'gender_Male' in df.columns and 'gender_Female' in df.columns:
            # New format - use gender_Male and gender_Female columns
            feature_columns = ['age', 'work_pressure', 'job_satisfaction', 'financial_stress',
                              'sleep_duration', 'dietary_habits', 'suicidal_thoughts',
                              'work_hours', 'family_history_of_mental_illness',
                              'gender_Female', 'gender_Male']
        else:
            # Old format - convert gender column to gender_Male and gender_Female
            print("📝 Converting old dataset format to new format...")
            df['gender_Male'] = (df['gender'] == 1).astype(int)
            df['gender_Female'] = (df['gender'] == 0).astype(int)
            feature_columns = ['age', 'work_pressure', 'job_satisfaction', 'financial_stress',
                              'sleep_duration', 'dietary_habits', 'suicidal_thoughts',
                              'work_hours', 'family_history_of_mental_illness',
                              'gender_Female', 'gender_Male']

        # Ensure all required columns exist
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            return {
                'success': False,
                'error': f'Missing columns in dataset: {missing_columns}',
                'message': f'Kolom dataset tidak lengkap: {missing_columns}'
            }

        dataset_vectors = df[feature_columns].values
        print(f"📊 Dataset vectors shape: {dataset_vectors.shape}")

        # Calculate cosine similarity
        similarities = cosine_similarity(user_vector, dataset_vectors)[0]

        # Find the most similar case
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        best_match_row = df.iloc[best_match_idx]

        # Find top 3 similar cases
        top_indices = np.argsort(similarities)[-3:][::-1]  # Top 3 in descending order
        top_matches = []

        for idx in top_indices:
            match_data = df.iloc[idx]
            similarity_score = similarities[idx]

            # Handle gender display for both old and new formats
            if 'gender' in match_data:
                gender_display = get_gender_display(match_data['gender'])
            else:
                # Use gender_Male and gender_Female columns
                gender_display = 'Laki-laki' if match_data['gender_Male'] == 1 else 'Perempuan'

            top_matches.append({
                'similarity': float(similarity_score * 100),  # Convert to percentage and ensure float
                'gender': gender_display,
                'age': int(match_data['age']),  # Convert to regular int
                'work_pressure': get_pressure_display(match_data['work_pressure']),
                'job_satisfaction': get_pressure_display(match_data['job_satisfaction']),
                'sleep_duration': get_sleep_duration_display(match_data['sleep_duration']),
                'dietary_habits': get_dietary_habits_display(match_data['dietary_habits']),
                'suicidal_thoughts': get_pernah_tidak_display(match_data['suicidal_thoughts']),
                'work_hours': int(match_data['work_hours']),
                'financial_stress': get_pressure_display(match_data['financial_stress']),
                'family_history': get_ada_tidak_display(match_data['family_history_of_mental_illness']),
                'depression': 'Positif' if match_data['depression'] == 1 else 'Negatif'
            })

        # Handle gender display for best match
        if 'gender' in best_match_row:
            best_match_gender = get_gender_display(best_match_row['gender'])
        else:
            # Use gender_Male and gender_Female columns
            best_match_gender = 'Laki-laki' if best_match_row['gender_Male'] == 1 else 'Perempuan'

        return {
            'success': True,
            'best_similarity': float(best_similarity * 100),  # Convert to percentage and ensure float
            'best_match_idx': int(best_match_idx),  # Store the index
            'best_similarity_score': float(best_similarity * 100),  # For API compatibility
            'best_match_index': int(best_match_idx),  # For API compatibility
            'best_match': {
                'gender': best_match_gender,
                'age': int(best_match_row['age']),  # Convert to regular int
                'work_pressure': get_pressure_display(best_match_row['work_pressure']),
                'job_satisfaction': get_pressure_display(best_match_row['job_satisfaction']),
                'sleep_duration': get_sleep_duration_display(best_match_row['sleep_duration']),
                'dietary_habits': get_dietary_habits_display(best_match_row['dietary_habits']),
                'suicidal_thoughts': get_pernah_tidak_display(best_match_row['suicidal_thoughts']),
                'work_hours': int(best_match_row['work_hours']),
                'financial_stress': get_pressure_display(best_match_row['financial_stress']),
                'family_history': get_ada_tidak_display(best_match_row['family_history_of_mental_illness']),
                'depression': 'Positif' if best_match_row['depression'] == 1 else 'Negatif'
            },
            'top_matches': top_matches,
            'total_cases': int(len(df))  # Convert to regular int
        }

    except Exception as e:
        print(f"❌ Similarity calculation error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'message': 'Tidak dapat menghitung kemiripan dengan dataset'
        }


def diagnosis(request):
    template = loader.get_template("diagnosis_form.html")

    if request.method == 'POST':
        # Log the raw request data
        print("=" * 60)
        print("🔍 FORM SUBMISSION REQUEST LOG")
        print("=" * 60)
        print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Method: {request.method}")
        print(f"📍 Path: {request.path}")
        print(f"🖥️  User Agent: {request.META.get('HTTP_USER_AGENT', 'Unknown')}")
        print(f"📡 Remote IP: {request.META.get('REMOTE_ADDR', 'Unknown')}")

        print("\n📦 Raw POST Data:")
        for key, value in request.POST.items():
            print(f"  {key}: {value}")

        form = MyForm(request.POST)
        if form.is_valid():
            print("\n✅ Form Validation: PASSED")
            print("\n🧹 Cleaned Form Data:")
            for key, value in form.cleaned_data.items():
                print(f"  {key}: {value}")
        else:
            print("\n❌ Form Validation: FAILED")
            print("\n🚨 Form Errors:")
            for field, errors in form.errors.items():
                print(f"  {field}: {errors}")
            print("=" * 60)
            return render(request, 'base.html', {'form': form})

        print("=" * 60)

        if form.is_valid():
            # Save form data to database with user information
            submission = FormSubmission.objects.create(
                user=request.user if request.user.is_authenticated else None,
                gender=form.cleaned_data['gender'],
                age=form.cleaned_data['age'],
                work_pressure=form.cleaned_data['work_pressure'],
                job_satisfaction=form.cleaned_data['job_satisfaction'],
                sleep_duration=form.cleaned_data['sleep_duration'],
                dietary_habits=form.cleaned_data['dietary_habits'],
                suicidal_thoughts=form.cleaned_data['suicidal_thoughts'],
                work_hours=form.cleaned_data['work_hours'],
                financial_stress=form.cleaned_data['financial_stress'],
                family_history_of_mental_illness=form.cleaned_data['family_history_of_mental_illness']
            )

            # Perform ML prediction
            prediction_result = predict_depression(form.cleaned_data)

            print("Prediction resukt: ", prediction_result)
            # Perform cosine similarity analysis
            print("\n🔍 Starting similarity analysis...")
            try:
                similarity_result = find_similar_cases(form.cleaned_data)
                print(f"✅ Similarity analysis completed: {similarity_result.get('success', False)}")
                if not similarity_result.get('success', False):
                    print(f"❌ Similarity analysis failed: {similarity_result.get('message', 'Unknown error')}")
            except Exception as e:
                print(f"❌ Similarity analysis exception: {str(e)}")
                similarity_result = {
                    'success': False,
                    'error': str(e),
                    'message': 'Terjadi kesalahan dalam analisis kemiripan'
                }
            

            # Update the submission with prediction results
            submission.prediction_result = prediction_result.get('prediction', 'Unknown')
            submission.prediction_probability = prediction_result.get('probability', 0)
            submission.prediction_message = prediction_result.get('message', '')

            # Update with similarity results if available
            if similarity_result.get('success', False):
                submission.similarity_score = float(similarity_result.get('best_similarity', 0))
                # Store the index of the best match in the dataset
                if 'best_match' in similarity_result:
                    submission.similar_case_id = int(similarity_result.get('best_match_idx', 0))

            submission.form_data = form.cleaned_data 
            submission.save()

            # Store form data, prediction result, and similarity result in session
            request.session['form_data'] = form.cleaned_data
            request.session['submission_id'] = submission.id
            request.session['prediction_result'] = prediction_result
            request.session['similarity_result'] = similarity_result
            return redirect('results_url')
    else:
        form = MyForm()

    # Pass user authentication info to template
    context = {
        'form': form,
        'user': request.user,
        'is_authenticated': request.user.is_authenticated,
        'is_admin': is_admin(request.user),
        'is_expert': is_expert(request.user),
        'has_admin_access': has_admin_access(request.user)
    }
    return HttpResponse(template.render(context, request))

    # return render(request, 'my_template.html', {'form': form})
    # return HttpResponse("Hello, world. You're at the pred index.")




def results_view(request):
    # Get form data, prediction result, and similarity result from session
    form_data = request.session.get('form_data', {})
    submission_id = request.session.get('submission_id')
    prediction_result = request.session.get('prediction_result', {})
    similarity_result = request.session.get('similarity_result', {})

    if not form_data:
        return HttpResponse("<h1>Data tidak ditemukan</h1><p><a href='/diagnosis'>Kembali ke formulir</a></p>")

    # Get the submission time if we have the submission ID
    submission_time = None
    if submission_id:
        try:
            submission = FormSubmission.objects.get(id=submission_id)
            submission_time = timezone.localtime(submission.submitted_at).strftime('%d %B %Y, %H:%M:%S WIB')
        except FormSubmission.DoesNotExist:
            pass

    # Diagnosis result text and desc
    diagnosis_result_text = ""
    diagnosis_result_desc = ""
    if prediction_result.get('prediction', '').lower() == 'positif':
        diagnosis_result_text = '<span style="color:#b46fc2;">Terindikasi Mengalami Depresi</span> (Diagnosis : 1)'
        diagnosis_result_desc = "Berdasarkan jawaban yang Anda berikan, sistem mendeteksi adanya gejala yang signifikan untuk diagnosis depresi. Segera konsultasikan dengan profesional kesehatan mental."
    elif prediction_result.get('prediction', '').lower() == 'negatif':
        diagnosis_result_text = 'Tidak Terindikasi Mengalami Depresi <span style="color:#b46fc2;">(Diagnosis : 0)</span>'
        diagnosis_result_desc = "Berdasarkan jawaban yang Anda berikan, sistem tidak menemukan adanya gejala yang signifikan untuk diagnosis depresi saat ini. Ini adalah indikasi yang sangat positif mengenai kondisi kesehatan mental Anda."
    else:
        diagnosis_result_text = prediction_result.get('prediction', 'Tidak Diketahui')
        diagnosis_result_desc = prediction_result.get('message', '')

    # --- Tambahkan log hasil prediksi ke server log ---
    import logging
    logger = logging.getLogger("django")
    logger.info(
        "\n🧠 Prediction Result:\n"
        f"  Prediction: {prediction_result.get('prediction', '-')}\n"
        f"  Probability: {prediction_result.get('probability', '-')}\n"
        f"  Message: {prediction_result.get('message', '-')}\n"
    )

    # Data summary for table
    field_labels = {
        'gender': 'Jenis Kelamin',
        'age': 'Usia',
        'work_pressure': 'Bagaimana Anda menilai tingkat tekanan dalam pekerjaan Anda akhir-akhir ini?',
        'job_satisfaction': 'Seberapa puas Anda dengan pekerjaan Anda saat ini?',
        'financial_stress': 'Bagaimana perasaan Anda mengenai kondisi keuangan Anda saat ini?',
        'sleep_duration': 'Rata-rata, berapa jam tidur Anda dalam semalam?',
        'dietary_habits': 'Bagaimana Anda menggambarkan pola makan Anda sehari-hari?',
        'suicidal_thoughts': 'Apakah Anda pernah memiliki pikiran untuk bunuh diri?',
        'work_hours': 'Rata-rata, berapa jam kerja yang Anda habiskan dalam sehari?',
        'family_history_of_mental_illness': 'Apakah ada anggota keluarga Anda yang memiliki riwayat gangguan kesehatan mental?'
    }
    data_summary = []
    for field_name, field_value in form_data.items():
        label = field_labels.get(field_name, field_name.title())
        # Convert database values to display values sesuai helper & forms.py
        if field_name == 'gender':
            display_value = get_gender_display(field_value)
        elif field_name == 'work_pressure':
            display_value = get_pressure_display(field_value)
        elif field_name == 'job_satisfaction':
            display_value = get_satisfaction_display(field_value)
        elif field_name == 'financial_stress':
            display_value = get_financial_stress_display(field_value)
        elif field_name == 'sleep_duration':
            display_value = get_sleep_duration_display(field_value)
        elif field_name == 'dietary_habits':
            display_value = get_dietary_habits_display(field_value)
        elif field_name == 'suicidal_thoughts':
            display_value = get_pernah_tidak_display(field_value)
        elif field_name == 'family_history_of_mental_illness':
            display_value = get_ada_tidak_display(field_value)
        else:
            display_value = field_value
        data_summary.append({'label': label, 'value': display_value})

    # Clear the session data after displaying
    if 'form_data' in request.session:
        del request.session['form_data']

    context = {
        'diagnosis_result_text': diagnosis_result_text,
        'diagnosis_result_desc': diagnosis_result_desc,
        'data_summary': data_summary,
        'similarity_result': similarity_result,
        'submission_time': submission_time,
    }
    return render(request, 'diagnosis_result.html', context)





def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f'Selamat datang, {user.username}!') # Gunakan f-string
            return redirect('landing_view')
        else:
            messages.error(request, 'Username atau password salah.')

    return render(request, 'login.html')


# def register_view(request):
#     if request.method == 'POST':
#         form = CustomUserCreationForm(request.POST)
#         if form.is_valid():
#             # Logika ini tidak dieksekusi jika form tidak valid
#             user = form.save()
#             username = form.cleaned_data.get('username')
#             messages.success(request, f'Akun berhasil dibuat untuk {username}!')
#             login(request, user)
#             return redirect('landing_view')
#         else:
#             # Tambahkan baris ini untuk melihat error di terminal Anda
#             print(form.errors) 
#             # Pesan error juga akan ditampilkan di template jika kode HTMLnya sudah benar
#     else:
#         form = CustomUserCreationForm()

#     return render(request, 'register.html', {'form': form})




def register_view(request):
    if request.method == "POST": 
        form = UserCreationForm(request.POST) 
        if form.is_valid(): 
            user = form.save() # Simpan user ke dalam variabel
            messages.success(request, f'Akun berhasil dibuat untuk {user.username}!') # Gunakan f-string
            # login(request, user) # Opsional: auto-login
            return redirect("landing_view")
    else:
        form = UserCreationForm()
    return render(request, "register.html", { "form": form })


def logout_view(request):
    logout(request)
    messages.success(request, 'Anda telah berhasil logout.')
    return redirect('landing_view')


@user_passes_test(has_admin_access)
def admin_dashboard(request):
    # Get statistics
    total_submissions = FormSubmission.objects.count()
    total_users = User.objects.count()
    reused_submissions = FormSubmission.objects.filter(is_reused_in_dataset=True).count()
    recent_submissions = FormSubmission.objects.order_by('-submitted_at')[:5]

    # Determine user role
    user_is_admin = is_admin(request.user)

    dashboard_title = "Admin Dashboard" if user_is_admin else "Expert Dashboard"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{dashboard_title} - Sistem Prediksi Depresi</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                min-height: 100vh;
                padding: 20px;
            }}

            .main-container {{
                max-width: 70%;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }}

            .header {{
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white;
                padding: 30px;
                text-align: center;
                position: relative;
            }}

            .header h1 {{
                font-size: 2.2em;
                font-weight: 700;
                margin-bottom: 10px;
            }}

            .header p {{
                font-size: 1.1em;
                opacity: 0.9;
            }}

            .nav-bar {{
                background: #f8f9fa;
                padding: 15px 30px;
                border-bottom: 1px solid #e9ecef;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 15px;
            }}

            .user-info {{
                display: flex;
                align-items: center;
                gap: 10px;
                font-weight: 600;
                color: #495057;
            }}

            .nav-links {{
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
            }}

            .nav-link {{
                text-decoration: none;
                padding: 8px 16px;
                border-radius: 25px;
                font-weight: 500;
                transition: all 0.3s ease;
                display: inline-flex;
                align-items: center;
                gap: 5px;
                font-size: 0.9em;
            }}

            .nav-link:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            }}

            .nav-link.primary {{ background: #6366f1; color: white; }}
            .nav-link.primary:hover {{ background: #4f46e5; }}
            .nav-link.success {{ background: #8b5cf6; color: white; }}
            .nav-link.success:hover {{ background: #7c3aed; }}
            .nav-link.danger {{ background: #a855f7; color: white; }}
            .nav-link.danger:hover {{ background: #9333ea; }}

            .content-section {{
                padding: 40px;
            }}

            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 25px;
                margin-bottom: 40px;
            }}

            .stat-card {{
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                color: white;
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
                transition: all 0.3s ease;
            }}

            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 12px 35px rgba(99, 102, 241, 0.4);
            }}

            .stat-card.users {{
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
            }}

            .stat-card.reused {{
                background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                box-shadow: 0 8px 25px rgba(245, 158, 11, 0.3);
            }}

            .stat-number {{
                font-size: 2.5em;
                font-weight: 700;
                margin-bottom: 10px;
            }}

            .stat-label {{
                font-size: 1em;
                opacity: 0.9;
                font-weight: 500;
            }}
            .admin-actions {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}

            .action-button {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                padding: 18px 25px;
                background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                color: white;
                text-decoration: none;
                border-radius: 15px;
                text-align: center;
                font-weight: 600;
                font-size: 1em;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
            }}

            .action-button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4);
                color: white;
                text-decoration: none;
            }}

            .action-button.danger {{
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
            }}

            .action-button.danger:hover {{
                box-shadow: 0 8px 25px rgba(239, 68, 68, 0.4);
            }}

            .action-button.success {{
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
            }}

            .action-button.success:hover {{
                box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
            }}

            .recent-submissions {{
                margin-top: 40px;
            }}

            .section-title {{
                font-size: 1.5em;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}

            .submission-item {{
                background: #f8f9fa;
                padding: 20px;
                margin-bottom: 15px;
                border-radius: 15px;
                border-left: 5px solid #6366f1;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
            }}

            .submission-item:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            }}

            .submission-header {{
                font-weight: 600;
                color: #6366f1;
                margin-bottom: 10px;
                font-size: 1.1em;
            }}

            .submission-details {{
                color: #495057;
                font-size: 0.9em;
                line-height: 1.5;
            }}

            @media (max-width: 768px) {{
                body {{
                    padding: 10px;
                }}

                .main-container {{
                    border-radius: 15px;
                }}

                .header {{
                    padding: 20px;
                }}

                .header h1 {{
                    font-size: 1.8em;
                }}

                .nav-bar {{
                    padding: 15px 20px;
                    flex-direction: column;
                    align-items: stretch;
                    gap: 10px;
                }}

                .nav-links {{
                    justify-content: center;
                }}

                .content-section {{
                    padding: 30px 20px;
                }}

                .stats-grid {{
                    grid-template-columns: 1fr;
                }}

                .admin-actions {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="main-container">
            <!-- Header Section -->
            <div class="header">
                <h1>{'⚙️ Dashboard Admin' if user_is_admin else '🔬 Dashboard Ahli'}</h1>
                <p>Kelola dan Pantau Sistem Prediksi Depresi</p>
            </div>

            <!-- Navigation Bar -->
            <div class="nav-bar">
                <div class="user-info">
                    <span>{'👑' if user_is_admin else '🎓'}</span>
                    <span>{request.user.get_full_name() or request.user.username}</span>
                </div>
                <div class="nav-links">
                    <a href="/pred/" class="nav-link primary">🏠 Formulir Utama</a>
                    <a href="/pred/history/" class="nav-link success">📋 Lihat Riwayat</a>
                    <a href="/pred/logout/" class="nav-link danger">🚪 Keluar</a>
                </div>
            </div>

            <!-- Content Section -->
            <div class="content-section">

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{total_submissions}</div>
                        <div class="stat-label">📋 Total Pengiriman</div>
                    </div>
                    <div class="stat-card users">
                        <div class="stat-number">{total_users}</div>
                        <div class="stat-label">👥 Total Pengguna</div>
                    </div>
                    <div class="stat-card reused">
                        <div class="stat-number">{reused_submissions}</div>
                        <div class="stat-label">🔄 Reused di Dataset</div>
                    </div>
                </div>

                <div class="admin-actions">
                    <a href="/pred/admin/all-submissions/" class="action-button">📊 Lihat Semua Pengiriman</a>
                    <a href="/pred/history/" class="action-button success">📋 Tampilan Riwayat Reguler</a>
                    """ + (f'<a href="/pred/admin/users/" class="action-button">👥 Kelola Pengguna</a>' if user_is_admin else '') + f"""
                    """ + (f'<a href="/admin/" class="action-button danger">⚙️ Django Admin</a>' if user_is_admin else '') + f"""
                </div>

                <div class="recent-submissions">
                    <div class="section-title">
                        📈 Pengiriman Terbaru
                    </div>
    """

    if recent_submissions:
        for submission in recent_submissions:
            local_time = timezone.localtime(submission.submitted_at)
            gender_display = get_gender_display(submission.gender)
            prediction_color = "#ef4444" if submission.prediction_result == "Positif" else "#10b981"
            html_content += f"""
                    <div class="submission-item">
                        <div class="submission-header">
                            📋 Submission #{submission.id} - {gender_display}, Usia {submission.age}
                        </div>
                        <div class="submission-details">
                            📅 {local_time.strftime('%d %B %Y, %H:%M')} WIB
                            {f'<br>🧠 <span style="color: {prediction_color}; font-weight: 600;">Prediksi: {submission.prediction_result}</span>' if submission.prediction_result else ''}
                        </div>
                    </div>
            """
    else:
        html_content += """
                    <div class="submission-item" style="text-align: center; color: #6c757d; font-style: italic;">
                        📭 Belum ada pengiriman formulir
                    </div>
        """

    html_content += """
                </div>
            </div>

            <!-- Footer -->
            <div style="background: #f8f9fa; padding: 20px; text-align: center; color: #6c757d; font-size: 0.9em; border-top: 1px solid #e9ecef;">
                <p>⚙️ Dashboard Admin - Sistem Prediksi Depresi</p>
                <p style="margin-top: 5px;">🔒 Akses terbatas untuk administrator dan ahli</p>
            </div>
        </div>
    </body>
    </html>
    """

    return HttpResponse(html_content)




@login_required
@user_passes_test(is_admin)
def expert_reuse_data(request, submission_id):
    """
    Admin-only view to reuse form submission data by adding it to the dataset
    """
    try:
        submission = FormSubmission.objects.get(id=submission_id)

        # Check if already reused
        if submission.is_reused_in_dataset:
            messages.warning(request, f'Pengiriman #{submission_id} sudah ditambahkan ke dataset pada {submission.reused_at.strftime("%Y-%m-%d %H:%M")} oleh {submission.reused_by.username if submission.reused_by else "Tidak Diketahui"}.')
            return redirect('user')

        # Prepare the data row to add to CSV
        new_row = {
            'gender': int(submission.gender),
            'age': submission.age,
            'work_pressure': submission.work_pressure,
            'job_satisfaction': submission.job_satisfaction,
            'sleep_duration': submission.sleep_duration,
            'dietary_habits': submission.dietary_habits,
            'suicidal_thoughts': int(submission.suicidal_thoughts),
            'work_hours': submission.work_hours,
            'financial_stress': submission.financial_stress,
            'family_history_of_mental_illness': int(submission.family_history_of_mental_illness),
            'depression': 1 if submission.prediction_result.lower() in ['positif', 'positive'] else 0
        }

        # Add to dataset
        success = add_to_dataset(new_row)

        if success:
            # Mark as reused
            from django.utils import timezone
            submission.is_reused_in_dataset = True
            submission.reused_at = timezone.now()
            submission.reused_by = request.user
            submission.save()

            messages.success(request, f'Berhasil menambahkan data pengiriman #{submission_id} ke dataset! Ukuran dataset bertambah.')
        else:
            messages.error(request, f'Gagal menambahkan data pengiriman #{submission_id} ke dataset.')

    except FormSubmission.DoesNotExist:
        messages.error(request, f'Pengiriman #{submission_id} tidak ditemukan.')
    except Exception as e:
        messages.error(request, f'Kesalahan memproses permintaan: {str(e)}')

    return redirect('user')


# @require_POST
# def expert_reuse_data(request, submission_id):
#     # Hanya admin/staff yang boleh
#     if not (request.user.is_superuser or request.user.is_staff):
#         messages.error(request, "Anda tidak memiliki akses.")
#         return HttpResponseRedirect(reverse('user'))

#     try:
#         submission = FormSubmission.objects.get(pk=submission_id)
#         # Ambil data form_data
#         form_data = submission.form_data or {}
#         # Siapkan urutan kolom sesuai dataset
#         fieldnames = [
#             'gender', 'age', 'work_hours', 'sleep_duration', 'work_pressure',
#             'job_satisfaction', 'financial_stress', 'dietary_habits',
#             'suicidal_thoughts', 'family_history_of_mental_illness', 'prediction_result'
#         ]
#         # Siapkan data baris
#         row = {field: form_data.get(field, '') for field in fieldnames}
#         row['prediction_result'] = getattr(submission, 'prediction_result', '')

#         # Simpan ke dataset_processed.csv
#         csv_path = '/home/fauziwig/Documents/coding/django-test/django/djangotutorial/dataset_processed.csv'
#         write_header = False
#         try:
#             with open(csv_path, 'r', newline='') as f:
#                 pass
#         except FileNotFoundError:
#             write_header = True

#         import csv
#         with open(csv_path, 'a', newline='') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             if write_header:
#                 writer.writeheader()
#             writer.writerow(row)

#         messages.success(request, "Data berhasil ditambahkan ke dataset.")
#     except Exception as e:
#         messages.error(request, f"Gagal menambah ke dataset: {e}")

#     return HttpResponseRedirect(reverse('user'))


def add_to_dataset(new_row):
    """
    Add a new row to the dataset CSV file.
    Retrain model if 20 new rows have been added.
    Returns: True if successful, False otherwise
    """
    try:
        

        dataset_path = os.path.join(settings.BASE_DIR, 'dataset_processed.csv')
        counter_path = os.path.join(settings.BASE_DIR, 'retrain_counter.txt')
        model_path = os.path.join(settings.BASE_DIR, 'svm_model.joblib')

        # Load existing dataset
        df = pd.read_csv(dataset_path)
        new_row_df = pd.DataFrame([new_row])
        updated_df = pd.concat([df, new_row_df], ignore_index=True)
        updated_df.to_csv(dataset_path, index=False)

        # --- Retrain logic ---
        # Read or initialize counter
        if os.path.exists(counter_path):
            with open(counter_path, 'r') as f:
                counter = int(f.read().strip() or 0)
        else:
            counter = 0

        counter += 1

        if counter >= 2:
            # Retrain model
            retrain_model(dataset_path, model_path)
            counter = 0  # Reset counter

        # Save counter
        with open(counter_path, 'w') as f:
            f.write(str(counter))

        print(f"Dataset updated. New size: {len(updated_df)} rows")
        return True

    except Exception as e:
        print(f"Error updating dataset: {e}")
        return False

def retrain_model(dataset_path, model_path):
    """
    Retrain the ML model using the updated dataset and save as joblib.
    """
    import pandas as pd
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(dataset_path)
    # Assume 'depression' is the target column
    X = df.drop('depression', axis=1)
    y = df['depression']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()

    # Pelajari skala dari data latih dan transformasikan
    X_train_scaled = scaler.fit_transform(X_train)

    # Terapkan skala yang sama ke data uji
    X_test_scaled = scaler.transform(X_test)


    # Retrain model (simple SVM, adjust as needed)
    model = SVC(probability=True, kernel='linear', random_state=42, gamma=1, C=100, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    joblib.dump(model, model_path)
    print("Model retrained and saved.")
