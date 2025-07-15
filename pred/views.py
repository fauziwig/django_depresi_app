from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import MyForm, CustomUserCreationForm
from django.template import loader
from .models import FormSubmission
from django.utils import timezone
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from django.contrib import messages
import joblib
import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings

# Helper function to check if user is admin
def is_admin(user):
    return user.is_authenticated and (user.is_superuser or user.is_staff)

# Helper function to check if user is expert
def is_expert(user):
    return user.is_authenticated and hasattr(user, 'groups') and user.groups.filter(name='Expert').exists()

# Helper function to check if user has admin-level access (admin or expert)
def has_admin_access(user):
    return is_admin(user) or is_expert(user)

# Load the ML model
def load_depression_model():
    """Load the depression prediction model"""
    try:
        model_path = os.path.join(settings.BASE_DIR, 'depression_prediction_model.joblib')
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Prediction function
def predict_depression(form_data):
    """
    Predict depression based on form data
    Returns: dict with prediction result and probability
    """
    try:
        model = load_depression_model()
        if model is None:
            return {
                'prediction': 'Error',
                'probability': 0,
                'message': 'Model could not be loaded'
            }

        # Prepare the input data as a pandas DataFrame with proper column names
        # The model expects specific lowercase column names
        input_data = pd.DataFrame({
            'gender': [int(form_data['gender'])],
            'age': [int(form_data['age'])],
            'work_pressure': [int(form_data['work_pressure'])],
            'job_satisfaction': [int(form_data['job_satisfaction'])],
            'financial_stress': [int(form_data['financial_stress'])],
            'sleep_duration': [int(form_data['sleep_duration'])],
            'dietary_habits': [int(form_data['dietary_habits'])],
            'suicidal_thoughts': [int(form_data['suicidal_thoughts'])],
            'work_hours': [int(form_data['work_hours'])],
            'family_history_of_mental_illness': [int(form_data['family_history_of_mental_illness'])]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Get prediction probability if available
        try:
            prediction_proba = model.predict_proba(input_data)[0]
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
        df = pd.read_csv(dataset_path)

        # Prepare user input vector (excluding depression column)
        user_vector = np.array([[
            int(form_data['gender']),
            int(form_data['age']),
            int(form_data['work_pressure']),
            int(form_data['job_satisfaction']),
            int(form_data['sleep_duration']),
            int(form_data['dietary_habits']),
            int(form_data['suicidal_thoughts']),
            int(form_data['work_hours']),
            int(form_data['financial_stress']),
            int(form_data['family_history_of_mental_illness'])
        ]])

        # Prepare dataset vectors (excluding depression column)
        feature_columns = ['gender', 'age', 'work_pressure', 'job_satisfaction',
                          'sleep_duration', 'dietary_habits', 'suicidal_thoughts',
                          'work_hours', 'financial_stress', 'family_history_of_mental_illness']

        dataset_vectors = df[feature_columns].values

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

            top_matches.append({
                'similarity': float(similarity_score * 100),  # Convert to percentage and ensure float
                'gender': 'Laki-laki' if match_data['gender'] == 1 else 'Perempuan',
                'age': int(match_data['age']),  # Convert to regular int
                'work_pressure': int(match_data['work_pressure']),
                'job_satisfaction': int(match_data['job_satisfaction']),
                'sleep_duration': int(match_data['sleep_duration']),
                'dietary_habits': int(match_data['dietary_habits']),
                'suicidal_thoughts': 'Ya' if match_data['suicidal_thoughts'] == 1 else 'Tidak',
                'work_hours': int(match_data['work_hours']),
                'financial_stress': int(match_data['financial_stress']),
                'family_history': 'Ya' if match_data['family_history_of_mental_illness'] == 1 else 'Tidak',
                'depression': 'Positif' if match_data['depression'] == 1 else 'Negatif'
            })

        return {
            'success': True,
            'best_similarity': float(best_similarity * 100),  # Convert to percentage and ensure float
            'best_match_idx': int(best_match_idx),  # Store the index
            'best_match': {
                'gender': 'Laki-laki' if best_match_row['gender'] == 1 else 'Perempuan',
                'age': int(best_match_row['age']),  # Convert to regular int
                'work_pressure': int(best_match_row['work_pressure']),
                'job_satisfaction': int(best_match_row['job_satisfaction']),
                'sleep_duration': int(best_match_row['sleep_duration']),
                'dietary_habits': int(best_match_row['dietary_habits']),
                'suicidal_thoughts': 'Ya' if best_match_row['suicidal_thoughts'] == 1 else 'Tidak',
                'work_hours': int(best_match_row['work_hours']),
                'financial_stress': int(best_match_row['financial_stress']),
                'family_history': 'Ya' if best_match_row['family_history_of_mental_illness'] == 1 else 'Tidak',
                'depression': 'Positif' if best_match_row['depression'] == 1 else 'Negatif'
            },
            'top_matches': top_matches,
            'total_cases': int(len(df))  # Convert to regular int
        }

    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Tidak dapat menghitung kemiripan dengan dataset'
        }

def my_view(request):
    template = loader.get_template("my_template.html")

    if request.method == 'POST':
        form = MyForm(request.POST)
        if form.is_valid():
            # Save form data to database
            submission = FormSubmission.objects.create(
                gender=form.cleaned_data['gender'],
                age=form.cleaned_data['age'],
                work_pressure=form.cleaned_data['work_pressure'],
                job_satisfaction=form.cleaned_data['job_satisfaction'],
                financial_stress=form.cleaned_data['financial_stress'],
                sleep_duration=form.cleaned_data['sleep_duration'],
                dietary_habits=form.cleaned_data['dietary_habits'],
                suicidal_thoughts=form.cleaned_data['suicidal_thoughts'],
                work_hours=form.cleaned_data['work_hours'],
                family_history_of_mental_illness=form.cleaned_data['family_history_of_mental_illness']
            )

            # Perform ML prediction
            prediction_result = predict_depression(form.cleaned_data)

            # Perform cosine similarity analysis
            similarity_result = find_similar_cases(form.cleaned_data)

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


def success_view(request):
    return HttpResponse("<h1>Form submitted successfully!</h1><p><a href='/pred/'>Go back to form</a></p>")

def results_view(request):
    # Get form data, prediction result, and similarity result from session
    form_data = request.session.get('form_data', {})
    submission_id = request.session.get('submission_id')
    prediction_result = request.session.get('prediction_result', {})
    similarity_result = request.session.get('similarity_result', {})

    if not form_data:
        return HttpResponse("<h1>Data tidak ditemukan</h1><p><a href='/pred/'>Kembali ke formulir</a></p>")

    # Get the submission time if we have the submission ID
    submission_time = None
    if submission_id:
        try:
            submission = FormSubmission.objects.get(id=submission_id)
            submission_time = timezone.localtime(submission.submitted_at)
        except FormSubmission.DoesNotExist:
            pass

    # Create styled HTML to display the form data
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hasil Formulir</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .results-container {
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .data-item {
                margin-bottom: 15px;
                padding: 10px;
                background-color: #f8f9fa;
                border-left: 4px solid #007bff;
            }
            .label {
                font-weight: bold;
                color: #333;
            }
            .value {
                color: #666;
                margin-top: 5px;
            }
            .back-link {
                display: inline-block;
                margin-top: 20px;
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 4px;
            }
            .back-link:hover {
                background-color: #0056b3;
            }
            .prediction-container {
                margin: 30px 0;
                padding: 25px;
                border-radius: 8px;
                text-align: center;
                font-size: 1.1em;
            }
            .prediction-positive {
                background-color: #f8d7da;
                border: 2px solid #dc3545;
                color: #721c24;
            }
            .prediction-negative {
                background-color: #d4edda;
                border: 2px solid #28a745;
                color: #155724;
            }
            .prediction-error {
                background-color: #fff3cd;
                border: 2px solid #ffc107;
                color: #856404;
            }
            .prediction-title {
                font-size: 1.3em;
                font-weight: bold;
                margin-bottom: 15px;
            }
            .prediction-message {
                margin-bottom: 10px;
                line-height: 1.5;
            }
            .prediction-probability {
                font-size: 0.9em;
                opacity: 0.8;
            }
            .similarity-container {
                margin: 30px 0;
                padding: 25px;
                background-color: #f8f9fa;
                border: 2px solid #007bff;
                border-radius: 8px;
            }
            .similarity-title {
                font-size: 1.3em;
                font-weight: bold;
                margin-bottom: 15px;
                color: #007bff;
            }
            .best-match {
                background-color: #e3f2fd;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 20px;
            }
            .similarity-score {
                font-size: 1.1em;
                font-weight: bold;
                color: #1976d2;
                margin-bottom: 10px;
            }
            .match-details {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin-top: 10px;
            }
            .match-field {
                background-color: white;
                padding: 8px 12px;
                border-radius: 4px;
                border: 1px solid #e0e0e0;
            }
            .field-name {
                font-weight: bold;
                font-size: 0.9em;
                color: #666;
            }
            .field-value {
                color: #333;
                margin-top: 2px;
            }
            .top-matches {
                margin-top: 20px;
            }
            .match-item {
                background-color: #f5f5f5;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 4px;
                border-left: 4px solid #007bff;
            }
        </style>
    </head>
    <body>
        <div class="results-container">
            <h1>Hasil Pengiriman Formulir</h1>

            <!-- Prediction Results Section -->
            """ + (f"""
            <div class="prediction-container prediction-{prediction_result.get('prediction', 'error').lower()}">
                <div class="prediction-title">
                    🧠 Hasil Prediksi Depresi
                </div>
                <div class="prediction-message">
                    <strong>Prediksi:</strong> {prediction_result.get('prediction', 'Tidak Diketahui')}
                </div>
                <div class="prediction-message">
                    {prediction_result.get('message', 'No prediction available')}
                </div>
                {f'<div class="prediction-probability">Kepercayaan: {prediction_result.get("probability", 0):.1f}%</div>' if prediction_result.get('probability', 0) > 0 else ''}
            </div>
            """ if prediction_result else "") + """

            <!-- Similarity Analysis Section -->
            """ + (f"""
            <div class="similarity-container">
                <div class="similarity-title">
                    📊 Analisis Kasus Serupa
                </div>
                <div class="best-match">
                    <div class="similarity-score">
                        Kecocokan Terbaik: {similarity_result.get('best_similarity', 0):.1f}% Kemiripan
                    </div>
                    <div class="match-details">
                        <div class="match-field">
                            <div class="field-name">Jenis Kelamin</div>
                            <div class="field-value">{similarity_result.get('best_match', {}).get('gender', 'N/A')}</div>
                        </div>
                        <div class="match-field">
                            <div class="field-name">Usia</div>
                            <div class="field-value">{similarity_result.get('best_match', {}).get('age', 'N/A')}</div>
                        </div>
                        <div class="match-field">
                            <div class="field-name">Tekanan Kerja</div>
                            <div class="field-value">{similarity_result.get('best_match', {}).get('work_pressure', 'N/A')}</div>
                        </div>
                        <div class="match-field">
                            <div class="field-name">Kepuasan Kerja</div>
                            <div class="field-value">{similarity_result.get('best_match', {}).get('job_satisfaction', 'N/A')}</div>
                        </div>
                        <div class="match-field">
                            <div class="field-name">Durasi Tidur</div>
                            <div class="field-value">{similarity_result.get('best_match', {}).get('sleep_duration', 'N/A')}</div>
                        </div>
                        <div class="match-field">
                            <div class="field-name">Jam Kerja</div>
                            <div class="field-value">{similarity_result.get('best_match', {}).get('work_hours', 'N/A')}</div>
                        </div>
                        <div class="match-field" style="background-color: {'#ffebee' if similarity_result.get('best_match', {}).get('depression') == 'Positive' else '#e8f5e8'};">
                            <div class="field-name">Status Depresi</div>
                            <div class="field-value" style="color: {'#d32f2f' if similarity_result.get('best_match', {}).get('depression') == 'Positive' else '#388e3c'}; font-weight: bold;">
                                {similarity_result.get('best_match', {}).get('depression', 'N/A')}
                            </div>
                        </div>
                    </div>
                </div>
                <div style="text-align: center; color: #666; font-size: 0.9em;">
                    Dianalisis terhadap {similarity_result.get('total_cases', 0)} kasus dari dataset
                </div>
            </div>
            """ if similarity_result.get('success') else f"""
            <div class="similarity-container">
                <div class="similarity-title">📊 Analisis Kasus Serupa</div>
                <div style="color: #d32f2f; text-align: center; padding: 20px;">
                    {similarity_result.get('message', 'Analisis kemiripan tidak tersedia')}
                </div>
            </div>
            """ if similarity_result else "") + """
    """

    # Field labels mapping
    field_labels = {
        'gender': 'Jenis Kelamin',
        'age': 'Usia',
        'work_pressure': 'Tekanan Kerja',
        'job_satisfaction': 'Kepuasan Kerja',
        'financial_stress': 'Stres Keuangan',
        'sleep_duration': 'Durasi Tidur',
        'dietary_habits': 'Kebiasaan Makan',
        'suicidal_thoughts': 'Pikiran Bunuh Diri',
        'work_hours': 'Jam Kerja',
        'family_history_of_mental_illness': 'Riwayat Keluarga Penyakit Mental'
    }

    for field_name, field_value in form_data.items():
        label = field_labels.get(field_name, field_name.title())
        html_content += f"""
            <div class="data-item">
                <div class="label">{label}:</div>
                <div class="value">{field_value}</div>
            </div>
        """

    # Add submission time if available
    if submission_time:
        html_content += f"""
            <div style="margin-top: 20px; padding: 15px; background-color: #e9ecef; border-radius: 4px; text-align: center;">
                <strong>Submitted at:</strong> {submission_time.strftime('%Y-%m-%d %H:%M:%S')} WIB
            </div>
        """

    # Add navigation links based on user authentication
    if request.user.is_authenticated:
        html_content += f"""
            <div style="margin-top: 20px;">
                <a href="/pred/" class="back-link">Kirim Formulir Lain</a>
                <a href="/pred/history/" class="back-link" style="margin-left: 10px; background-color: #28a745;">Lihat Riwayat</a>
                {f'<a href="/pred/admin/dashboard/" class="back-link" style="margin-left: 10px; background-color: #dc3545;">Dashboard Admin</a>' if is_admin(request.user) else f'<a href="/pred/admin/dashboard/" class="back-link" style="margin-left: 10px; background-color: #28a745;">Dashboard Ahli</a>' if is_expert(request.user) else ''}
            </div>
        """
    else:
        html_content += """
            <div style="margin-top: 20px;">
                <a href="/pred/" class="back-link">Kirim Formulir Lain</a>
                <a href="/pred/login/" class="back-link" style="margin-left: 10px; background-color: #28a745;">Masuk untuk Lihat Riwayat</a>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    # Clear the session data after displaying
    if 'form_data' in request.session:
        del request.session['form_data']

    return HttpResponse(html_content)


@login_required
def history_view(request):
    # Get form submissions based on user role
    if is_admin(request.user):
        # Admin can see all submissions
        submissions = FormSubmission.objects.all()
        page_title = "Semua Pengiriman Formulir (Tampilan Admin)"
        user_info = f"Admin: {request.user.get_full_name() or request.user.username}"
    elif is_expert(request.user):
        # Expert can see all submissions
        submissions = FormSubmission.objects.all()
        page_title = "Semua Pengiriman Formulir (Tampilan Ahli)"
        user_info = f"Ahli: {request.user.get_full_name() or request.user.username}"
    else:
        # Regular users see only their own submissions (if we implement user-specific submissions)
        # For now, regular users see all submissions but with limited info
        submissions = FormSubmission.objects.all()
        page_title = "Riwayat Pengiriman Formulir"
        user_info = f"Selamat datang, {request.user.get_full_name() or request.user.username}!"

    # Create styled HTML to display the history
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Form Submission History</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .history-container {
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .submission-item {
                margin-bottom: 20px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 6px;
                border-left: 4px solid #007bff;
            }
            .submission-header {
                font-weight: bold;
                color: #007bff;
                margin-bottom: 10px;
                font-size: 18px;
            }
            .submission-data {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin-bottom: 10px;
            }
            .data-field {
                background-color: white;
                padding: 8px 12px;
                border-radius: 4px;
                border: 1px solid #e9ecef;
            }
            .reuse-btn {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.9em;
                font-weight: bold;
                text-decoration: none;
                display: inline-block;
                transition: background-color 0.3s;
            }
            .reuse-btn:hover {
                background-color: #218838;
                color: white;
                text-decoration: none;
            }
            .field-label {
                font-weight: bold;
                color: #333;
                font-size: 12px;
                text-transform: uppercase;
            }
            .field-value {
                color: #666;
                margin-top: 2px;
            }
            .submission-time {
                color: #6c757d;
                font-size: 14px;
                font-style: italic;
            }
            .back-link {
                display: inline-block;
                margin-bottom: 20px;
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 4px;
            }
            .back-link:hover {
                background-color: #0056b3;
            }
            .no-data {
                text-align: center;
                color: #6c757d;
                font-style: italic;
                padding: 40px;
            }
        </style>
    </head>
    <body>
        <div class="history-container">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <div>
                    <a href="/pred/" class="back-link">← Kembali ke Formulir</a>
                    """ + (f'<a href="/pred/admin/dashboard/" style="margin-left: 15px; color: #dc3545; text-decoration: none;">⚙️ Admin Dashboard</a>' if is_admin(request.user) else f'<a href="/pred/admin/dashboard/" style="margin-left: 15px; color: #28a745; text-decoration: none;">🔬 Expert Dashboard</a>' if is_expert(request.user) else '') + f"""
                </div>
                <div style="text-align: right;">
                    <span style="color: #666;">{user_info}</span>
                    <a href="/pred/logout/" style="margin-left: 15px; color: #dc3545; text-decoration: none;">Keluar</a>
                </div>
            </div>
            <h1>{page_title}</h1>
    """

    if submissions.exists():
        for submission in submissions:
            # Convert UTC time to local time using Django's timezone.localtime
            local_time = timezone.localtime(submission.submitted_at)

            html_content += f"""
                <div class="submission-item">
                    <div class="submission-header">Submission #{submission.id}</div>
                    <div class="submission-data">
                        <div class="data-field">
                            <div class="field-label">Jenis Kelamin</div>
                            <div class="field-value">{'Laki-laki' if submission.gender == '1' else 'Perempuan'}</div>
                        </div>
                        <div class="data-field">
                            <div class="field-label">Usia</div>
                            <div class="field-value">{submission.age}</div>
                        </div>
                        <div class="data-field">
                            <div class="field-label">Tekanan Kerja</div>
                            <div class="field-value">{submission.work_pressure}</div>
                        </div>
                        <div class="data-field">
                            <div class="field-label">Kepuasan Kerja</div>
                            <div class="field-value">{submission.job_satisfaction}</div>
                        </div>
                        <div class="data-field">
                            <div class="field-label">Stres Keuangan</div>
                            <div class="field-value">{submission.financial_stress}</div>
                        </div>
                        <div class="data-field">
                            <div class="field-label">Durasi Tidur</div>
                            <div class="field-value">{submission.sleep_duration}</div>
                        </div>
                        <div class="data-field">
                            <div class="field-label">Jam Kerja</div>
                            <div class="field-value">{submission.work_hours}</div>
                        </div>
                        <div class="data-field" style="background-color: {'#f8d7da' if submission.prediction_result == 'Positive' else '#d4edda' if submission.prediction_result == 'Negative' else '#fff3cd'};">
                            <div class="field-label">🧠 Hasil Prediksi</div>
                            <div class="field-value">
                                <strong>{submission.prediction_result or 'Tidak Tersedia'}</strong>
                                {f' ({submission.prediction_probability:.1f}% kepercayaan)' if submission.prediction_probability else ''}
                            </div>
                        </div>
                        <div class="data-field" style="background-color: #e3f2fd;">
                            <div class="field-label">📊 Skor Kemiripan</div>
                            <div class="field-value">
                                <strong>{f'{submission.similarity_score:.1f}%' if submission.similarity_score else 'Tidak Tersedia'}</strong>
                                {f' (Case #{submission.similar_case_id})' if submission.similar_case_id else ''}
                            </div>
                        </div>
                        """ + (f"""
                        <div class="data-field" style="background-color: {'#d4edda' if submission.is_reused_in_dataset else '#fff3cd'}; text-align: center;">
                            {f'''
                            <div style="color: #155724; font-weight: bold;">
                                ✅ Ditambahkan ke Dataset
                                <br><small>By {submission.reused_by.username if submission.reused_by else "Unknown"} on {submission.reused_at.strftime("%Y-%m-%d %H:%M") if submission.reused_at else "Unknown date"}</small>
                            </div>
                            ''' if submission.is_reused_in_dataset else f'''
                            <a href="/pred/expert/reuse-data/{submission.id}/" class="reuse-btn" onclick="return confirm('Apakah Anda yakin ingin menambahkan data pengiriman ini ke dataset? Tindakan ini tidak dapat dibatalkan.')">
                                🔄 Reuse Data
                            </a>
                            '''}
                        </div>
                        """ if is_expert(request.user) else "") + f"""

                    </div>
                    <div class="submission-time">Submitted: {local_time.strftime('%Y-%m-%d %H:%M:%S')} WIB</div>
                </div>
            """
    else:
        html_content += '<div class="no-data">Tidak ada pengiriman formulir yang ditemukan.</div>'

    html_content += """
        </div>
    </body>
    </html>
    """

    return HttpResponse(html_content)


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            # Redirect to history page after successful login
            return redirect('history_url')
        else:
            messages.error(request, 'Username atau password salah.')

    return render(request, 'login.html')


def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Akun berhasil dibuat untuk {username}!')
            # Auto login after registration
            login(request, user)
            return redirect('history_url')
    else:
        form = CustomUserCreationForm()

    return render(request, 'register.html', {'form': form})


def logout_view(request):
    logout(request)
    messages.success(request, 'Anda telah berhasil logout.')
    return redirect('my_view')


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
    <html>
    <head>
        <title>{dashboard_title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .dashboard-container {{
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background-color: #007bff;
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .stat-label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            .admin-actions {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }}
            .action-button {{
                display: block;
                padding: 15px 20px;
                background-color: #28a745;
                color: white;
                text-decoration: none;
                border-radius: 6px;
                text-align: center;
                font-weight: bold;
                transition: background-color 0.3s;
            }}
            .action-button:hover {{
                background-color: #218838;
            }}
            .action-button.danger {{
                background-color: #dc3545;
            }}
            .action-button.danger:hover {{
                background-color: #c82333;
            }}
            .recent-submissions {{
                margin-top: 30px;
            }}
            .submission-item {{
                background-color: #f8f9fa;
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 6px;
                border-left: 4px solid #007bff;
            }}
            .header-nav {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 30px;
            }}
            .back-link {{
                color: #007bff;
                text-decoration: none;
                padding: 10px 15px;
                border: 1px solid #007bff;
                border-radius: 4px;
            }}
            .user-info {{
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="header-nav">
                <a href="/pred/" class="back-link">← Back to Form</a>
                <div class="user-info">
                    Admin: {request.user.get_full_name() or request.user.username}
                    <a href="/pred/logout/" style="margin-left: 15px; color: #dc3545;">Logout</a>
                </div>
            </div>

            <h1>{'🔧 Dashboard Admin' if user_is_admin else '🔬 Dashboard Ahli'}</h1>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{total_submissions}</div>
                    <div class="stat-label">Total Pengiriman</div>
                </div>
                <div class="stat-card" style="background-color: #28a745;">
                    <div class="stat-number">{total_users}</div>
                    <div class="stat-label">Total Pengguna</div>
                </div>
                <div class="stat-card" style="background-color: #17a2b8;">
                    <div class="stat-number">{reused_submissions}</div>
                    <div class="stat-label">Reused di Dataset</div>
                </div>
            </div>

            <div class="admin-actions">
                <a href="/pred/admin/all-submissions/" class="action-button">Lihat Semua Pengiriman</a>
                <a href="/pred/history/" class="action-button">Tampilan Riwayat Reguler</a>
                """ + (f'<a href="/pred/admin/users/" class="action-button">Kelola Pengguna</a>' if user_is_admin else '') + f"""
                """ + (f'<a href="/admin/" class="action-button" style="background-color: #6f42c1;">Django Admin</a>' if user_is_admin else '') + f"""
            </div>

            <div class="recent-submissions">
                <h3>Pengiriman Terbaru</h3>
    """

    if recent_submissions:
        for submission in recent_submissions:
            local_time = timezone.localtime(submission.submitted_at)
            gender_display = 'Male' if submission.gender == '1' else 'Female'
            html_content += f"""
                <div class="submission-item">
                    <strong>#{submission.id}</strong> - {gender_display}, Usia {submission.age}
                    <br><small>{local_time.strftime('%Y-%m-%d %H:%M:%S')} WIB</small>
                    {f'<br><span style="color: {"#dc3545" if submission.prediction_result == "Positif" else "#28a745"};">Prediksi: {submission.prediction_result}</span>' if submission.prediction_result else ''}
                </div>
            """
    else:
        html_content += '<p>Belum ada pengiriman.</p>'

    html_content += """
            </div>
        </div>
    </body>
    </html>
    """

    return HttpResponse(html_content)


@user_passes_test(has_admin_access)
def admin_all_submissions(request):
    submissions = FormSubmission.objects.all().order_by('-submitted_at')

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>All Submissions - Admin</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .admin-container {{
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header-nav {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 30px;
            }}
            .back-link {{
                color: #007bff;
                text-decoration: none;
                padding: 10px 15px;
                border: 1px solid #007bff;
                border-radius: 4px;
            }}
            .submission-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            .submission-table th,
            .submission-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .submission-table th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            .submission-table tr:hover {{
                background-color: #f5f5f5;
            }}
            .delete-btn {{
                background-color: #dc3545;
                color: white;
                padding: 5px 10px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
            }}
            .delete-btn:hover {{
                background-color: #c82333;
            }}
        </style>
    </head>
    <body>
        <div class="admin-container">
            <div class="header-nav">
                <a href="/pred/admin/dashboard/" class="back-link">← Back to Dashboard</a>
                <div style="color: #666;">
                    Admin: {request.user.get_full_name() or request.user.username}
                </div>
            </div>

            <h1>All Form Submissions</h1>
            <p>Total: {submissions.count()} submissions</p>

            <table class="submission-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Gender</th>
                        <th>Age</th>
                        <th>Work Pressure</th>
                        <th>Job Satisfaction</th>
                        <th>Prediction</th>
                        <th>Submitted</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
    """

    for submission in submissions:
        local_time = timezone.localtime(submission.submitted_at)
        html_content += f"""
                    <tr>
                        <td>#{submission.id}</td>
                        <td>{'Male' if submission.gender == '1' else 'Female'}</td>
                        <td>{submission.age}</td>
                        <td>{submission.work_pressure}</td>
                        <td>{submission.job_satisfaction}</td>
                        <td style="color: {'#dc3545' if submission.prediction_result == 'Positive' else '#28a745' if submission.prediction_result == 'Negative' else '#6c757d'};">
                            {submission.prediction_result or 'N/A'}
                            {f' ({submission.prediction_probability:.0f}%)' if submission.prediction_probability else ''}
                        </td>
                        <td>{local_time.strftime('%Y-%m-%d %H:%M')}</td>
                        <td>
                            <form method="post" action="/pred/admin/delete-submission/{submission.id}/" style="display: inline;">
                                <input type="hidden" name="csrfmiddlewaretoken" value="{request.META.get('CSRF_COOKIE', '')}">
                                <button type="submit" class="delete-btn" onclick="return confirm('Are you sure?')">Delete</button>
                            </form>
                        </td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

    return HttpResponse(html_content)


@user_passes_test(is_admin)
def admin_users(request):
    users = User.objects.all().order_by('-date_joined')

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>User Management - Admin</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .admin-container {{
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header-nav {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 30px;
            }}
            .back-link {{
                color: #007bff;
                text-decoration: none;
                padding: 10px 15px;
                border: 1px solid #007bff;
                border-radius: 4px;
            }}
            .user-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            .user-table th,
            .user-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .user-table th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            .user-table tr:hover {{
                background-color: #f5f5f5;
            }}
            .admin-badge {{
                background-color: #dc3545;
                color: white;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 12px;
            }}
            .staff-badge {{
                background-color: #ffc107;
                color: black;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="admin-container">
            <div class="header-nav">
                <a href="/pred/admin/dashboard/" class="back-link">← Back to Dashboard</a>
                <div style="color: #666;">
                    Admin: {request.user.get_full_name() or request.user.username}
                </div>
            </div>

            <h1>User Management</h1>
            <p>Total: {users.count()} users</p>

            <table class="user-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Username</th>
                        <th>Full Name</th>
                        <th>Email</th>
                        <th>Role</th>
                        <th>Joined</th>
                        <th>Last Login</th>
                    </tr>
                </thead>
                <tbody>
    """

    for user in users:
        role_badges = []
        if user.is_superuser:
            role_badges.append('<span class="admin-badge">Superuser</span>')
        if user.is_staff:
            role_badges.append('<span class="staff-badge">Staff</span>')

        role_display = ' '.join(role_badges) if role_badges else 'User'

        html_content += f"""
                    <tr>
                        <td>#{user.id}</td>
                        <td>{user.username}</td>
                        <td>{user.get_full_name() or '-'}</td>
                        <td>{user.email or '-'}</td>
                        <td>{role_display}</td>
                        <td>{user.date_joined.strftime('%Y-%m-%d')}</td>
                        <td>{user.last_login.strftime('%Y-%m-%d %H:%M') if user.last_login else 'Never'}</td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

    return HttpResponse(html_content)


@user_passes_test(has_admin_access)
def admin_delete_submission(request, submission_id):
    if request.method == 'POST':
        try:
            submission = FormSubmission.objects.get(id=submission_id)
            submission.delete()
            messages.success(request, f'Submission #{submission_id} has been deleted.')
        except FormSubmission.DoesNotExist:
            messages.error(request, 'Submission not found.')

    return redirect('admin_all_submissions')


@user_passes_test(is_expert)
def expert_reuse_data(request, submission_id):
    """
    Expert-only view to reuse form submission data by adding it to the dataset
    """
    try:
        submission = FormSubmission.objects.get(id=submission_id)

        # Check if already reused
        if submission.is_reused_in_dataset:
            messages.warning(request, f'Pengiriman #{submission_id} sudah ditambahkan ke dataset pada {submission.reused_at.strftime("%Y-%m-%d %H:%M")} oleh {submission.reused_by.username if submission.reused_by else "Tidak Diketahui"}.')
            return redirect('history_url')

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
            'depression': 1 if submission.prediction_result == 'Positive' else 0  # Use ML prediction as ground truth
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

    return redirect('history_url')


def add_to_dataset(new_row):
    """
    Add a new row to the dataset CSV file
    Returns: True if successful, False otherwise
    """
    try:
        import pandas as pd
        import os
        from django.conf import settings

        # Load existing dataset
        dataset_path = os.path.join(settings.BASE_DIR, 'dataset_processed.csv')
        df = pd.read_csv(dataset_path)

        # Create new row as DataFrame
        new_row_df = pd.DataFrame([new_row])

        # Append to existing dataset
        updated_df = pd.concat([df, new_row_df], ignore_index=True)

        # Save back to CSV
        updated_df.to_csv(dataset_path, index=False)

        print(f"Dataset updated. New size: {len(updated_df)} rows")
        return True

    except Exception as e:
        print(f"Error updating dataset: {e}")
        return False

