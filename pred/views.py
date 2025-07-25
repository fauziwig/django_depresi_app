from django.http import HttpResponse
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
        'financial_stress': int(form_data['financial_stress']),
        'sleep_duration': int(form_data['sleep_duration']),
        'dietary_habits': int(form_data['dietary_habits']),
        'suicidal_thoughts': int(form_data['suicidal_thoughts']),
        'work_hours': int(form_data['work_hours']),
        'family_history_of_mental_illness': int(form_data['family_history_of_mental_illness']),
        'gender_Female': gender_female,
        'gender_Male': gender_male
    }

    return transformed_data

# Display helper functions to convert database values to human-readable text
def get_gender_display(value):
    """Convert gender database value to display text"""
    gender_choices = {
        '1': 'Laki laki',
        '0': 'Perempuan'
    }
    return gender_choices.get(str(value), 'Tidak Diketahui')

def get_pressure_display(value):
    """Convert pressure-related database value to display text"""
    pressure_choices = {
        '1': 'Sangat Rendah',
        '2': 'Rendah',
        '3': 'Sedang',
        '4': 'Tinggi',
        '5': 'Sangat Tinggi'
    }
    return pressure_choices.get(str(value), 'Tidak Diketahui')

def get_sleep_duration_display(value):
    """Convert sleep duration database value to display text"""
    sleep_choices = {
        '0': '5-6 Jam',
        '1': '7-8 Jam',
        '2': 'Kurang dari 5 Jam',
        '3': 'Lebih dari 8 Jam'
    }
    return sleep_choices.get(str(value), 'Tidak Diketahui')

def get_dietary_habits_display(value):
    """Convert dietary habits database value to display text"""
    dietary_choices = {
        '0': 'Sehat',
        '1': 'Sedang',
        '2': 'Tidak Sehat'
    }
    return dietary_choices.get(str(value), 'Tidak Diketahui')

def get_yes_no_display(value):
    """Convert yes/no database value to display text"""
    yes_no_choices = {
        '1': 'Ya',
        '0': 'Tidak'
    }
    return yes_no_choices.get(str(value), 'Tidak Diketahui')

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

        # Transform form data to the format expected by the ML model
        transformed_data = transform_form_data_for_model(form_data)

        # Create DataFrame with the transformed data
        input_data = pd.DataFrame([transformed_data])

        # Log the transformed data for debugging
        print("\n🔄 Data Transformation for ML Model:")
        print(f"  Original gender: {form_data['gender']} ({'Male' if int(form_data['gender']) == 1 else 'Female'})")
        print(f"  Transformed to: gender_Male={transformed_data['gender_Male']}, gender_Female={transformed_data['gender_Female']}")
        print("\n📊 Final DataFrame for Model:")
        for col, val in transformed_data.items():
            print(f"  {col}: {val}")
        print()

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
                'suicidal_thoughts': get_yes_no_display(match_data['suicidal_thoughts']),
                'work_hours': int(match_data['work_hours']),
                'financial_stress': get_pressure_display(match_data['financial_stress']),
                'family_history': get_yes_no_display(match_data['family_history_of_mental_illness']),
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
                'suicidal_thoughts': get_yes_no_display(best_match_row['suicidal_thoughts']),
                'work_hours': int(best_match_row['work_hours']),
                'financial_stress': get_pressure_display(best_match_row['financial_stress']),
                'family_history': get_yes_no_display(best_match_row['family_history_of_mental_illness']),
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

def my_view(request):
    template = loader.get_template("my_template.html")

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
            return render(request, 'my_template.html', {'form': form})

        print("=" * 60)

        if form.is_valid():
            # Save form data to database with user information
            submission = FormSubmission.objects.create(
                user=request.user if request.user.is_authenticated else None,
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


    print(form_data)
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
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hasil Analisis Kesehatan Mental</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }

            .main-container {
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }

            .header {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }

            .header h1 {
                font-size: 2.2em;
                font-weight: 700;
                margin-bottom: 10px;
            }

            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }

            .results-content {
                padding: 40px;
            }

            .section {
                margin-bottom: 40px;
                background: #f8f9fa;
                border-radius: 15px;
                padding: 25px;
                border-left: 5px solid #4facfe;
            }

            .section-title {
                font-size: 1.4em;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .prediction-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 15px;
                padding: 30px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            }

            .prediction-positive {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
            }

            .prediction-negative {
                background: linear-gradient(135deg, #00d2d3 0%, #54a0ff 100%);
                box-shadow: 0 10px 30px rgba(0, 210, 211, 0.3);
            }

            .prediction-title {
                font-size: 1.5em;
                font-weight: 700;
                margin-bottom: 15px;
            }

            .prediction-message {
                font-size: 1.1em;
                margin-bottom: 10px;
                line-height: 1.5;
            }

            .prediction-probability {
                font-size: 1.3em;
                font-weight: 600;
                margin-top: 15px;
                padding: 10px 20px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 25px;
                display: inline-block;
            }

            .similarity-container {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 30px;
                border-left: 5px solid #17a2b8;
            }

            .similarity-title {
                font-size: 1.4em;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .best-match {
                background: white;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }

            .similarity-score {
                font-size: 1.2em;
                font-weight: 600;
                color: #17a2b8;
                margin-bottom: 15px;
                text-align: center;
            }

            .match-details {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }

            .match-field {
                background: #f8f9fa;
                padding: 12px;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }

            .field-name {
                font-weight: 600;
                font-size: 0.9em;
                color: #6c757d;
                margin-bottom: 5px;
            }

            .field-value {
                color: #2c3e50;
                font-weight: 500;
            }

            .data-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }

            .data-item {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                border-left: 4px solid #4facfe;
            }

            .data-label {
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 8px;
                font-size: 0.95em;
            }

            .data-value {
                color: #495057;
                font-size: 1.05em;
            }

            .action-buttons {
                text-align: center;
                margin-top: 40px;
                padding-top: 30px;
                border-top: 1px solid #e9ecef;
            }

            .btn {
                display: inline-block;
                padding: 12px 30px;
                margin: 0 10px;
                border-radius: 25px;
                text-decoration: none;
                font-weight: 600;
                transition: all 0.3s ease;
                border: none;
                cursor: pointer;
            }

            .btn-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }

            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
            }

            .btn-secondary {
                background: #6c757d;
                color: white;
            }

            .btn-secondary:hover {
                background: #5a6268;
                transform: translateY(-2px);
            }

            @media (max-width: 768px) {
                body {
                    padding: 10px;
                }

                .main-container {
                    border-radius: 15px;
                }

                .header {
                    padding: 20px;
                }

                .header h1 {
                    font-size: 1.8em;
                }

                .results-content {
                    padding: 20px;
                }

                .match-details {
                    grid-template-columns: 1fr;
                }

                .data-grid {
                    grid-template-columns: 1fr;
                }

                .btn {
                    display: block;
                    margin: 10px 0;
                }
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
        <div class="main-container">
            <!-- Header Section -->
            <div class="header">
                <h1>🎯 Hasil Analisis Kesehatan Mental</h1>
                <p>Laporan Komprehensif Berdasarkan AI dan Analisis Data</p>
            </div>

            <div class="results-content">
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
                        {prediction_result.get('message', 'Tidak ada prediksi tersedia')}
                    </div>
                    {f'<div class="prediction-probability">Tingkat Kepercayaan: {prediction_result.get("probability", 0):.1f}%</div>' if prediction_result.get('probability', 0) > 0 else ''}
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
                        <div class="match-field" style="background-color: {'#ffebee' if similarity_result.get('best_match', {}).get('depression') == 'Positif' else '#e8f5e8'};">
                            <div class="field-name">Status Depresi</div>
                            <div class="field-value" style="color: {'#d32f2f' if similarity_result.get('best_match', {}).get('depression') == 'Positif' else '#388e3c'}; font-weight: bold;">
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

                <!-- Form Data Summary Section -->
                <div class="section">
                    <div class="section-title">
                        📋 Ringkasan Data Anda
                    </div>
                    <div class="data-grid">
    """

    # Field labels mapping with icons
    field_labels = {
        'gender': ('👤', 'Jenis Kelamin'),
        'age': ('🎂', 'Usia'),
        'work_pressure': ('💼', 'Tekanan Kerja'),
        'job_satisfaction': ('😊', 'Kepuasan Kerja'),
        'financial_stress': ('💰', 'Stres Keuangan'),
        'sleep_duration': ('😴', 'Durasi Tidur'),
        'dietary_habits': ('🍽️', 'Kebiasaan Makan'),
        'suicidal_thoughts': ('🧠', 'Pikiran Bunuh Diri'),
        'work_hours': ('⏰', 'Jam Kerja'),
        'family_history_of_mental_illness': ('👨‍👩‍👧‍👦', 'Riwayat Keluarga')
    }

    for field_name, field_value in form_data.items():
        icon, label = field_labels.get(field_name, ('📝', field_name.title()))

        # Convert database values to display values
        if field_name == 'gender':
            display_value = get_gender_display(field_value)
        elif field_name in ['work_pressure', 'job_satisfaction', 'financial_stress']:
            display_value = get_pressure_display(field_value)
        elif field_name == 'sleep_duration':
            display_value = get_sleep_duration_display(field_value)
        elif field_name == 'dietary_habits':
            display_value = get_dietary_habits_display(field_value)
        elif field_name in ['suicidal_thoughts', 'family_history_of_mental_illness']:
            display_value = get_yes_no_display(field_value)
        else:
            display_value = field_value

        html_content += f"""
                        <div class="data-item">
                            <div class="data-label">{icon} {label}</div>
                            <div class="data-value">{display_value}</div>
                        </div>
        """

    html_content += """
                    </div>
                </div>
    """

    # Add submission time if available
    if submission_time:
        html_content += f"""
                <div class="section">
                    <div class="section-title">⏰ Waktu Pengiriman</div>
                    <div style="text-align: center; font-size: 1.1em; color: #495057;">
                        {submission_time.strftime('%d %B %Y, %H:%M:%S')} WIB
                    </div>
                </div>
        """

    # Add navigation links based on user authentication
    html_content += """
                <div class="action-buttons">
    """

    if request.user.is_authenticated:
        html_content += f"""
                    <a href="/pred/" class="btn btn-primary">🔄 Analisis Lagi</a>
                    <a href="/pred/history/" class="btn btn-secondary">📋 Lihat Riwayat</a>
                    {f'<a href="/pred/admin/dashboard/" class="btn btn-secondary">⚙️ Dashboard Admin</a>' if is_admin(request.user) else f'<a href="/pred/admin/dashboard/" class="btn btn-secondary">🔬 Dashboard Ahli</a>' if is_expert(request.user) else ''}
        """
    else:
        html_content += """
                    <a href="/pred/" class="btn btn-primary">🔄 Analisis Lagi</a>
                    <a href="/pred/login/" class="btn btn-secondary">🔐 Masuk untuk Riwayat</a>
        """

    html_content += """
                </div>
            </div>

            <!-- Footer -->
            <div style="background: #f8f9fa; padding: 20px; text-align: center; color: #6c757d; font-size: 0.9em; border-top: 1px solid #e9ecef;">
                <p>🔒 Hasil ini bersifat rahasia dan hanya untuk referensi awal</p>
                <p style="margin-top: 5px;">⚠️ Konsultasikan dengan profesional kesehatan mental untuk diagnosis yang akurat</p>
            </div>
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
        # Regular users see only their own submissions
        submissions = FormSubmission.objects.filter(user=request.user)
        page_title = "Riwayat Pengiriman Formulir Saya"
        user_info = f"Selamat datang, {request.user.get_full_name() or request.user.username}!"

    # Create styled HTML to display the history
    html_content = """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Riwayat Formulir - Sistem Prediksi Depresi</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                min-height: 100vh;
                padding: 20px;
            }

            .main-container {
                max-width: 70%;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }

            .header {
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white;
                padding: 30px;
                text-align: center;
                position: relative;
            }

            .header h1 {
                font-size: 2.2em;
                font-weight: 700;
                margin-bottom: 10px;
            }

            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }

            .nav-bar {
                background: #f8f9fa;
                padding: 15px 30px;
                border-bottom: 1px solid #e9ecef;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 15px;
            }

            .user-info {
                display: flex;
                align-items: center;
                gap: 10px;
                font-weight: 600;
                color: #495057;
            }

            .nav-links {
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
            }

            .nav-link {
                text-decoration: none;
                padding: 8px 16px;
                border-radius: 25px;
                font-weight: 500;
                transition: all 0.3s ease;
                display: inline-flex;
                align-items: center;
                gap: 5px;
                font-size: 0.9em;
            }

            .nav-link:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            }

            .nav-link.primary { background: #6366f1; color: white; }
            .nav-link.primary:hover { background: #4f46e5; }
            .nav-link.success { background: #8b5cf6; color: white; }
            .nav-link.success:hover { background: #7c3aed; }
            .nav-link.danger { background: #a855f7; color: white; }
            .nav-link.danger:hover { background: #9333ea; }

            .content-section {
                padding: 40px;
            }
            .page-title {
                text-align: center;
                margin-bottom: 30px;
                color: #2c3e50;
            }

            .page-title h2 {
                font-size: 1.8em;
                font-weight: 600;
                margin-bottom: 10px;
            }

            .page-title p {
                color: #6c757d;
                font-size: 1em;
            }

            .submissions-grid {
                display: grid;
                gap: 25px;
            }

            .submission-card {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 25px;
                border-left: 5px solid #6366f1;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
            }

            .submission-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            }

            .submission-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 2px solid #e9ecef;
            }

            .submission-id {
                font-size: 1.2em;
                font-weight: 600;
                color: #6366f1;
            }

            .submission-time {
                color: #6c757d;
                font-size: 0.9em;
                font-style: italic;
            }

            .submission-data {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }

            .data-field {
                background: white;
                padding: 12px 16px;
                border-radius: 10px;
                border: 1px solid #e9ecef;
                transition: all 0.3s ease;
            }

            .data-field:hover {
                border-color: #6366f1;
                box-shadow: 0 2px 8px rgba(99, 102, 241, 0.1);
            }

            .field-label {
                font-weight: 600;
                color: #2c3e50;
                font-size: 0.85em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 5px;
            }

            .field-value {
                color: #495057;
                font-size: 1em;
                font-weight: 500;
            }

            .prediction-result {
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                color: white;
                padding: 15px 20px;
                border-radius: 10px;
                margin-bottom: 15px;
                text-align: center;
            }

            .prediction-result.positive {
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            }

            .prediction-result.negative {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            }

            .reuse-btn {
                background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 0.9em;
                font-weight: 600;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 5px;
                transition: all 0.3s ease;
            }

            .reuse-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
                color: white;
                text-decoration: none;
            }

            .no-data {
                text-align: center;
                color: #6c757d;
                font-style: italic;
                padding: 60px 40px;
                background: #f8f9fa;
                border-radius: 15px;
                margin: 40px 0;
            }

            .no-data h3 {
                font-size: 1.5em;
                margin-bottom: 15px;
                color: #495057;
            }

            .no-data p {
                font-size: 1.1em;
                margin-bottom: 20px;
            }

            @media (max-width: 768px) {
                body {
                    padding: 10px;
                }

                .main-container {
                    border-radius: 15px;
                }

                .header {
                    padding: 20px;
                }

                .header h1 {
                    font-size: 1.8em;
                }

                .nav-bar {
                    padding: 15px 20px;
                    flex-direction: column;
                    align-items: stretch;
                    gap: 10px;
                }

                .nav-links {
                    justify-content: center;
                }

                .content-section {
                    padding: 30px 20px;
                }

                .submission-data {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="main-container">
            <!-- Header Section -->
            <div class="header">
                <h1>📋 {page_title}</h1>
                <p>Riwayat dan Analisis Data Formulir</p>
            </div>

            <!-- Navigation Bar -->
            <div class="nav-bar">
                <div class="user-info">
                    <span>👤</span>
                    <span>{user_info}</span>
                </div>
                <div class="nav-links">
                    <a href="/pred/" class="nav-link primary">🏠 Formulir Utama</a>
                    """ + (f'<a href="/pred/admin/dashboard/" class="nav-link success">⚙️ Dashboard Admin</a>' if is_admin(request.user) else f'<a href="/pred/admin/dashboard/" class="nav-link success">🔬 Dashboard Ahli</a>' if is_expert(request.user) else '') + f"""
                    <a href="/pred/logout/" class="nav-link danger">🚪 Keluar</a>
                </div>
            </div>

            <!-- Content Section -->
            <div class="content-section">
                <div class="page-title">
                    <h2>Daftar Pengiriman Formulir</h2>
                    <p>Lihat dan kelola riwayat pengiriman formulir Anda</p>
                </div>

                <div class="submissions-grid">
    """

    if submissions.exists():
        for submission in submissions:
            # Convert UTC time to local time using Django's timezone.localtime
            local_time = timezone.localtime(submission.submitted_at)

            # Determine prediction result class
            prediction_class = ""
            if submission.prediction_result == 'Positif':
                prediction_class = "positive"
            elif submission.prediction_result == 'Negatif':
                prediction_class = "negative"

            html_content += f"""
                    <div class="submission-card">
                        <div class="submission-header">
                            <div class="submission-id">📋 Submission #{submission.id}</div>
                            <div class="submission-time">📅 {local_time.strftime('%d %B %Y, %H:%M')} WIB</div>
                        </div>

                        <!-- Prediction Result -->
                        {f'<div class="prediction-result {prediction_class}"><strong>🧠 Hasil Prediksi: {submission.prediction_result}</strong>{f" ({submission.prediction_probability:.1f}%)" if submission.prediction_probability else ""}</div>' if submission.prediction_result else ''}

                            <div class="data-field">
                                <div class="field-label">👤 Jenis Kelamin</div>
                                <div class="field-value">{get_gender_display(submission.gender)}</div>
                            </div>
                            <div class="data-field">
                                <div class="field-label">🎂 Usia</div>
                                <div class="field-value">{submission.age} tahun</div>
                            </div>
                            <div class="data-field">
                                <div class="field-label">💼 Tekanan Kerja</div>
                                <div class="field-value">{get_pressure_display(submission.work_pressure)}</div>
                            </div>
                            <div class="data-field">
                                <div class="field-label">😊 Kepuasan Kerja</div>
                                <div class="field-value">{get_pressure_display(submission.job_satisfaction)}</div>
                            </div>
                            <div class="data-field">
                                <div class="field-label">💰 Stres Keuangan</div>
                                <div class="field-value">{get_pressure_display(submission.financial_stress)}</div>
                            </div>
                            <div class="data-field">
                                <div class="field-label">😴 Durasi Tidur</div>
                                <div class="field-value">{get_sleep_duration_display(submission.sleep_duration)}</div>
                            </div>
                            <div class="data-field">
                                <div class="field-label">⏰ Jam Kerja</div>
                                <div class="field-value">{submission.work_hours} jam/hari</div>
                            </div>
                            <div class="data-field">
                                <div class="field-label">🍽️ Kebiasaan Makan</div>
                                <div class="field-value">{get_dietary_habits_display(submission.dietary_habits)}</div>
                            </div>
                            <div class="data-field">
                                <div class="field-label">🧠 Pikiran Bunuh Diri</div>
                                <div class="field-value">{get_yes_no_display(submission.suicidal_thoughts)}</div>
                            </div>
                            <div class="data-field">
                                <div class="field-label">👨‍👩‍👧‍👦 Riwayat Keluarga</div>
                                <div class="field-value">{get_yes_no_display(submission.family_history_of_mental_illness)}</div>
                            </div>
                        </div>
                    </div>
            """

            # Add similarity score if available
            if submission.similarity_score:
                html_content += f"""
                    <div style="margin-top: 15px; padding: 15px; background: #e3f2fd; border-radius: 10px;">
                        <strong>📊 Skor Kemiripan:</strong> {submission.similarity_score:.1f}% (Kasus #{submission.similar_case_id})
                    </div>
                """

            # Add action buttons for experts
            if is_expert(request.user):
                if submission.is_reused_in_dataset:
                    html_content += f"""
                    <div style="margin-top: 15px; background: #d4edda; padding: 10px; border-radius: 8px; color: #155724; font-weight: bold; text-align: center;">
                        ✅ Ditambahkan ke Dataset
                        <br><small>Oleh {submission.reused_by.username if submission.reused_by else "Unknown"} pada {submission.reused_at.strftime("%d %B %Y, %H:%M") if submission.reused_at else "Unknown date"}</small>
                    </div>
                    """
                else:
                    html_content += f"""
                    <div style="margin-top: 15px; text-align: center;">
                        <a href="/pred/expert/reuse-data/{submission.id}/" class="reuse-btn" onclick="return confirm('Apakah Anda yakin ingin menambahkan data pengiriman ini ke dataset? Tindakan ini tidak dapat dibatalkan.')">
                            🔄 Tambahkan ke Dataset
                        </a>
                    </div>
                    """
    else:
        html_content += """
                <div class="no-data">
                    <h3>📭 Belum Ada Data</h3>
                    <p>Tidak ada pengiriman formulir yang ditemukan.</p>
                    <a href="/pred/" class="reuse-btn">🚀 Buat Formulir Pertama</a>
                </div>
        """

    html_content += """
                </div>
            </div>

            <!-- Footer -->
            <div style="background: #f8f9fa; padding: 20px; text-align: center; color: #6c757d; font-size: 0.9em; border-top: 1px solid #e9ecef;">
                <p>📊 Sistem Prediksi Depresi - Riwayat Data Formulir</p>
                <p style="margin-top: 5px;">🔒 Data Anda aman dan terlindungi</p>
            </div>
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
                        <td>{get_gender_display(submission.gender)}</td>
                        <td>{submission.age}</td>
                        <td>{get_pressure_display(submission.work_pressure)}</td>
                        <td>{get_pressure_display(submission.job_satisfaction)}</td>
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

