# 🧠 Django Depression Prediction System

A comprehensive Django web application that uses machine learning to predict depression likelihood based on user responses to a psychological assessment questionnaire. The system includes cosine similarity analysis, role-based access control, and dataset management features.

## ✨ Features

### 🎯 Core Functionality
- **ML Depression Prediction**: Uses a pre-trained joblib model for depression likelihood assessment
- **Cosine Similarity Analysis**: Compares user input with existing dataset to find similar cases
- **Indonesian Language Interface**: Complete Indonesian localization for better user experience
- **Real-time Results**: Instant prediction results with confidence scores

### 👥 User Roles
- **Anonymous Users**: Can submit forms and view results
- **Regular Users**: Access to personal history after login
- **Expert Users**: All admin features except user management + dataset reuse functionality
- **Admin Users**: Complete system access including user management

### 📊 Advanced Features
- **Dataset Reuse**: Expert users can add form submissions back to the training dataset
- **Similarity Matching**: Find and display most similar cases from the dataset
- **Comprehensive Dashboard**: Statistics and management interface for admins/experts
- **Audit Trail**: Complete tracking of data reuse and user actions

## 🛠️ Technology Stack

- **Backend**: Django 4.2+
- **Machine Learning**: scikit-learn, pandas, numpy
- **Model Format**: joblib
- **Database**: SQLite (default, easily changeable)
- **Frontend**: HTML5, CSS3, Tailwind CSS
- **Authentication**: Django built-in auth system

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## 🚀 Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/django-depression-prediction.git
cd django-depression-prediction
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Database Setup
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Create Superuser (Admin)
```bash
python manage.py createsuperuser
```

### 6. Create Expert Group
```bash
python manage.py create_expert_group --create-group
```

### 7. Add Users to Expert Group (Optional)
```bash
python manage.py create_expert_group --add-user username
```

### 8. Run Development Server
```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000/pred/` to access the application.

## 📁 Project Structure

```
djangotutorial/
├── mysite/                 # Django project settings
├── pred/                   # Main application
│   ├── management/         # Custom management commands
│   ├── migrations/         # Database migrations
│   ├── templates/          # HTML templates
│   ├── admin.py           # Admin interface configuration
│   ├── forms.py           # Form definitions
│   ├── models.py          # Database models
│   ├── urls.py            # URL routing
│   └── views.py           # View functions
├── dataset_processed.csv   # Training dataset
├── depression_prediction_model.joblib  # Pre-trained ML model
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎮 Usage Guide

### For Regular Users
1. Visit the main form page
2. Fill out the depression assessment questionnaire
3. Submit the form to get ML prediction results
4. View similarity analysis with existing cases
5. Register/login to access submission history

### For Expert Users
1. Login with expert credentials
2. Access expert dashboard for system overview
3. View all form submissions with detailed analysis
4. Use "Reuse Data" feature to add submissions to dataset
5. Monitor dataset growth and system statistics

### For Admin Users
1. Login with admin credentials
2. Access full admin dashboard
3. Manage users and assign roles
4. Access Django admin panel
5. Monitor system usage and data

## 🔧 Configuration

### Adding New Expert Users
```bash
python manage.py create_expert_group --add-user username
```

### Listing Expert Users
```bash
python manage.py create_expert_group --list-experts
```

### Removing Expert Users
```bash
python manage.py create_expert_group --remove-user username
```

## 📊 Model Information

The system uses a pre-trained machine learning model (`depression_prediction_model.joblib`) that analyzes:

- Gender, Age, Work Pressure, Job Satisfaction
- Financial Stress, Sleep Duration, Dietary Habits
- Suicidal Thoughts, Work Hours, Family History

**Note**: This is for educational/research purposes only and should not replace professional medical diagnosis.

## 🌐 Deployment

For production deployment, consider:

1. **Database**: Switch to PostgreSQL or MySQL
2. **Static Files**: Configure static file serving
3. **Security**: Update `SECRET_KEY` and security settings
4. **Environment Variables**: Use environment variables for sensitive data
5. **Web Server**: Use Gunicorn + Nginx for production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes only. The predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the development team.
