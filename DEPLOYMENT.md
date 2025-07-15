# 🚀 Deployment Guide

This guide will help you deploy the Django Depression Prediction System on different environments.

## 📋 Quick Setup (New PC)

### Option 1: Automated Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/django-depression-prediction.git
cd django-depression-prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Run automated setup
python setup.py
```

### Option 2: Manual Setup
```bash
# Clone and navigate
git clone https://github.com/yourusername/django-depression-prediction.git
cd django-depression-prediction

# Create and activate virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Database setup
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Create Expert group
python manage.py create_expert_group --create-group

# Run server
python manage.py runserver
```

## 🔧 Environment Configuration

### Development Environment
```bash
# .env file (create this file)
DEBUG=True
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///db.sqlite3
```

### Production Environment
```bash
# .env file for production
DEBUG=False
SECRET_KEY=your-production-secret-key
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
```

## 🗄️ Database Configuration

### SQLite (Default - Development)
No additional configuration needed. Database file will be created automatically.

### PostgreSQL (Recommended for Production)
```bash
# Install PostgreSQL adapter
pip install psycopg2-binary

# Update settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'depression_prediction_db',
        'USER': 'your_username',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

### MySQL
```bash
# Install MySQL adapter
pip install mysqlclient

# Update settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'depression_prediction_db',
        'USER': 'your_username',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
```

## 🌐 Production Deployment

### Using Gunicorn + Nginx

#### 1. Install Gunicorn
```bash
pip install gunicorn
```

#### 2. Create Gunicorn Configuration
```python
# gunicorn.conf.py
bind = "127.0.0.1:8000"
workers = 3
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 2
```

#### 3. Run with Gunicorn
```bash
gunicorn mysite.wsgi:application -c gunicorn.conf.py
```

#### 4. Nginx Configuration
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location /static/ {
        alias /path/to/your/project/static/;
    }

    location /media/ {
        alias /path/to/your/project/media/;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Using Docker

#### 1. Create Dockerfile
```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN python manage.py collectstatic --noinput
RUN python manage.py migrate

EXPOSE 8000

CMD ["gunicorn", "mysite.wsgi:application", "--bind", "0.0.0.0:8000"]
```

#### 2. Create docker-compose.yml
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      - DEBUG=False
    depends_on:
      - db

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: depression_prediction_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

#### 3. Run with Docker
```bash
docker-compose up -d
```

## 🔒 Security Considerations

### Production Settings
```python
# settings.py for production
DEBUG = False
ALLOWED_HOSTS = ['yourdomain.com', 'www.yourdomain.com']

# Security settings
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# HTTPS settings (if using HTTPS)
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
```

### Environment Variables
```bash
# Use environment variables for sensitive data
export SECRET_KEY="your-secret-key"
export DATABASE_URL="your-database-url"
export DEBUG="False"
```

## 📊 Static Files

### Development
```bash
# Collect static files
python manage.py collectstatic
```

### Production
Configure your web server (Nginx/Apache) to serve static files directly.

## 🔄 Updates and Maintenance

### Updating the Application
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic

# Restart server
```

### Backup Database
```bash
# SQLite
cp db.sqlite3 backup_$(date +%Y%m%d_%H%M%S).sqlite3

# PostgreSQL
pg_dump depression_prediction_db > backup_$(date +%Y%m%d_%H%M%S).sql
```

## 🐛 Troubleshooting

### Common Issues

1. **Module not found errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Database errors**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

3. **Static files not loading**
   ```bash
   python manage.py collectstatic
   ```

4. **Permission errors**
   ```bash
   # Make sure virtual environment is activated
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

### Logs and Debugging
```python
# Enable logging in settings.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'django.log',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
```

## 📞 Support

For deployment issues:
1. Check the logs
2. Verify all dependencies are installed
3. Ensure database is properly configured
4. Check file permissions
5. Open an issue on GitHub if problems persist
