#!/usr/bin/env python3
"""
Django Depression Prediction System Setup Script
This script automates the initial setup process for the application.
"""

import os
import sys
import subprocess
import django
from django.core.management import execute_from_command_line

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def setup_django():
    """Setup Django environment"""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
    django.setup()

def main():
    print("🧠 Django Depression Prediction System Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies. Please check your pip installation.")
        sys.exit(1)
    
    # Setup Django
    setup_django()
    
    # Run migrations
    if not run_command("python manage.py makemigrations", "Creating migrations"):
        print("❌ Failed to create migrations.")
        sys.exit(1)
    
    if not run_command("python manage.py migrate", "Applying migrations"):
        print("❌ Failed to apply migrations.")
        sys.exit(1)
    
    # Create Expert group
    if not run_command("python manage.py create_expert_group --create-group", "Creating Expert group"):
        print("⚠️ Expert group creation failed, but continuing...")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Create a superuser: python manage.py createsuperuser")
    print("2. Add users to Expert group: python manage.py create_expert_group --add-user username")
    print("3. Run the server: python manage.py runserver")
    print("4. Visit: http://127.0.0.1:8000/pred/")
    
    # Ask if user wants to create superuser
    create_superuser = input("\n❓ Do you want to create a superuser now? (y/n): ").lower().strip()
    if create_superuser in ['y', 'yes']:
        run_command("python manage.py createsuperuser", "Creating superuser")
    
    print("\n🚀 Setup complete! You can now run: python manage.py runserver")

if __name__ == "__main__":
    main()
