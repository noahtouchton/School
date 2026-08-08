#!/usr/bin/env python3
"""
Single Launcher Script for Antigravity Fantasy Football AI Platform.
Running this single file initializes all databases, verifies dependencies, and launches the full Streamlit web application.

Usage:
    python3 run.py
"""

import os
import sys
import subprocess

def main():
    # Workspace root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Ensure project root is in sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    print("=" * 70)
    print("🏈 ANTIGRAVITY FANTASY FOOTBALL AI PLATFORM")
    print("=" * 70)
    print("Initializing SQLite database & checking local player cache...")
    
    # Auto-install missing requirements if needed
    try:
        import streamlit
        import pandas
        import pulp
        import nfl_data_py
        import espn_api
    except ImportError as missing_err:
        print(f"📦 Missing dependency detected ({missing_err}). Automatically installing requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies successfully installed!")
        except Exception as install_err:
            print(f"⚠️ Warning: Automatic pip install encountered an issue: {install_err}")
            
    try:
        from src.data import db
        db.init_db()
        print("✅ Database successfully initialized!")
    except Exception as e:
        print(f"⚠️ Note: Database initialization notice: {e}")
        
    app_path = os.path.join(project_root, "src", "ui", "app.py")
    if not os.path.exists(app_path):
        print(f"❌ Error: Could not find main UI app at {app_path}")
        sys.exit(1)
        
    print("\n🚀 Launching Web Application Dashboard...")
    print("Open your browser at: http://localhost:8501\n")
    print("=" * 70)
    
    # Launch Streamlit using the active Python executable
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless=false"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user. Goodbye!")

if __name__ == "__main__":
    main()
