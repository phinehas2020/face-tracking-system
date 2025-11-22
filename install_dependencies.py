#!/usr/bin/env python3
"""
Dependency installer with retry logic and alternative sources
"""

import subprocess
import sys
import time

def run_command(cmd, retries=3, delay=2):
    """Run a command with retry logic"""
    for attempt in range(retries):
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                return False, str(e)
    return False, "Max retries exceeded"

def main():
    print("Installing dependencies with retry logic...")
    
    # Essential packages - install one by one with retries
    packages = [
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "ultralytics>=8.2.0",
        "insightface>=0.7.3",
        "onnxruntime>=1.16.0",
        "scikit-learn>=1.3.0",
        "sqlalchemy>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0"
    ]
    
    failed = []
    
    for package in packages:
        print(f"\nInstalling {package}...")
        success, output = run_command(f"{sys.executable} -m pip install '{package}'")
        if success:
            print(f"✓ {package} installed successfully")
        else:
            print(f"✗ Failed to install {package}")
            failed.append(package)
    
    if failed:
        print("\n⚠️  The following packages failed to install:")
        for pkg in failed:
            print(f"  - {pkg}")
        print("\nYou can try installing them manually with:")
        print(f"  {sys.executable} -m pip install " + " ".join(f"'{pkg}'" for pkg in failed))
        return 1
    else:
        print("\n✅ All packages installed successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())