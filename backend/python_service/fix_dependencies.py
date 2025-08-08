#!/usr/bin/env python3
"""
Fix NumPy compatibility issues by downgrading to compatible versions
"""

import subprocess
import sys
import os

def run_cmd(cmd):
    """Run command and return result"""
    print(f"🔧 Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success: {cmd}")
            return True
        else:
            print(f"❌ Failed: {cmd}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception running {cmd}: {e}")
        return False

def fix_numpy_compatibility():
    """Fix NumPy compatibility issues"""
    print("🔧 Fixing NumPy compatibility issues...")
    
    # Commands to fix dependencies
    commands = [
        # Uninstall problematic packages
        "pip uninstall -y numpy pandas matplotlib easyocr paddlepaddle paddleocr",
        
        # Install compatible NumPy first
        "pip install numpy==1.23.5",
        
        # Install compatible versions
        "pip install pandas==1.5.3",
        "pip install matplotlib==3.6.3",
        "pip install Pillow==9.5.0",
        
        # Install OCR packages with compatible versions
        "pip install easyocr==1.6.2",
        "pip install paddlepaddle==2.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple",
        "pip install paddleocr==2.6.1.3",
        
        # Reinstall other packages
        "pip install pytesseract opencv-python-headless scipy openpyxl",
    ]
    
    success_count = 0
    for cmd in commands:
        if run_cmd(cmd):
            success_count += 1
        else:
            print(f"⚠️ Command failed, continuing...")
    
    print(f"✅ Completed {success_count}/{len(commands)} commands successfully")
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_imports = [
        "import numpy",
        "import pandas", 
        "import matplotlib",
        "import easyocr",
        "import paddleocr",
        "import cv2",
        "import PIL"
    ]
    
    for test_import in test_imports:
        try:
            exec(test_import)
            print(f"✅ {test_import}")
        except Exception as e:
            print(f"❌ {test_import}: {e}")

if __name__ == "__main__":
    fix_numpy_compatibility()
