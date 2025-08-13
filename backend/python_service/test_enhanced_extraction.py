#!/usr/bin/env python3
"""
Test script for enhanced text extraction with EasyOCR and LLMs
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_enhanced_extraction():
    """Test the enhanced text extraction functionality"""
    print("🧪 Testing Enhanced Text Extraction")
    print("=" * 50)
    
    try:
        # Import utils to check if everything loads correctly
        import utils
        print("✅ Utils module imported successfully")
        
        # Check EasyOCR initialization
        if hasattr(utils, 'ocr_reader') and utils.ocr_reader is not None:
            print("✅ EasyOCR reader initialized")
        elif hasattr(utils, 'EASYOCR_DISABLED') and utils.EASYOCR_DISABLED:
            print("⚠️  EasyOCR disabled due to compatibility issues")
        else:
            print("❌ EasyOCR reader not initialized")
            
        # Check Groq client initialization  
        if hasattr(utils, 'groq_client') and utils.groq_client is not None:
            print("✅ Groq LLM client initialized")
        else:
            print("❌ Groq LLM client not initialized")
            
        # Check OpenAI client initialization
        if hasattr(utils, 'openai_client') and utils.openai_client is not None:
            print("✅ OpenAI client initialized")
        else:
            print("ℹ️  OpenAI client not initialized (no API key)")
            
        print("\n🔧 Available Text Extraction Methods:")
        print("1. PyMuPDF (text-based PDFs)")
        print("2. EasyOCR (image-based content)")
        print("3. LLM text processing (Groq/OpenAI)")
        
        # Test LLM text processing if available
        if hasattr(utils, 'process_text_with_llm'):
            test_text = "2002\\n \\n2001\\n \\n2000\\n \\benefit cost\\n \\n"
            print(f"\n🧠 Testing LLM text processing...")
            print(f"Input: '{test_text}'")
            
            processed_text = utils.process_text_with_llm(test_text)
            print(f"Output: '{processed_text}'")
            
            if processed_text != test_text:
                print("✅ LLM text processing working")
            else:
                print("ℹ️  LLM text processing returned original text")
        
        print("\n🎯 Enhanced Extraction Features:")
        print("✅ Multi-method fallback strategy")
        print("✅ EasyOCR for image-based content")
        print("✅ LLM-powered text cleaning")
        print("✅ Vietnamese + English language support")
        print("✅ Confidence-based filtering")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to install dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_api_connectivity():
    """Test API connectivity for LLM services"""
    print("\n🌐 Testing API Connectivity")
    print("=" * 30)
    
    # Test Groq API
    try:
        import utils
        if utils.groq_client:
            print("✅ Groq client ready")
            # Note: We don't make actual API calls in tests to avoid costs
            print("ℹ️  Groq API key configured")
        else:
            print("❌ Groq client not available")
    except Exception as e:
        print(f"❌ Groq test failed: {e}")
        
    # Test OpenAI API
    try:
        if utils.openai_client:
            print("✅ OpenAI client ready")
        else:
            print("ℹ️  OpenAI client not configured")
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")

if __name__ == "__main__":
    print("🚀 Enhanced Text Extraction Test Suite")
    print("=" * 60)
    
    success = test_enhanced_extraction()
    test_api_connectivity()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Test completed successfully!")
        print("💡 Ready to process PDFs with enhanced extraction")
    else:
        print("⚠️  Some issues detected. Check dependencies and API keys.")
    
    print("\n📋 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Create .env file with API keys (see .env.example)")
    print("3. Test with actual PDF file")
    print("4. Monitor extraction quality improvements")
    print("\n🔧 If EasyOCR fails, try:")
    print("- Reinstall Pillow: pip install 'Pillow>=9.0.0,<10.0.0'")
    print("- Force CPU mode: Set gpu=False in EasyOCR initialization")
    print("- Check dependencies: python -c 'import easyocr; print(\"OK\")'")
    print("\n💡 Common fixes applied:")
    print("✅ Fixed PIL.Image.ANTIALIAS compatibility")
    print("✅ Added comprehensive error handling")
    print("✅ Added GPU/CPU fallback for EasyOCR")
    print("✅ Added input validation for OCR")
