#!/usr/bin/env python3
"""
Simple test for utils import and EasyOCR status
"""

print("🧪 Testing Utils Import Only")
print("=" * 40)

try:
    print("Importing utils...")
    import utils
    print(f"✅ Utils imported successfully")
    
    # Check capabilities
    if hasattr(utils, 'get_extraction_capabilities'):
        capabilities, methods = utils.get_extraction_capabilities()
        print(f"\n📋 Extraction Methods:")
        for i, method in enumerate(methods, 1):
            status = "✅" if ("disabled" not in method) else "⚠️"
            print(f"  {i}. {status} {method}")
            
        print(f"\n🔧 Capabilities:")
        for key, value in capabilities.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
    
    # Check EasyOCR specifically
    print(f"\n🔍 EasyOCR Status:")
    print(f"  ocr_reader: {utils.ocr_reader is not None}")
    if hasattr(utils, 'EASYOCR_DISABLED'):
        print(f"  EASYOCR_DISABLED: {utils.EASYOCR_DISABLED}")
    
    print(f"\n🎉 Utils loaded successfully without critical errors!")
    
except Exception as e:
    print(f"❌ Error importing utils: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 40)
print(f"💡 If EasyOCR is disabled, that's OK - PyMuPDF + LLM still work")
print(f"💡 The main goal is no critical import errors")
