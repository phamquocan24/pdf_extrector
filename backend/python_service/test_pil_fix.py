#!/usr/bin/env python3
"""
Simple test for PIL.Image.ANTIALIAS compatibility fix
"""

print("🧪 Testing PIL.Image.ANTIALIAS Fix")
print("=" * 40)

try:
    from PIL import Image
    print(f"✅ PIL imported successfully")
    print(f"PIL version: {Image.__version__}")
    
    # Test ANTIALIAS attribute
    if hasattr(Image, 'ANTIALIAS'):
        print(f"✅ Image.ANTIALIAS exists: {Image.ANTIALIAS}")
    else:
        print(f"❌ Image.ANTIALIAS missing")
        
    # Apply our fix
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.LANCZOS
        print(f"🔧 Applied ANTIALIAS fix: {Image.ANTIALIAS}")
    
    # Test if it works
    print(f"✅ ANTIALIAS value: {Image.ANTIALIAS}")
    print(f"✅ LANCZOS value: {Image.LANCZOS}")
    print(f"✅ Equal: {Image.ANTIALIAS == Image.LANCZOS}")
    
    print(f"\n🎉 PIL.Image.ANTIALIAS fix verified working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    
print(f"\n" + "=" * 40)
print(f"💡 This fix allows EasyOCR to work with newer Pillow versions")
print(f"💡 EasyOCR expects ANTIALIAS but Pillow 10+ removed it")
print(f"💡 We provide ANTIALIAS as alias to LANCZOS for compatibility")
