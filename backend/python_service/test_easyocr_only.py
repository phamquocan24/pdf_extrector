#!/usr/bin/env python3
"""
Test OCR integration with EasyOCR only
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Disable PaddleOCR to avoid long download time
import utils
utils.PADDLEOCR_AVAILABLE = False

from utils import (
    EasyOCREngine, AdvancedImagePreprocessor,
    enhanced_cell_cropping_and_ocr, organize_structured_table_data,
    initialize_ocr_engine, EASYOCR_AVAILABLE, PYTESSERACT_AVAILABLE
)
import cv2
import numpy as np

def create_test_image_with_text():
    """Create a clear test image with readable text"""
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Add clear text
    cv2.putText(img, "HEADER 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "HEADER 2", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "HEADER 3", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.putText(img, "DATA 1", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "VALUE A", (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "RESULT X", (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    cv2.putText(img, "DATA 2", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "VALUE B", (250, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "RESULT Y", (450, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return img

def test_easyocr_availability():
    """Test EasyOCR availability"""
    print("🧪 Testing EasyOCR Availability...")
    print(f"EasyOCR Available: {EASYOCR_AVAILABLE}")
    print(f"Tesseract Available: {PYTESSERACT_AVAILABLE}")
    return EASYOCR_AVAILABLE

def test_easyocr_engine():
    """Test EasyOCREngine with EasyOCR"""
    print("\n🧪 Testing EasyOCREngine with EasyOCR...")
    
    try:
        advanced_ocr = EasyOCREngine()
        print("✅ EasyOCREngine initialized")
        
        # Check which engines are available
        engines = []
        if advanced_ocr.easyocr_available:
            engines.append("EasyOCR")
        # EasyOCREngine is designed for EasyOCR only
        # Tesseract and PaddleOCR are not available in this engine
        
        print(f"🎯 Available engines: {engines}")
        
        if advanced_ocr.easyocr_available:
            print("✅ EasyOCR is ready for testing!")
            return True
        else:
            print("❌ EasyOCR is not available")
            return False
            
    except Exception as e:
        print(f"❌ EasyOCREngine initialization failed: {e}")
        return False

def test_easyocr_text_extraction():
    """Test actual text extraction with EasyOCR"""
    print("\n🧪 Testing EasyOCR Text Extraction...")
    
    try:
        # Create test image
        test_image = create_test_image_with_text()
        
        # Initialize OCR engine
        advanced_ocr = EasyOCREngine()
        
        if not advanced_ocr.easyocr_available:
            print("❌ EasyOCR not available for testing")
            return False
        
        # Test on different parts of the image
        test_regions = [
            test_image[20:80, 30:180],   # "HEADER 1"
            test_image[70:130, 30:180],  # "DATA 1"
            test_image[120:180, 30:180], # "DATA 2"
            test_image[20:80, 230:380],  # "HEADER 2"
        ]
        
        success_count = 0
        total_tests = len(test_regions)
        
        for i, region in enumerate(test_regions):
            print(f"   Testing region {i+1}/{total_tests}...")
            
            # Extract text using comprehensive method
            text, confidence, engine = advanced_ocr.extract_text_comprehensive(region)
            
            print(f"      Result: '{text}' (confidence: {confidence:.3f}, engine: {engine})")
            
            # Check if we got meaningful text
            if text and text.strip() and confidence > 0.3:
                success_count += 1
                print(f"      ✅ Success!")
            else:
                print(f"      ⚠️ Low quality result")
        
        print(f"\n📊 EasyOCR Test Results: {success_count}/{total_tests} successful extractions")
        
        # Save test image for inspection
        cv2.imwrite("test_easyocr_image.png", test_image)
        print("📁 Test image saved as: test_easyocr_image.png")
        
        return success_count > 0
        
    except Exception as e:
        print(f"❌ EasyOCR text extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline_with_easyocr():
    """Test full pipeline with EasyOCR"""
    print("\n🧪 Testing Full Pipeline with EasyOCR...")
    
    try:
        # Initialize OCR engine
        initialize_ocr_engine()
        
        # Create test table
        test_image = create_test_image_with_text()
        
        # Mock cell detections based on text positions
        cell_detections = {
            'cells': [
                [30, 20, 180, 80],   # HEADER 1
                [230, 20, 380, 80],  # HEADER 2
                [430, 20, 580, 80],  # HEADER 3
                [30, 70, 180, 130],  # DATA 1
                [230, 70, 380, 130], # VALUE A
                [430, 70, 580, 130], # RESULT X
                [30, 120, 180, 180], # DATA 2
                [230, 120, 380, 180], # VALUE B
                [430, 120, 580, 180], # RESULT Y
            ]
        }
        
        # Process with enhanced OCR
        enhanced_results = enhanced_cell_cropping_and_ocr(
            cell_detections, 
            test_image, 
            save_enhanced_crops=False
        )
        
        print(f"✅ Enhanced processing: {len(enhanced_results)} cells processed")
        
        # Show results
        text_found = 0
        for i, result in enumerate(enhanced_results):
            text = result['text']
            if text and text.strip():
                text_found += 1
            print(f"   Cell {i+1}: '{text}' (conf: {result['confidence']:.3f}, engine: {result['ocr_engine']})")
        
        # Test structured data organization
        structured = organize_structured_table_data(enhanced_results)
        
        print(f"\n📊 Pipeline Summary:")
        print(f"   Total cells: {structured['summary']['total_cells']}")
        print(f"   Cells with text: {structured['summary']['cells_with_text']}")
        print(f"   Average confidence: {structured['summary']['confidence_avg']:.3f}")
        print(f"   Engines used: {structured['summary']['engines_used']}")
        
        # Success if we extracted text from at least half the cells
        success = text_found >= len(enhanced_results) // 2
        
        if success:
            print("✅ Full pipeline test: SUCCESS!")
        else:
            print("⚠️ Full pipeline test: Partial success")
        
        return success
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run EasyOCR-only tests"""
    print("🚀 Starting EasyOCR-Only Integration Tests")
    print("=" * 60)
    
    tests = [
        test_easyocr_availability,
        test_easyocr_engine,
        test_easyocr_text_extraction,
        test_full_pipeline_with_easyocr
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"🏆 EasyOCR Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED! EasyOCR integration is working perfectly!")
        print("🎉 OCR engines are now properly integrated and functional!")
    elif passed >= total - 1:
        print("✅ Almost all tests passed! EasyOCR is working well.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed >= total - 1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
