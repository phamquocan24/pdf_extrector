#!/usr/bin/env python3
"""
Advanced OCR Integration Test
Tests the new AdvancedOCREngine without fallback methods
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    AdvancedOCREngine, AdvancedImagePreprocessor,
    enhanced_cell_cropping_and_ocr, organize_structured_table_data,
    initialize_ocr_engine
)
import cv2
import numpy as np

def test_advanced_ocr_initialization():
    """Test AdvancedOCREngine initialization"""
    print("🧪 Testing AdvancedOCREngine Initialization...")
    
    try:
        advanced_ocr = AdvancedOCREngine()
        print("✅ AdvancedOCREngine initialized successfully")
        
        # Check available engines
        engines = []
        if advanced_ocr.easyocr_available:
            engines.append("EasyOCR")
        if advanced_ocr.tesseract_available:
            engines.append("Tesseract")
        if advanced_ocr.paddleocr_available:
            engines.append("PaddleOCR")
        
        print(f"🎯 Available engines: {engines}")
        
        if not engines:
            print("❌ No OCR engines available!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedOCREngine initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_preprocessing():
    """Test AdvancedImagePreprocessor"""
    print("\n🧪 Testing Advanced Image Preprocessing...")
    
    try:
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        preprocessor = AdvancedImagePreprocessor()
        
        # Test multi-scale enhancement
        enhanced = preprocessor.multi_scale_enhancement(test_image)
        print(f"✅ Multi-scale enhancement: {test_image.shape} -> {enhanced.shape}")
        
        # Test adaptive binarization
        binary_methods = preprocessor.adaptive_binarization(enhanced)
        print(f"✅ Adaptive binarization: {len(binary_methods)} methods")
        
        return True
        
    except Exception as e:
        print(f"❌ Image preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ocr_comprehensive_extraction():
    """Test comprehensive OCR extraction"""
    print("\n🧪 Testing OCR Comprehensive Extraction...")
    
    try:
        # Initialize OCR engine
        advanced_ocr = AdvancedOCREngine()
        
        # Create a simple test image with text
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255  # White background
        
        # Add some simple text-like pattern
        cv2.rectangle(test_image, (50, 30), (250, 70), (0, 0, 0), 2)  # Black rectangle
        cv2.putText(test_image, "TEST", (80, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test comprehensive extraction
        text, confidence, engine = advanced_ocr.extract_text_comprehensive(test_image)
        
        print(f"✅ OCR Results:")
        print(f"   Text: '{text}'")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Engine: {engine}")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR comprehensive extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_cell_processing():
    """Test enhanced cell cropping and OCR"""
    print("\n🧪 Testing Enhanced Cell Processing...")
    
    try:
        # Initialize OCR engine first
        initialize_ocr_engine()
        
        # Create test image and detections
        test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        
        # Add some test cells
        cv2.rectangle(test_image, (50, 50), (150, 90), (0, 0, 0), 1)
        cv2.putText(test_image, "Cell1", (60, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.rectangle(test_image, (200, 50), (300, 90), (0, 0, 0), 1)
        cv2.putText(test_image, "Cell2", (210, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Create detections data
        detections = {
            'cells': [
                [50, 50, 150, 90],
                [200, 50, 300, 90]
            ]
        }
        
        # Test enhanced processing
        results = enhanced_cell_cropping_and_ocr(detections, test_image, save_enhanced_crops=False)
        
        print(f"✅ Enhanced processing: {len(results)} results")
        for i, result in enumerate(results):
            print(f"   Cell {i+1}: '{result['text']}' (conf: {result['confidence']:.3f}, engine: {result['ocr_engine']})")
        
        # Test structured data organization
        structured = organize_structured_table_data(results)
        print(f"✅ Structured data organization: {len(structured['structured_data'])} detections")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced cell processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Advanced OCR Integration Tests")
    print("=" * 50)
    
    tests = [
        test_advanced_ocr_initialization,
        test_image_preprocessing,
        test_ocr_comprehensive_extraction,
        test_enhanced_cell_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"🏆 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Advanced OCR integration is working properly.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
