#!/usr/bin/env python3
"""
Test OCR fallback functionality when engines are not available
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock unavailable OCR engines
import utils
utils.EASYOCR_AVAILABLE = False
utils.PYTESSERACT_AVAILABLE = False  
utils.PADDLEOCR_AVAILABLE = False

from utils import (
    AdvancedOCREngine, AdvancedImagePreprocessor,
    enhanced_cell_cropping_and_ocr, organize_structured_table_data,
    initialize_ocr_engine
)
import cv2
import numpy as np

def test_fallback_ocr():
    """Test OCR with fallback when no engines available"""
    print("🧪 Testing OCR Fallback Mode...")
    
    try:
        # Force unavailable engines
        print("🔧 Simulating unavailable OCR engines...")
        
        # Initialize OCR engine in fallback mode
        advanced_ocr = AdvancedOCREngine()
        print("✅ AdvancedOCREngine initialized in fallback mode")
        
        # Create test image
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(test_image, (50, 30), (250, 70), (0, 0, 0), 2)
        cv2.putText(test_image, "TEST", (80, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test comprehensive extraction in fallback mode
        text, confidence, engine = advanced_ocr.extract_text_comprehensive(test_image)
        
        print(f"✅ Fallback OCR Results:")
        print(f"   Text: '{text}'")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Engine: {engine}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback OCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_processing_fallback():
    """Test enhanced cell processing with fallback"""
    print("\n🧪 Testing Enhanced Processing with Fallback...")
    
    try:
        # Initialize OCR engine first (will be in fallback mode)
        initialize_ocr_engine()
        
        # Create test image and detections
        test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        
        # Add test cells
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
        
        print(f"✅ Enhanced processing with fallback: {len(results)} results")
        for i, result in enumerate(results):
            print(f"   Cell {i+1}: '{result['text']}' (conf: {result['confidence']:.3f}, engine: {result['ocr_engine']})")
        
        # Test structured data organization
        structured = organize_structured_table_data(results)
        print(f"✅ Structured data with fallback: {len(structured['structured_data'])} detections")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced processing fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_preprocessing_standalone():
    """Test image preprocessing without OCR"""
    print("\n🧪 Testing Standalone Image Preprocessing...")
    
    try:
        # Create test image
        test_image = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
        
        preprocessor = AdvancedImagePreprocessor()
        
        # Test multi-scale enhancement
        enhanced = preprocessor.multi_scale_enhancement(test_image)
        print(f"✅ Multi-scale enhancement: {test_image.shape} -> {enhanced.shape}")
        
        # Test adaptive binarization
        binary_methods = preprocessor.adaptive_binarization(enhanced)
        print(f"✅ Adaptive binarization: {len(binary_methods)} methods")
        
        for method_name, binary_img in binary_methods:
            print(f"   - Method: {method_name}, Shape: {binary_img.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run fallback tests"""
    print("🚀 Starting OCR Fallback Integration Tests")
    print("=" * 50)
    
    tests = [
        test_fallback_ocr,
        test_enhanced_processing_fallback,
        test_image_preprocessing_standalone
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
        print("✅ All fallback tests passed! OCR integration works even without engines.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
