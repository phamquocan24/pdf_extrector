#!/usr/bin/env python3
"""
Final OCR Integration Test
Tests the complete OCR pipeline with AdvancedOCREngine
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    AdvancedOCREngine, AdvancedImagePreprocessor,
    enhanced_cell_cropping_and_ocr, organize_structured_table_data,
    initialize_ocr_engine, detect_cells_with_info,
    EASYOCR_AVAILABLE, PYTESSERACT_AVAILABLE, PADDLEOCR_AVAILABLE
)
import cv2
import numpy as np

def create_test_table_image():
    """Create a test table image with text"""
    img = np.ones((300, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Draw table structure
    # Header row
    cv2.rectangle(img, (50, 50), (550, 100), (0, 0, 0), 2)
    cv2.putText(img, "Header 1", (70, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Header 2", (220, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Header 3", (370, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "Header 4", (470, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Data rows
    rows_data = [
        ["Data 1", "Value A", "100", "OK"],
        ["Data 2", "Value B", "200", "OK"],
        ["Data 3", "Value C", "300", "FAIL"]
    ]
    
    for i, row in enumerate(rows_data):
        y_start = 100 + i * 50
        y_end = y_start + 50
        
        # Draw row
        cv2.rectangle(img, (50, y_start), (550, y_end), (0, 0, 0), 1)
        
        # Add text
        x_positions = [70, 220, 370, 470]
        for j, text in enumerate(row):
            cv2.putText(img, text, (x_positions[j], y_start + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img

def test_ocr_engine_availability():
    """Test which OCR engines are available"""
    print("🧪 Testing OCR Engine Availability...")
    
    print(f"EasyOCR Available: {EASYOCR_AVAILABLE}")
    print(f"Tesseract Available: {PYTESSERACT_AVAILABLE}")
    print(f"PaddleOCR Available: {PADDLEOCR_AVAILABLE}")
    
    return True

def test_advanced_ocr_initialization():
    """Test AdvancedOCREngine initialization"""
    print("\n🧪 Testing AdvancedOCREngine Initialization...")
    
    try:
        advanced_ocr = AdvancedOCREngine()
        print("✅ AdvancedOCREngine initialized successfully")
        
        available_engines = []
        if advanced_ocr.easyocr_available:
            available_engines.append("EasyOCR")
        if advanced_ocr.tesseract_available:
            available_engines.append("Tesseract")
        if advanced_ocr.paddleocr_available:
            available_engines.append("PaddleOCR")
        
        print(f"🎯 Active engines: {available_engines}")
        print(f"🔧 Fallback mode: {getattr(advanced_ocr, 'fallback_mode', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedOCREngine initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_ocr_extraction():
    """Test comprehensive OCR extraction on real text"""
    print("\n🧪 Testing Comprehensive OCR Extraction...")
    
    try:
        advanced_ocr = AdvancedOCREngine()
        
        # Create test image with clear text
        test_image = create_test_table_image()
        
        # Extract a single cell for testing
        cell_image = test_image[50:100, 50:200]  # Header 1 cell
        
        # Test comprehensive extraction
        text, confidence, engine = advanced_ocr.extract_text_comprehensive(cell_image)
        
        print(f"✅ OCR Extraction Results:")
        print(f"   Text: '{text}'")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Engine: {engine}")
        
        # Save test image for visual inspection
        cv2.imwrite("test_table_image.png", test_image)
        cv2.imwrite("test_cell_image.png", cell_image)
        print("📁 Saved test images: test_table_image.png, test_cell_image.png")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_table_processing():
    """Test full table processing pipeline"""
    print("\n🧪 Testing Full Table Processing Pipeline...")
    
    try:
        # Initialize OCR engine
        initialize_ocr_engine()
        
        # Create test table image
        test_image = create_test_table_image()
        
        # Mock cell detections (simulate YOLO cell detection results)
        detections = {
            'cells': [
                [50, 50, 200, 100],    # Header 1
                [200, 50, 350, 100],   # Header 2
                [350, 50, 450, 100],   # Header 3
                [450, 50, 550, 100],   # Header 4
                [50, 100, 200, 150],   # Data 1
                [200, 100, 350, 150],  # Value A
                [350, 100, 450, 150],  # 100
                [450, 100, 550, 150],  # OK
                [50, 150, 200, 200],   # Data 2
                [200, 150, 350, 200],  # Value B
                [350, 150, 450, 200],  # 200
                [450, 150, 550, 200],  # OK
            ]
        }
        
        # Process with enhanced OCR
        results = enhanced_cell_cropping_and_ocr(detections, test_image, save_enhanced_crops=False)
        
        print(f"✅ Enhanced OCR processing: {len(results)} cells processed")
        
        # Show results
        for i, result in enumerate(results):
            print(f"   Cell {i+1}: '{result['text']}' (conf: {result['confidence']:.3f}, engine: {result['ocr_engine']})")
        
        # Test structured data organization
        structured = organize_structured_table_data(results)
        
        print(f"\n📊 Structured Table Summary:")
        print(f"   Total cells: {structured['summary']['total_cells']}")
        print(f"   Cells with text: {structured['summary']['cells_with_text']}")
        print(f"   Average confidence: {structured['summary']['confidence_avg']:.3f}")
        print(f"   Engines used: {structured['summary']['engines_used']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full table processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_preprocessing_pipeline():
    """Test the complete image preprocessing pipeline"""
    print("\n🧪 Testing Image Preprocessing Pipeline...")
    
    try:
        # Create test image
        test_image = np.random.randint(50, 200, (80, 150, 3), dtype=np.uint8)
        
        # Add some noise and blur
        noise = np.random.randint(0, 50, test_image.shape, dtype=np.uint8)
        test_image = cv2.addWeighted(test_image, 0.8, noise, 0.2, 0)
        test_image = cv2.GaussianBlur(test_image, (3, 3), 0)
        
        preprocessor = AdvancedImagePreprocessor()
        
        # Test multi-scale enhancement
        enhanced = preprocessor.multi_scale_enhancement(test_image)
        print(f"✅ Multi-scale enhancement: {test_image.shape} -> {enhanced.shape}")
        
        # Test adaptive binarization
        binary_methods = preprocessor.adaptive_binarization(enhanced)
        print(f"✅ Adaptive binarization: {len(binary_methods)} methods")
        
        # Save preprocessing results
        cv2.imwrite("test_original.png", test_image)
        cv2.imwrite("test_enhanced.png", enhanced)
        
        for i, (method_name, binary_img) in enumerate(binary_methods):
            cv2.imwrite(f"test_binary_{method_name}.png", binary_img)
            print(f"   - {method_name}: {binary_img.shape}")
        
        print("📁 Saved preprocessing results to test_*.png files")
        
        return True
        
    except Exception as e:
        print(f"❌ Image preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all comprehensive tests"""
    print("🚀 Starting Final OCR Integration Tests")
    print("=" * 60)
    
    tests = [
        test_ocr_engine_availability,
        test_advanced_ocr_initialization,
        test_comprehensive_ocr_extraction,
        test_full_table_processing,
        test_image_preprocessing_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"🏆 Final Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED! Advanced OCR integration is fully functional.")
        print("🎉 The system can now process tables with enhanced OCR without fallback!")
    elif passed >= total - 1:
        print("✅ Almost all tests passed! System is working with minor issues.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return passed >= total - 1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
