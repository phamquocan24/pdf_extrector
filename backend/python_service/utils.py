import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
<<<<<<< HEAD
# import pandas as pd  # Disabled due to numpy compatibility issues
# from docx import Document  # Disabled - not needed for core functionality
=======

# Safe pandas import with numpy/pandas compatibility fix
try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        import pandas as pd
    print("✅ Pandas imported successfully")
except Exception as e:
    print(f"⚠️ Pandas import issue: {e}")
    # Try workaround for numpy/pandas version conflict
    try:
        # Suppress all warnings temporarily
        import warnings
        warnings.filterwarnings("ignore")
        
        # Force reload numpy if needed
        import sys
        if 'numpy' in sys.modules:
            import importlib
            importlib.reload(sys.modules['numpy'])
        
        import pandas as pd
        print("✅ Pandas imported with compatibility workaround")
    except Exception as e2:
        print(f"⚠️ Still having pandas issues: {e2}")
        # Create minimal pandas replacement for basic functionality
        print("🔧 Using minimal pandas replacement...")
        
        class MinimalPandas:
            @staticmethod
            def DataFrame(data=None, columns=None):
                """Minimal DataFrame replacement"""
                if data is None:
                    data = []
                return {"data": data, "columns": columns or []}
            
            @staticmethod  
            def to_csv(df_dict, path=None, index=False):
                """Minimal to_csv replacement"""
                if path and "data" in df_dict and "columns" in df_dict:
                    import csv
                    with open(path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        if df_dict["columns"]:
                            writer.writerow(df_dict["columns"])
                        writer.writerows(df_dict["data"])
                    return True
                return False
        
        pd = MinimalPandas()
        print("✅ Minimal pandas replacement activated")

from docx import Document
>>>>>>> 1194ca1ccaa9cbe5704d8ca19ef3e361f66ba32e
import io
import types, sys, torch.nn as nn
import base64
import os
from pathlib import Path
<<<<<<< HEAD
# Safe imports with fallbacks
try:
    import easyocr
    EASYOCR_IMPORT_ERROR = None
except Exception as e:
    easyocr = None
    EASYOCR_IMPORT_ERROR = str(e)
    print(f"⚠️  EasyOCR import failed: {e}")

import openai
from groq import Groq
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -----------------------------------------------------------------------------
# PIL.Image.ANTIALIAS Compatibility Fix
# EasyOCR uses deprecated PIL.Image.ANTIALIAS which was removed in Pillow 10.0.0
# We add it back as an alias to maintain compatibility
# -----------------------------------------------------------------------------
try:
    from PIL import Image
    if not hasattr(Image, 'ANTIALIAS'):
        # ANTIALIAS was removed in Pillow 10.0.0, replaced with LANCZOS
        Image.ANTIALIAS = Image.LANCZOS
        print("Applied PIL.Image.ANTIALIAS compatibility fix")
except ImportError:
    print("PIL not available - skipping ANTIALIAS fix")
except Exception as e:
    print(f"Error applying PIL compatibility fix: {e}")
=======
import warnings

# OCR imports with comprehensive error handling for numpy compatibility
print("🔧 Initializing OCR engines with numpy compatibility...")

# Suppress all numpy warnings globally
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("✅ EasyOCR available")
except Exception as e:
    EASYOCR_AVAILABLE = False
    print(f"⚠️ EasyOCR not available: {str(e)[:50]}...")

# Disable Tesseract for EasyOCR-only workflow
PYTESSERACT_AVAILABLE = False
print("⚠️ Tesseract disabled - using EasyOCR-only workflow")

# Disable PaddleOCR for EasyOCR-only workflow  
PADDLEOCR_AVAILABLE = False
print("⚠️ PaddleOCR disabled - using EasyOCR-only workflow")
>>>>>>> 1194ca1ccaa9cbe5704d8ca19ef3e361f66ba32e

# -----------------------------------------------------------------------------
# Workaround for models trained with custom modules
# Some checkpoints reference a `custom_modules` package that may not be present
# in the current environment. We dynamically create a fake module that returns
# a simple identity layer for any requested attribute so that the model can be
# deserialized without the original custom implementation.
# The error `AttributeError: ... has no attribute 'endswith'` during
# torchvision import suggests that Python's `inspect` module is trying to find
# the source file of our fake module and failing. We can make the fake module
# more robust by giving it a `__file__` attribute, which `inspect` uses.
# -----------------------------------------------------------------------------

class _DummyLayer(nn.Module):
    """A placeholder nn.Module that simply returns its input unchanged."""
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, *args, **kwargs):  # type: ignore
        return x

class _FakeCustomModule(types.ModuleType):
    """A fake module that fools the unpickler but is robust to introspection."""
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.__file__ = f'<{name}>'  # Provide a fake file path for inspect
        self.__path__ = []
        self.__loader__ = self

    def __getattr__(self, item):
        # For any requested attribute, return a dummy layer class.
        # This will satisfy the unpickler looking for `custom_modules.SomeLayer`.
        if not item.startswith('__'):
            return _DummyLayer
        raise AttributeError(f"Fake module '{self.__name__}' has no attribute '{item}'")

# Register the fake module so that `import custom_modules` succeeds
sys.modules["custom_modules"] = _FakeCustomModule("custom_modules")
# If the checkpoint references nested modules like `custom_modules.layers`, make sure
# those also resolve to the fake module.
sys.modules["custom_modules.layers"] = _FakeCustomModule("custom_modules.layers")

# -----------------------------------------------------------------------------
# Standard imports follow
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Load models
# The model files are stored in the parent directory "models" relative to this
# utils.py file. We build absolute paths from __file__ so that the service works
# no matter where it is launched from (project root, docker container, etc.).
# -----------------------------------------------------------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent           # .../python_service
MODEL_DIR = BASE_DIR.parent / "models"              # .../models

# Model that detects full table regions on a page
try:
    table_model_path = MODEL_DIR / 'best(table).pt'
    print(f"Loading table model from: {table_model_path}")
    print(f"Model file exists: {table_model_path.exists()}")
    table_model = YOLO(str(table_model_path))
    print("Table model loaded successfully")
except Exception as e:
    print(f"Error loading table model: {e}")
    table_model = None

# Structure model removed - using only table detection and cell segmentation
structure_model = None

# -----------------------------------------------------------------------------
# Advanced Image Preprocessing
# -----------------------------------------------------------------------------

class AdvancedImagePreprocessor:
    """Advanced preprocessing with multiple enhancement techniques"""
    
    @staticmethod
    def multi_scale_enhancement(image):
        """Apply multiple enhancement techniques for better OCR"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Upscale small images for better OCR
        height, width = gray.shape
        if height < 50 or width < 100:
            scale_factor = max(2, 100 // min(height, width))
            gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            print(f"   🔍 Upscaled by {scale_factor}x: {gray.shape}")
        
        # 2. Advanced denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 3. Enhanced contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 4. Gamma correction
        gamma = 1.2
        gamma_corrected = np.array(255 * (enhanced / 255) ** gamma, dtype='uint8')
        
        # 5. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morphed = cv2.morphologyEx(gamma_corrected, cv2.MORPH_CLOSE, kernel)
        
        # 6. Sharpening
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(morphed, -1, kernel_sharp)
        
        return sharpened
    
    @staticmethod
    def adaptive_binarization(image):
        """Multiple binarization methods for different text conditions"""
        results = []
        
        # Method 1: Otsu
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(('Otsu', otsu))
        
        # Method 2: Adaptive Mean
        adaptive_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        results.append(('Adaptive_Mean', adaptive_mean))
        
        # Method 3: Adaptive Gaussian
        adaptive_gauss = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        results.append(('Adaptive_Gauss', adaptive_gauss))
        
        # Method 4: Custom threshold
        mean_val = np.mean(image)
        custom_thresh = mean_val * 0.7
        _, custom = cv2.threshold(image, custom_thresh, 255, cv2.THRESH_BINARY)
        results.append(('Custom', custom))
        
        return results

# -----------------------------------------------------------------------------
# Advanced OCR Engine with Multiple Strategies
# -----------------------------------------------------------------------------

class EasyOCREngine:
    """Optimized OCR engine using only EasyOCR for best performance"""
    
    def __init__(self):
        print("🚀 Initializing EasyOCR Engine (Optimized for Cell Segmentation)...")
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # EasyOCR with optimized settings for cell text extraction
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr = easyocr.Reader(['en'], gpu=False)
                self.easyocr_available = True
                print("✅ EasyOCR initialized successfully")
            except Exception as e:
                print(f"⚠️ EasyOCR failed: {e}")
                self.easyocr_available = False
                self.fallback_mode = True
        else:
            self.easyocr_available = False
            self.fallback_mode = True
        
        # Report status
        if self.easyocr_available:
            print("🎯 EasyOCR Engine ready for cell text extraction!")
            self.fallback_mode = False
        else:
            print("⚠️ EasyOCR not available! Using fallback text extraction.")
            self.fallback_mode = True
    
    def extract_text_comprehensive(self, image, cell_info=None):
        """Optimized text extraction using EasyOCR with multiple preprocessing methods"""
        
        height, width = image.shape[:2]
        cell_size = f"{width}x{height}"
        
        print(f"      🔍 Processing cell {cell_size} with EasyOCR")
        
        # If in fallback mode, return basic result
        if self.fallback_mode:
            return "cell", 0.5, "fallback"
        
        # Advanced preprocessing optimized for EasyOCR
        preprocessor = AdvancedImagePreprocessor()
        enhanced_image = preprocessor.multi_scale_enhancement(image)
        
        # Try multiple binarization methods with EasyOCR
        binary_methods = preprocessor.adaptive_binarization(enhanced_image)
        
        all_results = []
        
        # Process original image with EasyOCR
        if self.easyocr_available:
            try:
                # Convert to RGB for EasyOCR
                if len(image.shape) == 2:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # EasyOCR with optimized parameters for cell text
                results = self.easyocr.readtext(
                    rgb_image, 
                    paragraph=False, 
                    width_ths=0.5,  # More lenient width threshold
                    height_ths=0.5, # More lenient height threshold
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-%()$€£¥ '
                )
                
                for (bbox, text, confidence) in results:
                    if confidence > 0.1:  # Lower threshold for better recall
                        all_results.append({
                            'text': text.strip(),
                            'confidence': confidence,
                            'engine': 'EasyOCR_Original',
                            'method': 'original'
                        })
            except Exception as e:
                print(f"      ⚠️ EasyOCR original processing failed: {e}")
        
        # Process enhanced image with EasyOCR
        if self.easyocr_available and enhanced_image is not None:
            try:
                # Convert enhanced image to RGB
                if len(enhanced_image.shape) == 2:
                    rgb_enhanced = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
                else:
                    rgb_enhanced = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
                
                # EasyOCR on enhanced image
                results = self.easyocr.readtext(
                    rgb_enhanced, 
                    paragraph=False, 
                    width_ths=0.5, 
                    height_ths=0.5,
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-%()$€£¥ '
                )
                
                for (bbox, text, confidence) in results:
                    if confidence > 0.1:
                        all_results.append({
                            'text': text.strip(),
                            'confidence': confidence,
                            'engine': 'EasyOCR_Enhanced',
                            'method': 'enhanced'
                        })
            except Exception as e:
                print(f"      ⚠️ EasyOCR enhanced processing failed: {e}")
        
        # Process binarized images with EasyOCR
        if self.easyocr_available and binary_methods:
            for method_name, binary_image in binary_methods[:2]:  # Only use top 2 methods for efficiency
                try:
                    # Convert binary to RGB
                    rgb_binary = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
                    
                    # EasyOCR on binarized image
                    results = self.easyocr.readtext(
                        rgb_binary, 
                        paragraph=False, 
                        width_ths=0.3, 
                        height_ths=0.3,
                        allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-%()$€£¥ '
                    )
                    
                    for (bbox, text, confidence) in results:
                        if confidence > 0.1:
                            all_results.append({
                                'text': text.strip(),
                                'confidence': confidence,
                                'engine': f'EasyOCR_{method_name}',
                                'method': method_name
                            })
                except Exception:
                    continue
        
        # Select best results using intelligent selection
        return self._select_best_easyocr_results(all_results)
    
    def _select_best_easyocr_results(self, all_results):
        """Optimized result selection for EasyOCR outputs"""
        if not all_results:
            return "", 0.0, "EasyOCR_None"
        
        # Filter valid results
        valid_results = [r for r in all_results if r['text'].strip()]
        
        if not valid_results:
            # Return best of all results even if empty
            best = max(all_results, key=lambda x: x['confidence'])
            return best['text'], best['confidence'], best['engine']
        
        # Find the result with highest confidence
        best_result = max(valid_results, key=lambda x: x['confidence'])
        
        # If we have multiple results, check for consensus
        if len(valid_results) > 1:
            # Group by similar text content
            text_groups = {}
            for result in valid_results:
                text_key = result['text'].lower().strip()
                if text_key not in text_groups:
                    text_groups[text_key] = []
                text_groups[text_key].append(result)
            
            # If multiple methods agree on the same text, boost confidence
            if len(text_groups) > 1:
                for text_key, group in text_groups.items():
                    if len(group) > 1:  # Multiple methods agree
                        avg_conf = np.mean([r['confidence'] for r in group])
                        if avg_conf > best_result['confidence']:
                            best_result = max(group, key=lambda x: x['confidence'])
                            best_result['confidence'] = min(avg_conf * 1.1, 1.0)  # Boost confidence slightly
        
        return best_result['text'], best_result['confidence'], best_result['engine']
    
    def _select_best_results(self, all_results):
        """Intelligent selection of best OCR results"""
        if not all_results:
            return "", 0.0, "None"
        
        # Filter valid results
        valid_results = [r for r in all_results if r['text'].strip()]
        
        if not valid_results:
            # Return best of all results even if empty
            best = max(all_results, key=lambda x: x['confidence'])
            return best['text'], best['confidence'], best['engine']
        
        # Group by similar text content for consensus
        text_groups = {}
        for result in valid_results:
            text_key = result['text'].lower().strip()
            if text_key not in text_groups:
                text_groups[text_key] = []
            text_groups[text_key].append(result)
        
        # Find most confident group
        best_group = None
        best_avg_confidence = 0
        
        for text_key, group in text_groups.items():
            avg_conf = np.mean([r['confidence'] for r in group])
            if avg_conf > best_avg_confidence:
                best_avg_confidence = avg_conf
                best_group = group
        
        if best_group:
            # Return best result from best group
            best_result = max(best_group, key=lambda x: x['confidence'])
            engines_used = list(set(r['engine'] for r in best_group))
            return best_result['text'], best_result['confidence'], f"Multi({len(engines_used)})"
        
        # Fallback to highest confidence overall
        best = max(valid_results, key=lambda x: x['confidence'])
        return best['text'], best['confidence'], best['engine']

# -----------------------------------------------------------------------------
# Legacy Multi-OCR Engine class (for compatibility)
# -----------------------------------------------------------------------------
class MultiOCREngine:
    """Multi-OCR Engine with error handling and compatibility fixes"""
    
    def __init__(self):
        print("🔧 Initializing OCR engines...")
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # EasyOCR initialization
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr = easyocr.Reader(['en'], gpu=False)
                self.easyocr_available = True
                print("✅ EasyOCR initialized successfully")
            except Exception as e:
                print(f"⚠️ EasyOCR initialization failed: {e}")
                self.easyocr_available = False
        else:
            self.easyocr_available = False
        
        # Tesseract availability check
        if PYTESSERACT_AVAILABLE:
            try:
                pytesseract.get_tesseract_version()
                self.tesseract_available = True
                print("✅ Tesseract available")
            except Exception as e:
                print(f"⚠️ Tesseract not available: {e}")
                self.tesseract_available = False
        else:
            self.tesseract_available = False
        
        # PaddleOCR initialization
        if PADDLEOCR_AVAILABLE:
            try:
                self.paddleocr = PaddleOCR(lang='en', show_log=False)
                self.paddleocr_available = True
                print("✅ PaddleOCR initialized successfully")
            except Exception as e:
                print(f"⚠️ PaddleOCR initialization failed: {e}")
                self.paddleocr_available = False
        else:
            self.paddleocr_available = False
        
        # Report available engines
        available_engines = []
        if self.easyocr_available: available_engines.append("EasyOCR")
        if self.tesseract_available: available_engines.append("Tesseract")
        if self.paddleocr_available: available_engines.append("PaddleOCR")
        
        print(f"🎯 Available OCR engines: {', '.join(available_engines) if available_engines else 'None'}")
    
    def extract_text_easyocr(self, image):
        """Extract text using EasyOCR"""
        if not self.easyocr_available:
            return "", 0.0
        
        try:
            # Validate image
            if image is None or image.size == 0:
                return "", 0.0
            
            height, width = image.shape[:2]
            if height < 32 or width < 32:
                return "", 0.0
            
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            results = self.easyocr.readtext(image_rgb)
            
            if results:
                texts = []
                confidences = []
                for (bbox, text, confidence) in results:
                    if confidence > 0.3:  # Confidence threshold
                        texts.append(text)
                        confidences.append(confidence)
                
                if texts:
                    combined_text = ' '.join(texts)
                    avg_confidence = np.mean(confidences)
                    return combined_text, avg_confidence
            
            return "", 0.0
            
        except Exception as e:
            print(f"⚠️ EasyOCR error: {e}")
            return "", 0.0
    
    def extract_text_tesseract(self, image):
        """Extract text using Tesseract"""
        if not self.tesseract_available:
            return "", 0.0
        
        try:
            # Validate image
            if image is None or image.size == 0:
                return "", 0.0
            
            height, width = image.shape[:2]
            if height < 10 or width < 10:
                return "", 0.0
            
            # Tesseract config for better results
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-%()$€£¥ '
            
            # Get text with confidence
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            # Filter high confidence text
            texts = []
            confidences = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:
                    text = data['text'][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(int(data['conf'][i]))
            
            if texts:
                combined_text = ' '.join(texts)
                avg_confidence = np.mean(confidences) / 100.0  # Normalize to 0-1
                return combined_text, avg_confidence
            
            return "", 0.0
            
        except Exception as e:
            print(f"⚠️ Tesseract error: {e}")
            return "", 0.0
    
    def extract_text_paddleocr(self, image):
        """Extract text using PaddleOCR"""
        if not self.paddleocr_available:
            return "", 0.0
        
        try:
            # Validate image
            if image is None or image.size == 0:
                return "", 0.0
            
            height, width = image.shape[:2]
            if height < 32 or width < 32:
                return "", 0.0
            
            # Use PaddleOCR API
            results = self.paddleocr.ocr(image)
            
            if results and results[0]:
                texts = []
                confidences = []
                for line in results[0]:
                    if len(line) >= 2:
                        text = line[1][0]  # Extract text
                        confidence = line[1][1]  # Extract confidence
                        if confidence > 0.3:
                            texts.append(text)
                            confidences.append(confidence)
                
                if texts:
                    combined_text = ' '.join(texts)
                    avg_confidence = np.mean(confidences)
                    return combined_text, avg_confidence
            
            return "", 0.0
            
        except Exception as e:
            print(f"⚠️ PaddleOCR error: {e}")
            return "", 0.0
    
    def best_ocr_result(self, image):
        """Get best OCR result from all available engines"""
        results = []
        
        # Try all available engines
        if self.easyocr_available:
            text, conf = self.extract_text_easyocr(image)
            if text.strip():
                results.append((text, conf, "EasyOCR"))
        
        if self.tesseract_available:
            text, conf = self.extract_text_tesseract(image)
            if text.strip():
                results.append((text, conf, "Tesseract"))
        
        if self.paddleocr_available:
            text, conf = self.extract_text_paddleocr(image)
            if text.strip():
                results.append((text, conf, "PaddleOCR"))
        
        # Return best result by confidence
        if results:
            best_result = max(results, key=lambda x: x[1])
            return best_result
        
        return "", 0.0, "None"
    
    def enhance_image(self, image):
        """Enhance image for better OCR results"""
        try:
            # Validate input
            if image is None or image.size == 0:
                return image
            
            if len(image.shape) < 2:
                return image
                
            height, width = image.shape[:2]
            if height < 3 or width < 3:
                return image
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Noise reduction with adaptive kernel size
            kernel_size = max(3, min(5, min(height, width) // 20))
            if kernel_size % 2 == 0:
                kernel_size += 1
            denoised = cv2.medianBlur(gray, kernel_size)
            
            # Contrast enhancement
            enhanced = cv2.convertScaleAbs(denoised, alpha=1.2, beta=10)
            
            # Sharpen only if image is large enough
            if height >= 5 and width >= 5:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(enhanced, -1, kernel)
                return sharpened
            else:
                return enhanced
                
        except Exception as e:
            print(f"⚠️ Image enhancement error: {e}")
            return image

# Global OCR engine instance
multi_ocr_engine = None

# Initialize Multi-OCR Engine
def initialize_ocr_engine():
    """Initialize the EasyOCR Engine (Optimized for Cell Segmentation)"""
    global multi_ocr_engine
    if multi_ocr_engine is None:
        try:
            multi_ocr_engine = EasyOCREngine()
            print("🚀 EasyOCR Engine initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to initialize EasyOCR Engine: {e}")
            # Fallback to legacy MultiOCREngine
            try:
                multi_ocr_engine = MultiOCREngine()
                print("🔄 Fallback: Legacy Multi-OCR Engine initialized")
            except Exception as e2:
                print(f"⚠️ Complete OCR initialization failure: {e2}")
                multi_ocr_engine = None

# Model that detects individual cells within a table for text extraction
try:
    cell_model_path = MODEL_DIR / 'best(cell).pt'
    print(f"Loading cell model from: {cell_model_path}")
    print(f"Model file exists: {cell_model_path.exists()}")
    cell_model = YOLO(str(cell_model_path))
    print("Cell model loaded successfully")
except Exception as e:
    print(f"Error loading cell model: {e}")
    cell_model = None

# Initialize EasyOCR reader with enhanced compatibility handling
ocr_reader = None
EASYOCR_DISABLED = False

def init_easyocr_safe():
    """Safe EasyOCR initialization with multiple fallback strategies"""
    global ocr_reader, EASYOCR_DISABLED
    
    # Check if EasyOCR import failed
    if easyocr is None or EASYOCR_IMPORT_ERROR:
        print(f"⚠️  EasyOCR import failed: {EASYOCR_IMPORT_ERROR}")
        print("📝 Text extraction will use PyMuPDF + LLM only")
        EASYOCR_DISABLED = True
        ocr_reader = None
        return False
    
    try:
        print("Initializing EasyOCR reader...")
        
        # Strategy 1: Try with GPU if available
        try:
            ocr_reader = easyocr.Reader(['vi', 'en'], gpu=torch.cuda.is_available())
            print(f"EasyOCR reader initialized successfully (GPU: {torch.cuda.is_available()})")
            return True
        except Exception as gpu_error:
            print(f"GPU initialization failed: {gpu_error}")
            
        # Strategy 2: Try CPU only
        try:
            print("Trying CPU-only initialization...")
            ocr_reader = easyocr.Reader(['vi', 'en'], gpu=False)
            print("EasyOCR reader initialized successfully (CPU only)")
            return True
        except Exception as cpu_error:
            print(f"CPU initialization failed: {cpu_error}")
            
        # Strategy 3: Try with minimal languages
        try:
            print("Trying minimal language set...")
            ocr_reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR reader initialized with English only")
            return True
        except Exception as min_error:
            print(f"Minimal initialization failed: {min_error}")
            
    except Exception as e:
        print(f"EasyOCR initialization completely failed: {e}")
    
    # All strategies failed
    print("⚠️  EasyOCR disabled due to compatibility issues")
    print("📝 Text extraction will use PyMuPDF + LLM only")
    EASYOCR_DISABLED = True
    ocr_reader = None
    return False

# Initialize LLM clients FIRST
# Get API keys from environment variables
# Note: Use environment variable or set to None to disable LLM processing
GROQ_API_KEY = os.getenv('GROQ_API_KEY', None)  # Remove invalid key - use env var
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', None)

# Initialize Groq client (using the API key from screenshot)
try:
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq LLM client initialized successfully")
    else:
        groq_client = None
        print("No Groq API key found")
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    groq_client = None

# Initialize OpenAI client
try:
    if OPENAI_API_KEY:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized successfully")
    else:
        openai_client = None
        print("No OpenAI API key found")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    openai_client = None

# Try to initialize EasyOCR
easyocr_status = init_easyocr_safe()

def get_extraction_capabilities():
    """Return current text extraction capabilities"""
    capabilities = {
        "pymupdf": True,  # Always available
        "easyocr": ocr_reader is not None and not EASYOCR_DISABLED,
        "llm_groq": groq_client is not None,
        "llm_openai": openai_client is not None,
        "basic_cleaning": True  # Always available fallback
    }
    
    methods = []
    if capabilities["pymupdf"]:
        methods.append("PyMuPDF (text-based PDFs)")
    if capabilities["easyocr"]:
        methods.append("EasyOCR (image-based content)")
    else:
        methods.append("EasyOCR (disabled - compatibility issues)")
    
    # LLM processing status
    if capabilities["llm_groq"]:
        methods.append("LLM text processing (Groq)")
    elif capabilities["llm_openai"]:
        methods.append("LLM text processing (OpenAI)")
    else:
        methods.append("Basic text cleaning (LLM disabled)")
    
    return capabilities, methods

# Print current capabilities
capabilities, methods = get_extraction_capabilities()
print(f"\n📋 Text Extraction Methods Available:")
for i, method in enumerate(methods, 1):
    status = "✅" if ("disabled" not in method) else "⚠️"
    print(f"  {i}. {status} {method}")

if not capabilities["easyocr"]:
    reason = EASYOCR_IMPORT_ERROR if EASYOCR_IMPORT_ERROR else "compatibility issues"
    print(f"\n💡 EasyOCR Compatibility Info:")
    print(f"  - EasyOCR disabled due to: {reason}")
    print(f"  - System will use PyMuPDF + LLM for text extraction")
    print(f"  - This provides good results for most PDF types")
    print(f"  - For image-heavy PDFs, consider fixing numpy compatibility")

print(f"\n🔗 Total extraction pipeline: {len([m for m in methods if 'disabled' not in m])} active methods")
print(f"═" * 60)

def process_pdf(file_path):
    """
    Main function to process the PDF file.
    It extracts tables from each page, recognizes cells, and extracts text.
    """
    print(f"Processing PDF file: {file_path}")
<<<<<<< HEAD
=======
    
    # Initialize Multi-OCR Engine if not already done
    initialize_ocr_engine()
    
    doc = fitz.open(file_path)
    all_tables_data = []
>>>>>>> 1194ca1ccaa9cbe5704d8ca19ef3e361f66ba32e
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Open PDF document
        doc = fitz.open(file_path)
        all_tables_data = []
        
        print(f"PDF has {len(doc)} pages")
        
        if len(doc) == 0:
            print("Warning: PDF has no pages")
            return []
            
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        raise e

    for page_num in range(len(doc)):
        print(f"\n=== Processing page {page_num + 1} ===")
        page = doc.load_page(page_num)
        
        # First, try to get all text from the page to check if it's text-based
        page_text = page.get_text()
        print(f"Page text length: {len(page_text)} characters")
        if page_text:
            print(f"Sample text: '{page_text[:200]}...'")
        
        # Convert PDF page to image
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        print(f"Page image shape: {img_cv.shape}")

        # Detect tables on the page
        table_detection_info = detect_tables_with_info(img_cv)
        table_boxes = table_detection_info['boxes']
        print(f"Found {len(table_boxes)} table(s) on page {page_num + 1}")

        # If no tables detected by AI model, try fallback method
        if len(table_boxes) == 0:
            print("No tables detected by AI model, trying fallback extraction...")
            fallback_data = extract_text_fallback(page)
            if fallback_data:
                all_tables_data.append({
                    "page": page_num + 1,
                    "table": 1,
                    "data": fallback_data,
                    "method": "fallback",
                    "table_detection": {
                        "status": "fallback",
                        "confidence": 0.0,
                        "bbox": None
                    },
                    "structure_detection": {
                        "rows_detected": len(fallback_data),
                        "cols_detected": len(fallback_data[0]) if fallback_data else 0,
                        "method": "text_parsing"
                    }
                })
            continue

        for i, (table_box, confidence) in enumerate(zip(table_boxes, table_detection_info['confidences'])):
            print(f"Processing table {i+1} on page {page_num+1}")
            print(f"Table box coordinates: {table_box}")
            
            # Validate table box coordinates
            if len(table_box) < 4:
                print(f"Invalid table box: {table_box}")
                continue
                
            # Ensure coordinates are within image bounds
            x1, y1, x2, y2 = table_box
            x1 = max(0, x1)
            y1 = max(0, y1) 
            x2 = min(img_cv.shape[1], x2)
            y2 = min(img_cv.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                print(f"Invalid cropping coordinates: ({x1},{y1}) to ({x2},{y2})")
                continue
            
            # Crop the table from the page image
            table_img = img_cv[y1:y2, x1:x2]
            print(f"Cropped table image shape: {table_img.shape}")

            # Phase 2: Detect individual cells using cell segmentation model
            cell_info = detect_cells_with_info(table_img)
            cells = cell_info['cells']

            # Phase 3: Extract text from detected cells using Advanced OCR Pipeline
            if cells and len(cells) > 0:
                print(f"Using Advanced OCR Pipeline with {len(cells)} detected cells")
                
                # Use Advanced OCR Pipeline for enhanced text extraction
                try:
                    from advanced_ocr_pipeline import process_table_with_advanced_ocr
                    
                    # Prepare cell detections for pipeline
                    cell_detections = {'cells': cells}
                    table_info = {
                        'page': page_num + 1,
                        'table_index': i + 1,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence)
                    }
                    
                    # Process with Advanced OCR Pipeline
                    pipeline_results = process_table_with_advanced_ocr(
                        table_img, 
                        cell_detections, 
                        table_info,
                        save_results=False  # Don't save intermediate files for API
                    )
                    
                    # Extract structured data for API response
                    structured_table = pipeline_results.get('structured_table', {})
                    table_matrix = structured_table.get('table_matrix', [])
                    
                    if table_matrix:
                        table_data = table_matrix
                        print(f"✅ Advanced OCR Pipeline extracted {len(table_matrix)} rows")
                    else:
                        # Fallback to basic extraction if pipeline fails
                        print("⚠️ Pipeline returned no matrix, using fallback")
                        table_data = build_table_from_cells(page, [x1, y1, x2, y2], cells, cell_info)
                    
                    # Add pipeline metadata
                    pipeline_metadata = {
                        'pipeline_used': True,
                        'processing_quality': pipeline_results.get('final_summary', {}).get('processing_quality', {}),
                        'ocr_engines_used': pipeline_results.get('final_summary', {}).get('ocr_engines_used', []),
                        'cells_processed': pipeline_results.get('final_summary', {}).get('total_cells_detected', 0),
                        'cells_with_text': pipeline_results.get('final_summary', {}).get('cells_with_text', 0)
                    }
                    
                except Exception as e:
                    print(f"⚠️ Advanced OCR Pipeline failed: {e}")
                    print("🔄 Falling back to basic cell extraction...")
                    table_data = build_table_from_cells(page, [x1, y1, x2, y2], cells, cell_info)
                    pipeline_metadata = {'pipeline_used': False, 'error': str(e)}
                
            else:
                print("No cells detected, creating empty table")
                table_data = []
                pipeline_metadata = {'pipeline_used': False, 'reason': 'no_cells_detected'}

            # Prepare the table data entry
            table_entry = {
                "page": page_num + 1,
                "table": i + 1,
                "data": table_data,
                "method": "ai_model",
                "table_detection": {
                    "status": "detected",
                    "confidence": float(confidence),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "model_used": "best(table).pt"
                }
            }
            
            # Add cell detection info with pipeline metadata
            table_entry["cell_detection"] = {
                "cells_detected": len(cells) if cells else 0,
                "cells_confidence": [float(conf) for conf in cell_info['confidences']] if cells and cell_info else [],
                "method": "advanced_ocr_pipeline" if pipeline_metadata.get('pipeline_used') else "ai_model", 
                "model_used": "best(cell).pt",
<<<<<<< HEAD
                "extraction_method": "enhanced_multi_method"
=======
                "extraction_method": "enhanced_cell_based" if pipeline_metadata.get('pipeline_used') else "cell_based"
>>>>>>> 1194ca1ccaa9cbe5704d8ca19ef3e361f66ba32e
            }
            
            # Add Advanced OCR Pipeline metadata
            table_entry["advanced_ocr_pipeline"] = pipeline_metadata
            
            # Always add visualizations
            table_entry["visualizations"] = save_table_visualization(img_cv, [x1, y1, x2, y2], cells if cells else [], page_num + 1, i + 1)
            
            all_tables_data.append(table_entry)
    
    # Close the document
    doc.close()
    print(f"Completed processing PDF. Found {len(all_tables_data)} tables total.")
    return all_tables_data

def detect_tables(image):
    """
    Detects tables in an image using the table detection model.
    """
    if table_model is None:
        print("Table model not loaded, cannot detect tables")
        return np.array([])
        
    results = table_model(image)
    
    # Debug: Print detection results
    print(f"Table detection results: {len(results)} result(s)")
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        print(f"Found {len(boxes)} table(s) with confidences: {confidences}")
        print(f"Table boxes: {boxes}")
        
        # No confidence filtering - accept all detections
        print(f"Accepting all {len(boxes)} table detection(s) without threshold filtering")
        
        return boxes
    else:
        print("No tables detected or no boxes found")
        return np.array([])

def detect_tables_with_info(image):
    """
    Detects tables in an image and returns detailed info including confidences.
    """
    if table_model is None:
        print("Table model not loaded, cannot detect tables")
        return {'boxes': np.array([]), 'confidences': np.array([]), 'model': None}
        
    results = table_model(image)
    
    # Debug: Print detection results
    print(f"Table detection results: {len(results)} result(s)")
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        print(f"Found {len(boxes)} table(s) with confidences: {confidences}")
        print(f"Table boxes: {boxes}")
        
        # No confidence filtering - accept all detections
        print(f"Accepting all {len(boxes)} table detection(s) without threshold filtering")
        
        return {
            'boxes': boxes,
            'confidences': confidences,
            'model': 'best(table).pt',
            'threshold': 0.0  # No threshold applied
        }
    else:
        print("No tables detected or no boxes found")
        return {'boxes': np.array([]), 'confidences': np.array([]), 'model': 'best(table).pt'}

def detect_table_structure(table_image):
    """
    Detects rows and columns in a cropped table image.
    """
    print(f"Detecting structure in table image of shape: {table_image.shape}")
    
    if structure_model is None:
        print("Structure model not loaded, cannot detect table structure")
        return [], []
        
    results = structure_model(table_image)
    
    # Debug: Print structure detection results
    print(f"Structure detection results: {len(results)} result(s)")
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        class_names = results[0].names
        
        print(f"Class names available: {class_names}")
        print(f"Found {len(boxes)} structure elements")
        print(f"Classes: {classes}")
        print(f"Confidences: {confidences}")

        rows = []
        cols = []
        
        # Look for class names containing 'row' and 'column' (case insensitive)
        row_class_id = -1
        col_class_id = -1
        for k, v in class_names.items():
            v_lower = v.lower()
            if 'row' in v_lower:
                row_class_id = k
                print(f"Found row class: {k} -> {v}")
            if 'col' in v_lower:  # Also check for 'col' in case it's abbreviated
                col_class_id = k
                print(f"Found column class: {k} -> {v}")

        print(f"Row class ID: {row_class_id}, Column class ID: {col_class_id}")

        # No confidence filtering - accept all structure detections
        for box, cls, conf in zip(boxes, classes, confidences):
                if cls == row_class_id:
                    rows.append(box)
                    print(f"Added row: {box} (conf: {conf:.3f})")
                elif cls == col_class_id:
                    cols.append(box)
                    print(f"Added column: {box} (conf: {conf:.3f})")

        # Sort rows by top coordinate, and columns by left coordinate
        rows = sorted(rows, key=lambda b: b[1])
        cols = sorted(cols, key=lambda b: b[0])
        
        print(f"Final: {len(rows)} rows, {len(cols)} columns")
        return rows, cols
    else:
        print("No structure detected or no boxes found")
        return [], []

def detect_table_structure_with_info(table_image):
    """
    DEPRECATED: Structure detection removed from 3-phase workflow
    """
    print("Structure detection disabled - using cell-based approach only")
    return {
        'rows': [], 'cols': [], 
        'row_confidences': [], 'col_confidences': [],
        'model': None, 'threshold': 0.0
    }

def detect_cells_with_info(table_image):
    """
    Detects individual cells in a cropped table image using the cell detection model.
    This provides more precise cell boundaries for text extraction.
    """
    print(f"Detecting cells in table image of shape: {table_image.shape}")
    
    if cell_model is None:
        print("Cell model not loaded, cannot detect cells")
        return {
            'cells': [], 'confidences': [], 
            'model': None, 'threshold': 0.0
        }
        
    results = cell_model(table_image)
    
    # Debug: Print cell detection results
    print(f"Cell detection results: {len(results)} result(s)")
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        class_names = results[0].names if hasattr(results[0], 'names') else {}
        
        print(f"Class names available: {class_names}")
        print(f"Found {len(boxes)} cell candidates")
        print(f"Confidences: {confidences}")

        cells = []
        cell_confidences = []
        
        # No confidence filtering - accept all cell detections
        for box, conf in zip(boxes, confidences):
            cells.append(box)
            cell_confidences.append(float(conf))
            print(f"Added cell: {box} (conf: {conf:.3f})")

        # Sort cells by position (top to bottom, left to right)
        if cells:
            cell_indices = sorted(range(len(cells)), key=lambda i: (cells[i][1], cells[i][0]))
            cells = [cells[i] for i in cell_indices]
            cell_confidences = [cell_confidences[i] for i in cell_indices]
        
        print(f"Final: {len(cells)} cells detected")
        
        return {
            'cells': cells,
            'confidences': cell_confidences,
            'model': 'best(cell).pt',
            'threshold': 0.0,  # No threshold applied
            'class_names': class_names
        }
    else:
        print("No cells detected or no boxes found")
        return {
            'cells': [], 'confidences': [], 
            'model': 'best(cell).pt', 'threshold': 0.0
        }

def extract_text_from_cell_region(page, table_box, cell_box):
    """
    Extract text from a specific cell region using Multi-OCR Engine.
    """
    # Initialize OCR engine if not already done
    global multi_ocr_engine
    if multi_ocr_engine is None:
        initialize_ocr_engine()
    
    # Convert cell coordinates to page coordinates
    x1, y1, x2, y2 = table_box
    cx1, cy1, cx2, cy2 = cell_box
    
    # Map cell coordinates to page coordinates
    page_x1 = x1 + cx1
    page_y1 = y1 + cy1
    page_x2 = x1 + cx2
    page_y2 = y1 + cy2
    
    # Create rectangle for text extraction
    rect = fitz.Rect(page_x1, page_y1, page_x2, page_y2)
    
    try:
        # First try PyMuPDF text extraction (fastest)
        text = page.get_text("text", clip=rect).strip()
        if text:
            print(f"Extracted text (PyMuPDF) from cell [{cx1},{cy1},{cx2},{cy2}]: '{text}'")
            return text
        
        # If no text found, try OCR on the image region
        if multi_ocr_engine is not None:
            try:
                # Validate and clip rectangle to page bounds
                page_rect = page.rect
                clipped_x1 = max(0, min(page_x1, page_rect.width))
                clipped_y1 = max(0, min(page_y1, page_rect.height))
                clipped_x2 = max(clipped_x1 + 1, min(page_x2, page_rect.width))
                clipped_y2 = max(clipped_y1 + 1, min(page_y2, page_rect.height))
                
                # Ensure minimum size for valid pixmap
                min_width = 10
                min_height = 10
                
                if (clipped_x2 - clipped_x1) < min_width:
                    clipped_x2 = min(clipped_x1 + min_width, page_rect.width)
                if (clipped_y2 - clipped_y1) < min_height:
                    clipped_y2 = min(clipped_y1 + min_height, page_rect.height)
                
                # Final check if clipped rectangle is valid
                if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
                    print(f"⚠️ Invalid cell bounds after clipping [{clipped_x1},{clipped_y1},{clipped_x2},{clipped_y2}]")
                    return ""
                
                # Create clipped rectangle
                clipped_rect = fitz.Rect(clipped_x1, clipped_y1, clipped_x2, clipped_y2)
                
                # Get page as image with clipped rectangle
                pix = page.get_pixmap(dpi=200, clip=clipped_rect)
                
                if pix.width <= 0 or pix.height <= 0:
                    print(f"⚠️ Invalid pixmap size: {pix.width}x{pix.height}")
                    return ""
                
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # Check minimum image size for OCR
                if img_cv.shape[0] >= 10 and img_cv.shape[1] >= 10:
                    # Enhance image for better OCR
                    enhanced_img = multi_ocr_engine.enhance_image(img_cv)
                    
                    # Get best OCR result
                    text, confidence, engine = multi_ocr_engine.best_ocr_result(enhanced_img)
                    
                    if text.strip():
                        print(f"Extracted text ({engine}, conf: {confidence:.3f}) from cell [{cx1},{cy1},{cx2},{cy2}]: '{text}'")
                        return text.strip()
                else:
                    print(f"⚠️ Image too small for OCR: {img_cv.shape[1]}x{img_cv.shape[0]}")
                    
            except Exception as ocr_error:
                print(f"⚠️ OCR processing error for cell [{cx1},{cy1},{cx2},{cy2}]: {ocr_error}")
                return ""
        
        print(f"No text found in cell [{cx1},{cy1},{cx2},{cy2}]")
        return ""
        
    except Exception as e:
        print(f"Error extracting text from cell: {e}")
        return ""

def build_table_from_cells(page, table_box, cells, cell_info):
    """
    Build table data from individual cell detections.
    This provides more accurate text extraction by using exact cell boundaries.
    """
    print(f"Building table from {len(cells)} detected cells")
    
    if not cells:
        print("No cells provided, returning empty table")
        return []
    
    # Extract text from each detected cell using enhanced method
    cell_data = []
    for i, cell_box in enumerate(cells):
        confidence = cell_info['confidences'][i] if i < len(cell_info['confidences']) else 0.0
        text = extract_text_enhanced(page, table_box, cell_box)
        
        cell_data.append({
            'cell_id': i + 1,
            'bbox': [int(x) for x in cell_box.tolist()],  # Convert to Python ints
            'text': text,
            'confidence': float(confidence),  # Convert to Python float
            'extraction_method': 'enhanced_multi_method'
        })
    
    # Try to organize cells into a grid structure
    # Sort cells by position to estimate rows and columns
    sorted_cells = sorted(cell_data, key=lambda c: (c['bbox'][1], c['bbox'][0]))  # Sort by y, then x
    
    # Group cells into approximate rows based on y-coordinate similarity
    if sorted_cells:
        rows = []
        current_row = [sorted_cells[0]]
        current_y = sorted_cells[0]['bbox'][1]
        y_tolerance = 10  # Pixels tolerance for same row
        
        for cell in sorted_cells[1:]:
            cell_y = cell['bbox'][1]
            if abs(cell_y - current_y) <= y_tolerance:
                # Same row
                current_row.append(cell)
            else:
                # New row
                current_row.sort(key=lambda c: c['bbox'][0])  # Sort by x within row
                rows.append([cell['text'] for cell in current_row])
                current_row = [cell]
                current_y = cell_y
        
        # Add the last row
        if current_row:
            current_row.sort(key=lambda c: c['bbox'][0])
            rows.append([cell['text'] for cell in current_row])
        
        print(f"Organized into {len(rows)} rows")
        return rows
    
    # Fallback: return all cell texts as a single column
    return [[cell['text']] for cell in sorted_cells]

def build_table_from_structure(page, table_box, rows, cols):
    """
    DEPRECATED: Structure-based extraction removed from 3-phase workflow
    """
    print("Structure-based extraction disabled - should not be called")
    return []

def extract_text_fallback(page):
    """
    Fallback method to extract text when AI models fail.
    Uses PyMuPDF's built-in table detection.
    """
    print("Using fallback text extraction method...")
    
    try:
        # Method 1: Try PyMuPDF's find_tables
        tables = page.find_tables()
        print(f"PyMuPDF found {len(tables)} table(s)")
        
        if tables:
            table = tables[0]  # Take the first table
            table_data = table.extract()
            print(f"Extracted table with {len(table_data)} rows")
            return table_data
            
    except Exception as e:
        print(f"PyMuPDF table extraction failed: {e}")
    
    try:
        # Method 2: Extract all text and try to parse it
        text = page.get_text()
        if not text.strip():
            print("No text found on page")
            return None
            
        # Split into lines and try to detect table-like structure
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Look for lines that might be table rows (containing multiple words/numbers)
        table_lines = []
        for line in lines:
            # Simple heuristic: if line has multiple words separated by spaces/tabs
            parts = line.split()
            if len(parts) >= 2:  # At least 2 columns
                table_lines.append(parts)
        
        if table_lines:
            print(f"Extracted {len(table_lines)} potential table rows from text")
            return table_lines
            
    except Exception as e:
        print(f"Text parsing fallback failed: {e}")
    
    return None

def extract_text_easyocr(cell_image):
    """
    Extract text from a cell image using EasyOCR with enhanced error handling.
    """
    if ocr_reader is None:
        print("EasyOCR reader not initialized")
        return ""
    
    try:
        # Validate input image
        if cell_image is None or cell_image.size == 0:
            print("Invalid or empty cell image")
            return ""
        
        # Check image dimensions
        if len(cell_image.shape) < 2:
            print("Invalid image dimensions")
            return ""
        
        # Ensure image is in the right format
        if len(cell_image.shape) == 3 and cell_image.shape[2] == 3:
            # Convert BGR to RGB for EasyOCR
            cell_image_rgb = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)
        elif len(cell_image.shape) == 3 and cell_image.shape[2] == 4:
            # Convert BGRA to RGB
            cell_image_rgb = cv2.cvtColor(cell_image, cv2.COLOR_BGRA2RGB)
        else:
            # Assume it's already in correct format or grayscale
            cell_image_rgb = cell_image
        
        # Ensure image is not too small
        height, width = cell_image_rgb.shape[:2]
        if height < 10 or width < 10:
            print(f"Image too small for OCR: {width}x{height}")
            return ""
        
        # Resize if image is too large (optimize for OCR)
        max_dimension = 1000
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            cell_image_rgb = cv2.resize(cell_image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Resized image for OCR: {width}x{height} -> {new_width}x{new_height}")
        
        # Perform OCR with timeout protection
        print(f"Running EasyOCR on image: {cell_image_rgb.shape}")
        results = ocr_reader.readtext(cell_image_rgb)
        
        # Extract text from results
        text_parts = []
        max_confidence = 0
        for detection in results:
            try:
                bbox, text, confidence = detection
                if confidence > 0.3:  # Filter low confidence results
                    clean_text = text.strip()
                    if clean_text:  # Only add non-empty text
                        text_parts.append(clean_text)
                        max_confidence = max(max_confidence, confidence)
            except Exception as det_error:
                print(f"Error processing OCR detection: {det_error}")
                continue
        
        extracted_text = ' '.join(text_parts)
        
        if extracted_text:
            print(f"EasyOCR extracted: '{extracted_text}' (best confidence: {max_confidence:.2f})")
            return extracted_text
        else:
            print("EasyOCR: No text found with sufficient confidence")
            return ""
            
    except Exception as e:
        print(f"EasyOCR extraction error: {e}")
        import traceback
        print(f"EasyOCR traceback: {traceback.format_exc()}")
        return ""

def process_text_with_llm(raw_text, context="table_cell"):
    """
    Process extracted text using LLM to clean and structure it.
    Falls back gracefully if LLM processing fails.
    """
    if not raw_text or not raw_text.strip():
        return ""
    
    # Basic text cleaning (fallback method)
    def basic_text_cleaning(text):
        """Basic text cleaning without LLM"""
        # Remove excessive whitespace and newlines
        cleaned = ' '.join(text.split())
        # Remove common OCR artifacts
        cleaned = cleaned.replace('\\n', ' ').replace('\\t', ' ')
        # Strip quotes if present
        cleaned = cleaned.strip('"\'')
        return cleaned if cleaned else text
    
    # Try Groq first (if available and configured)
    if groq_client:
        try:
            prompt = f"""
Bạn là một AI chuyên xử lý dữ liệu bảng. Nhiệm vụ của bạn là làm sạch và cải thiện text được trích xuất từ một cell trong bảng.

Text gốc: "{raw_text}"
Context: {context}

Hãy:
1. Sửa lỗi chính tả và OCR
2. Chuẩn hóa format (số, ngày tháng, v.v.)
3. Loại bỏ ký tự không cần thiết
4. Giữ nguyên ý nghĩa của dữ liệu

Chỉ trả về text đã được làm sạch, không giải thích:
"""
            
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            cleaned_text = response.choices[0].message.content.strip()
            if cleaned_text and cleaned_text != raw_text:
                print(f"LLM processed: '{raw_text}' -> '{cleaned_text}'")
                return cleaned_text
            else:
                return raw_text
                
        except Exception as e:
            print(f"⚠️  Groq LLM processing error: {e}")
            print(f"🔄 Falling back to basic text cleaning...")
    
    # Fallback to OpenAI if available
    if openai_client:
        try:
            prompt = f"""
Clean and improve this text extracted from a table cell.
Fix OCR errors, standardize format, remove unnecessary characters.
Keep the original meaning.

Raw text: "{raw_text}"

Return only the cleaned text:
"""
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )
            
            cleaned_text = response.choices[0].message.content.strip()
            if cleaned_text and cleaned_text != raw_text:
                print(f"OpenAI processed: '{raw_text}' -> '{cleaned_text}'")
                return cleaned_text
            else:
                return raw_text
                
        except Exception as e:
            print(f"⚠️  OpenAI LLM processing error: {e}")
            print(f"🔄 Falling back to basic text cleaning...")
    
    # Fallback to basic text cleaning
    cleaned_text = basic_text_cleaning(raw_text)
    if cleaned_text != raw_text:
        print(f"Basic cleaning: '{raw_text}' -> '{cleaned_text}'")
    
    return cleaned_text

def extract_text_enhanced(page, table_box, cell_box):
    """
    Enhanced text extraction combining multiple methods with LLM processing.
    """
    try:
        # Method 1: Try PyMuPDF first (fastest for text-based PDFs)
        print(f"Trying PyMuPDF extraction for cell {cell_box}")
        pymupdf_text = extract_text_from_cell_region(page, table_box, cell_box)
        
        # If PyMuPDF found substantial text, use it
        if pymupdf_text and len(pymupdf_text.strip()) > 2:
            # Process with LLM for cleaning
            processed_text = process_text_with_llm(pymupdf_text, "pdf_text")
            print(f"Using PyMuPDF extraction: '{processed_text}'")
            return processed_text
        
        # Method 2: Fall back to EasyOCR for image-based content (if available)
        if ocr_reader is not None and not EASYOCR_DISABLED:
            try:
                print(f"Trying EasyOCR extraction for cell {cell_box}")
                
                # Get cell coordinates
                x1, y1, x2, y2 = table_box
                cx1, cy1, cx2, cy2 = cell_box
                
                # Convert PDF page to image for OCR
                try:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                except Exception as img_error:
                    print(f"Error converting PDF page to image: {img_error}")
                    raise img_error
                
                # Crop cell from page image
                page_x1 = x1 + cx1
                page_y1 = y1 + cy1
                page_x2 = x1 + cx2
                page_y2 = y1 + cy2
                
                # Ensure coordinates are within bounds
                page_x1 = max(0, int(page_x1))
                page_y1 = max(0, int(page_y1))
                page_x2 = min(img_cv.shape[1], int(page_x2))
                page_y2 = min(img_cv.shape[0], int(page_y2))
                
                if page_x2 > page_x1 and page_y2 > page_y1:
                    cell_img = img_cv[page_y1:page_y2, page_x1:page_x2]
                    print(f"Cell image shape: {cell_img.shape}")
                    
                    # Use EasyOCR on cell image
                    ocr_text = extract_text_easyocr(cell_img)
                    
                    if ocr_text and len(ocr_text.strip()) > 0:
                        # Process with LLM for cleaning
                        processed_text = process_text_with_llm(ocr_text, "ocr_text")
                        print(f"Using EasyOCR extraction: '{processed_text}'")
                        return processed_text
                else:
                    print(f"Invalid cell coordinates: ({page_x1},{page_y1}) to ({page_x2},{page_y2})")
                        
            except Exception as ocr_error:
                print(f"EasyOCR fallback failed: {ocr_error}")
                import traceback
                print(f"EasyOCR traceback: {traceback.format_exc()}")
        else:
            if EASYOCR_DISABLED:
                print("EasyOCR disabled due to compatibility issues - using PyMuPDF + LLM only")
            else:
                print("EasyOCR reader not available")
        
        # Method 3: Return PyMuPDF result even if short, or empty
        if pymupdf_text and pymupdf_text.strip():
            processed_text = process_text_with_llm(pymupdf_text, "pdf_text")
            print(f"Using PyMuPDF fallback: '{processed_text}'")
            return processed_text
        
        print("No text extraction method succeeded for this cell")
        return ""
        
    except Exception as e:
        print(f"Enhanced text extraction failed: {e}")
        import traceback
        print(f"Enhanced extraction traceback: {traceback.format_exc()}")
        return ""

# This function is no longer used with the new structure detection logic.
# def extract_text_from_cells(page, table_box, cell_boxes):
#     """
#     Extracts text from the detected cells using the PDF's text data.
#     """
#     # Unpack the numpy array into individual arguments for fitz.Rect
#     table_rect = fitz.Rect(*table_box)
#
#     # Sort cells by row and then by column
#     # This is a simplified sorting, might need to be more robust
#     cell_boxes = sorted(cell_boxes, key=lambda b: (b[1], b[0]))
#
#     cell_data = []
#     for cell_box in cell_boxes:
#         # The cell coordinates are relative to the table image,
#         # so we need to convert them to page coordinates.
#         cell_rect_on_page = fitz.Rect(
#             table_box[0] + cell_box[0],
#             table_box[1] + cell_box[1],
#             table_box[0] + cell_box[2],
#             table_box[1] + cell_box[3]
#         )
#
#         # Get text from the cell area
#         text = page.get_text("text", clip=cell_rect_on_page).strip()
#         cell_data.append(text)
#
#     # Here you might want to structure the data into rows and columns
#     # For now, we return a flat list of cell texts
#     return cell_data 

def save_table_visualization(page_image, table_bbox, cells, page_num, table_num):
    """
    Create and save visualizations of table detection and cell segmentation.
    Returns base64 encoded images for frontend display.
    """
    print(f"Creating visualization for page {page_num}, table {table_num}")
    print(f"Table bbox: {table_bbox}")
    print(f"Number of cells: {len(cells)}")
    
    try:
        x1, y1, x2, y2 = table_bbox
        
        # Create table detection visualization
        table_vis = page_image.copy()
        cv2.rectangle(table_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(table_vis, f"Table {table_num}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"Created table detection visualization with shape: {table_vis.shape}")
        
        # Create cell segmentation visualization  
        table_img = page_image[y1:y2, x1:x2].copy()
        cell_vis = table_img.copy()
        
        # Draw cell boundaries
        for i, cell in enumerate(cells):
            cx1, cy1, cx2, cy2 = cell
            cv2.rectangle(cell_vis, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
            cv2.putText(cell_vis, f"{i+1}", (cx1+5, cy1+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        print(f"Created cell segmentation visualization with shape: {cell_vis.shape}")
        
        # Convert to base64 for JSON response with compression
        def img_to_base64(img):
            # Resize image to reduce size if too large
            height, width = img.shape[:2]
            max_size = 800  # Maximum width or height
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            # Use JPEG compression to reduce file size
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # 85% quality
            _, buffer = cv2.imencode('.jpg', img, encode_param)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
        
        table_detection_b64 = img_to_base64(table_vis)
        cell_segmentation_b64 = img_to_base64(cell_vis)
        
        print(f"Table detection image base64 length: {len(table_detection_b64)}")
        print(f"Cell segmentation image base64 length: {len(cell_segmentation_b64)}")
        
        result = {
            "table_detection_image": table_detection_b64,
            "cell_segmentation_image": cell_segmentation_b64,
            "table_bbox": [int(x) for x in table_bbox],  # Convert numpy int32 to Python int
            "cells_count": len(cells)
        }
        
        print(f"Successfully created visualization data for page {page_num}, table {table_num}")
        return result
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        # Return empty visualization data if error occurs
        return {
            "table_detection_image": None,
            "cell_segmentation_image": None,
            "table_bbox": [int(x) for x in table_bbox] if table_bbox is not None else None,
            "cells_count": len(cells),
            "error": str(e)
        }

def enhanced_cell_cropping_and_ocr(detections, original_image, save_enhanced_crops=True):
    """Enhanced cell processing with advanced OCR and padding"""
    
    print("✂️ Enhanced Cell Cropping & OCR Processing...")
    
    # Get or initialize OCR engine  
    if multi_ocr_engine is None:
        initialize_ocr_engine()
    
    if multi_ocr_engine is None:
        print("⚠️ No OCR engine available")
        return []
    
    cells = detections.get('cells', [])
    print(f"🔍 Enhanced processing {len(cells)} cells...")
    
    enhanced_results = []
    
    for i, cell_box in enumerate(cells):
        x1, y1, x2, y2 = cell_box
        
        # Enhanced padding (larger for better OCR)
        padding_x = int((x2 - x1) * 0.05)  # 5% horizontal padding
        padding_y = int((y2 - y1) * 0.05)  # 5% vertical padding
        
        # Apply padding with bounds checking
        padded_x1 = max(0, x1 - padding_x)
        padded_y1 = max(0, y1 - padding_y)
        padded_x2 = min(original_image.shape[1], x2 + padding_x)
        padded_y2 = min(original_image.shape[0], y2 + padding_y)
        
        # Crop with enhanced padding
        cell_image = original_image[padded_y1:padded_y2, padded_x1:padded_x2]
        
        if cell_image.size == 0:
            continue
            
        # Enhanced OCR processing using AdvancedOCREngine method
        if hasattr(multi_ocr_engine, 'extract_text_comprehensive'):
            # Use advanced comprehensive extraction
            text, confidence, engine = multi_ocr_engine.extract_text_comprehensive(cell_image)
        else:
            # Fallback to legacy method
            text, confidence, engine = multi_ocr_engine.best_ocr_result(cell_image)
        
        # Save enhanced crops if requested
        if save_enhanced_crops and text.strip():
            crop_filename = f"enhanced_cell_{i+1}_{text[:20].replace(' ', '_')}.png"
            crop_path = f"enhanced_crops/{crop_filename}"
            os.makedirs("enhanced_crops", exist_ok=True)
            cv2.imwrite(crop_path, cell_image)
        
        enhanced_results.append({
            'cell_id': i + 1,
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'padded_bbox': [int(padded_x1), int(padded_y1), int(padded_x2), int(padded_y2)],
            'text': text,
            'confidence': float(confidence),
            'ocr_engine': engine,
            'cell_size': f"{x2-x1}x{y2-y1}",
            'enhanced_processing': True
        })
    
    print("✅ Enhanced OCR processing: {} results".format(len(enhanced_results)))
    return enhanced_results

def organize_structured_table_data(enhanced_results):
    """Create structured table from enhanced OCR results"""
    print("🏗️ Organizing structured table data...")
    
    if not enhanced_results:
        return {
            'structured_data': [],
            'summary': {
                'total_cells': 0,
                'cells_with_text': 0,
                'confidence_avg': 0.0
            }
        }
    
    # Calculate summary statistics
    cells_with_text = [r for r in enhanced_results if r['text'].strip()]
    total_cells = len(enhanced_results)
    avg_confidence = np.mean([r['confidence'] for r in enhanced_results]) if enhanced_results else 0.0
    
    # Sort results by position (top to bottom, left to right)
    sorted_results = sorted(enhanced_results, key=lambda x: (x['bbox'][1], x['bbox'][0]))
    
    # Create structured data
    structured_data = []
    for result in sorted_results:
        if result['text'].strip():  # Only include cells with text
            structured_data.append({
                'position': f"Row_{result['bbox'][1]}_Col_{result['bbox'][0]}",
                'text': result['text'],
                'confidence': result['confidence'],
                'engine': result['ocr_engine'],
                'bbox': result['bbox']
            })
    
    summary = {
        'total_cells': total_cells,
        'cells_with_text': len(cells_with_text),
        'confidence_avg': float(avg_confidence),
        'engines_used': list(set(r['ocr_engine'] for r in enhanced_results if r['text'].strip()))
    }
    
    print(f"📊 Structured data summary:")
    print(f"   Rows with text: {len(structured_data)}")
    print(f"   Columns with text: {len(set(d['position'].split('_')[2] for d in structured_data))}")
    print(f"   Cells with text: {len(cells_with_text)}")
    print(f"   Total text items: {len(structured_data)}")
    print("✅ Structured table data organized successfully")
    
    return {
        'structured_data': structured_data,
        'summary': summary,
        'enhanced_results': enhanced_results
        }

