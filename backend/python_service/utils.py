import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
# import pandas as pd  # Disabled due to numpy compatibility issues
# from docx import Document  # Disabled - not needed for core functionality
import io
import types, sys, torch.nn as nn
import base64
import os
from pathlib import Path
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

            # Phase 3: Extract text from detected cells
            if cells and len(cells) > 0:
                print(f"Using cell-based extraction with {len(cells)} detected cells")
                table_data = build_table_from_cells(page, [x1, y1, x2, y2], cells, cell_info)
            else:
                print("No cells detected, creating empty table")
                table_data = []

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
            
            # Add cell detection info
            table_entry["cell_detection"] = {
                "cells_detected": len(cells) if cells else 0,
                "cells_confidence": [float(conf) for conf in cell_info['confidences']] if cells and cell_info else [],
                "method": "ai_model", 
                "model_used": "best(cell).pt",
                "extraction_method": "enhanced_multi_method"
            }
            
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
    Extract text from a specific cell region using PyMuPDF.
    """
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
        # Extract text from the rectangle
        text = page.get_text("text", clip=rect).strip()
        if text:
            print(f"Extracted text from cell [{cx1},{cy1},{cx2},{cy2}]: '{text}'")
            return text
        else:
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

