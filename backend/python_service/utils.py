import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import pandas as pd
from docx import Document
import io
import types, sys, torch.nn as nn

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

# Model that detects the grid structure – rows and columns – inside a table
try:
    structure_model_path = MODEL_DIR / 'best(rowxcolumn).pt'
    print(f"Loading structure model from: {structure_model_path}")
    print(f"Model file exists: {structure_model_path.exists()}")
    structure_model = YOLO(str(structure_model_path))
    print("Structure model loaded successfully")
except Exception as e:
    print(f"Error loading structure model: {e}")
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

def process_pdf(file_path):
    """
    Main function to process the PDF file.
    It extracts tables from each page, recognizes cells, and extracts text.
    """
    print(f"Processing PDF file: {file_path}")
    doc = fitz.open(file_path)
    all_tables_data = []
    
    print(f"PDF has {len(doc)} pages")

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

            # Detect rows and columns in the table image
            structure_info = detect_table_structure_with_info(table_img)
            rows, cols = structure_info['rows'], structure_info['cols']

            # Detect individual cells for better text extraction
            cell_info = detect_cells_with_info(table_img)
            cells = cell_info['cells']

            # Reconstruct the table from the structure (using both row/col structure and cell detection)
            if cells and len(cells) > 0:
                print(f"Using cell-based extraction with {len(cells)} detected cells")
                table_data = build_table_from_cells(page, [x1, y1, x2, y2], cells, cell_info)
            else:
                print("Falling back to row/column-based extraction")
                table_data = build_table_from_structure(page, [x1, y1, x2, y2], rows, cols)

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
                },
                "structure_detection": {
                    "rows_detected": len(rows),
                    "cols_detected": len(cols),
                    "rows_confidence": structure_info['row_confidences'],
                    "cols_confidence": structure_info['col_confidences'],
                    "method": "ai_model",
                    "model_used": "best(rowxcolumn).pt"
                }
            }
            
            # Add cell detection info if cells were used
            if cells and len(cells) > 0:
                table_entry["cell_detection"] = {
                    "cells_detected": len(cells),
                    "cells_confidence": cell_info['confidences'],
                    "method": "ai_model",
                    "model_used": "best(cell).pt",
                    "extraction_method": "cell_based"
                }
            else:
                table_entry["cell_detection"] = {
                    "cells_detected": 0,
                    "method": "fallback",
                    "extraction_method": "structure_based"
                }
            
            all_tables_data.append(table_entry)

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
        
        # Filter by confidence threshold (adjust as needed)
        confidence_threshold = 0.5
        valid_indices = confidences >= confidence_threshold
        boxes = boxes[valid_indices]
        print(f"After filtering (conf >= {confidence_threshold}): {len(boxes)} table(s)")
        
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
        
        # Filter by confidence threshold (adjust as needed)
        confidence_threshold = 0.5
        valid_indices = confidences >= confidence_threshold
        filtered_boxes = boxes[valid_indices]
        filtered_confidences = confidences[valid_indices]
        print(f"After filtering (conf >= {confidence_threshold}): {len(filtered_boxes)} table(s)")
        
        return {
            'boxes': filtered_boxes,
            'confidences': filtered_confidences,
            'model': 'best(table).pt',
            'threshold': confidence_threshold
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

        # Filter by confidence and class
        confidence_threshold = 0.3  # Lower threshold for structure detection
        for box, cls, conf in zip(boxes, classes, confidences):
            if conf >= confidence_threshold:
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
    Detects rows and columns in a cropped table image and returns detailed info.
    """
    print(f"Detecting structure in table image of shape: {table_image.shape}")
    
    if structure_model is None:
        print("Structure model not loaded, cannot detect table structure")
        return {
            'rows': [], 'cols': [], 
            'row_confidences': [], 'col_confidences': [],
            'model': None, 'threshold': 0.3
        }
        
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
        row_confidences = []
        col_confidences = []
        
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

        # Filter by confidence and class
        confidence_threshold = 0.3  # Lower threshold for structure detection
        for box, cls, conf in zip(boxes, classes, confidences):
            if conf >= confidence_threshold:
                if cls == row_class_id:
                    rows.append(box)
                    row_confidences.append(float(conf))
                    print(f"Added row: {box} (conf: {conf:.3f})")
                elif cls == col_class_id:
                    cols.append(box)
                    col_confidences.append(float(conf))
                    print(f"Added column: {box} (conf: {conf:.3f})")

        # Sort rows by top coordinate, and columns by left coordinate
        # Also maintain the confidence order
        if rows:
            row_indices = sorted(range(len(rows)), key=lambda i: rows[i][1])
            rows = [rows[i] for i in row_indices]
            row_confidences = [row_confidences[i] for i in row_indices]
        
        if cols:
            col_indices = sorted(range(len(cols)), key=lambda i: cols[i][0])
            cols = [cols[i] for i in col_indices]
            col_confidences = [col_confidences[i] for i in col_indices]
        
        print(f"Final: {len(rows)} rows, {len(cols)} columns")
        
        return {
            'rows': rows,
            'cols': cols,
            'row_confidences': row_confidences,
            'col_confidences': col_confidences,
            'model': 'best(rowxcolumn).pt',
            'threshold': confidence_threshold,
            'class_names': class_names
        }
    else:
        print("No structure detected or no boxes found")
        return {
            'rows': [], 'cols': [], 
            'row_confidences': [], 'col_confidences': [],
            'model': 'best(rowxcolumn).pt', 'threshold': 0.3
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
            'model': None, 'threshold': 0.3
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
        
        # Filter by confidence threshold
        confidence_threshold = 0.3  # Threshold for cell detection
        for box, conf in zip(boxes, confidences):
            if conf >= confidence_threshold:
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
            'threshold': confidence_threshold,
            'class_names': class_names
        }
    else:
        print("No cells detected or no boxes found")
        return {
            'cells': [], 'confidences': [], 
            'model': 'best(cell).pt', 'threshold': 0.3
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
    
    # Extract text from each detected cell
    cell_data = []
    for i, cell_box in enumerate(cells):
        confidence = cell_info['confidences'][i] if i < len(cell_info['confidences']) else 0.0
        text = extract_text_from_cell_region(page, table_box, cell_box)
        
        cell_data.append({
            'cell_id': i + 1,
            'bbox': cell_box.tolist(),
            'text': text,
            'confidence': confidence,
            'extraction_method': 'cell_detection'
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
    Extracts text by finding the intersection of row and column boxes.
    """
    print(f"Building table from {len(rows)} rows and {len(cols)} columns")
    
    # If no rows or cols detected, try to extract text from the entire table area
    if not rows or not cols:
        print("No structure detected, extracting text from entire table area")
        table_rect = fitz.Rect(table_box[0], table_box[1], table_box[2], table_box[3])
        text = page.get_text("text", clip=table_rect, sort=True).strip()
        print(f"Extracted text from entire table: '{text[:100]}...'")
        
        # Return as a single cell if we got text
        if text:
            return [[text]]
        else:
            return [[""]]
    
    table_data = []
    table_origin_x, table_origin_y = table_box[0], table_box[1]

    for row_idx, row_box in enumerate(rows):
        row_data = []
        print(f"Processing row {row_idx}: {row_box}")
        
        for col_idx, col_box in enumerate(cols):
            # Calculate intersection of row and column to define the cell
            cell_x1 = max(row_box[0], col_box[0])
            cell_y1 = max(row_box[1], col_box[1])
            cell_x2 = min(row_box[2], col_box[2])
            cell_y2 = min(row_box[3], col_box[3])
            
            if cell_x1 < cell_x2 and cell_y1 < cell_y2:
                # The cell coordinates are relative to the table image,
                # so we need to convert them to page coordinates.
                cell_rect_on_page = fitz.Rect(
                    table_origin_x + cell_x1,
                    table_origin_y + cell_y1,
                    table_origin_x + cell_x2,
                    table_origin_y + cell_y2
                )
                
                # Get text from the cell area
                text = page.get_text("text", clip=cell_rect_on_page, sort=True).strip()
                print(f"  Cell ({row_idx},{col_idx}): '{text}' from rect {cell_rect_on_page}")
                row_data.append(text)
            else:
                print(f"  Cell ({row_idx},{col_idx}): No intersection")
                row_data.append("") # No intersection
        table_data.append(row_data)

    print(f"Final table data: {len(table_data)} rows")
    return table_data

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

