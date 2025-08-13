#!/usr/bin/env python3
"""
Advanced OCR Pipeline Integration
Based on OCR_cell.ipynb workflow: CELL 1→2→3→4→8→9→10

This module integrates the enhanced OCR pipeline into the main system
for processing tables after cell segmentation.
"""

import cv2
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path

from utils import (
    EasyOCREngine, AdvancedImagePreprocessor,
    enhanced_cell_cropping_and_ocr, organize_structured_table_data,
    initialize_ocr_engine, detect_cells_with_info
)

class TableOCRPipeline:
    """
    Optimized Table OCR Pipeline using EasyOCR
    
    Pipeline Structure (optimized for cell segmentation):
    1. CELL 1-2: Setup & Model Loading ✅ (Done in utils.py)
    2. CELL 3: Table Detection ✅ (Done in main workflow)  
    3. CELL 4: Visualization ✅ (Done in main workflow)
    4. CELL 5: EasyOCR Setup ✅ (EasyOCREngine)
    5. CELL 8: EasyOCR Engine ✅ (EasyOCREngine - optimized)
    6. CELL 9: Enhanced Processing ✅ (enhanced_cell_cropping_and_ocr)
    7. CELL 10: Final Results ✅ (organize_structured_table_data)
    
    Optimizations:
    - Uses only EasyOCR for best performance
    - Multiple preprocessing methods for better accuracy
    - Optimized parameters for cell text extraction
    """
    
    def __init__(self, save_intermediate_results=True):
        """Initialize the OCR pipeline"""
        self.save_intermediate = save_intermediate_results
        self.ocr_engine = None
        self.results_dir = Path("pipeline_results")
        self.results_dir.mkdir(exist_ok=True)
        
        print("🚀 Initializing Optimized Table OCR Pipeline (EasyOCR)...")
        self._initialize_ocr_engine()
    
    def _initialize_ocr_engine(self):
        """CELL 5 & 8: Initialize EasyOCR Engine"""
        print("🔧 CELL 5 & 8: EasyOCR Engine Setup (Optimized)...")
        
        try:
            initialize_ocr_engine()
            from utils import multi_ocr_engine
            self.ocr_engine = multi_ocr_engine
            
            if self.ocr_engine:
                print("✅ EasyOCR Engine initialized successfully")
                return True
            else:
                print("⚠️ EasyOCR Engine initialized in fallback mode")
                return True
                
        except Exception as e:
            print(f"❌ OCR Engine initialization failed: {e}")
            return False
    
    def process_table_cells(self, table_image: np.ndarray, cell_detections: Dict[str, Any], 
                          table_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main pipeline processing function
        
        Args:
            table_image: Cropped table image (from table detection)
            cell_detections: Cell detection results with 'cells' key
            table_info: Additional table information
            
        Returns:
            Complete pipeline results with structured data
        """
        print("🚀 Starting Advanced Table OCR Pipeline Processing...")
        
        pipeline_results = {
            'pipeline_stage': 'complete',
            'table_info': table_info or {},
            'cell_detections': cell_detections,
            'processing_steps': []
        }
        
        # CELL 9: Enhanced Cell Processing
        print("\n⚡ CELL 9: Enhanced Cell Processing")
        enhanced_results = self._enhanced_cell_processing(table_image, cell_detections)
        pipeline_results['enhanced_cell_results'] = enhanced_results
        pipeline_results['processing_steps'].append('enhanced_cell_processing')
        
        # CELL 10: Final Results & Structured Table
        print("\n🎉 CELL 10: Final Results & Structured Table")
        structured_results = self._create_structured_table(enhanced_results)
        pipeline_results['structured_table'] = structured_results
        pipeline_results['processing_steps'].append('structured_table_creation')
        
        # Save intermediate results if requested
        if self.save_intermediate:
            self._save_pipeline_results(pipeline_results)
        
        # Create final summary
        final_summary = self._create_final_summary(pipeline_results)
        pipeline_results['final_summary'] = final_summary
        
        print("✅ Advanced Table OCR Pipeline completed successfully!")
        return pipeline_results
    
    def _enhanced_cell_processing(self, table_image: np.ndarray, 
                                cell_detections: Dict[str, Any]) -> List[Dict[str, Any]]:
        """CELL 9: Enhanced cell cropping and OCR processing"""
        
        if not self.ocr_engine:
            print("⚠️ No OCR engine available for enhanced processing")
            return []
        
        # Use the enhanced cell processing function from utils
        enhanced_results = enhanced_cell_cropping_and_ocr(
            cell_detections, 
            table_image, 
            save_enhanced_crops=self.save_intermediate
        )
        
        print(f"✅ Enhanced processing completed: {len(enhanced_results)} cells processed")
        
        # Add additional metadata
        for i, result in enumerate(enhanced_results):
            result['pipeline_step'] = 'enhanced_processing'
            result['processing_order'] = i + 1
            result['table_region'] = True
        
        return enhanced_results
    
    def _create_structured_table(self, enhanced_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """CELL 10: Create structured table from enhanced OCR results"""
        
        # Use the organize function from utils
        structured_data = organize_structured_table_data(enhanced_results)
        
        # Add pipeline-specific metadata
        structured_data['pipeline_info'] = {
            'processing_method': 'advanced_ocr_pipeline',
            'cell_detection_method': 'yolo_cell_model',
            'ocr_method': 'multi_engine_comprehensive',
            'preprocessing': 'advanced_multi_scale_enhancement'
        }
        
        # Create table matrix if possible
        if enhanced_results:
            table_matrix = self._create_table_matrix(enhanced_results)
            structured_data['table_matrix'] = table_matrix
        
        return structured_data
    
    def _create_table_matrix(self, enhanced_results: List[Dict[str, Any]]) -> List[List[str]]:
        """Create 2D table matrix from cell results"""
        if not enhanced_results:
            return []
        
        # Find table dimensions
        max_row = max(result['bbox'][1] for result in enhanced_results)
        max_col = max(result['bbox'][0] for result in enhanced_results)
        min_row = min(result['bbox'][1] for result in enhanced_results)
        min_col = min(result['bbox'][0] for result in enhanced_results)
        
        # Create a simple grid based on positions
        # This is a basic implementation - could be improved with better row/column detection
        sorted_results = sorted(enhanced_results, key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        # Group by approximate rows
        rows = []
        current_row = []
        current_y = None
        threshold = 20  # pixels threshold for same row
        
        for result in sorted_results:
            y = result['bbox'][1]
            if current_y is None or abs(y - current_y) <= threshold:
                current_row.append(result['text'])
                current_y = y if current_y is None else current_y
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [result['text']]
                current_y = y
        
        if current_row:
            rows.append(current_row)
        
        return rows
    
    def _save_pipeline_results(self, pipeline_results: Dict[str, Any]):
        """Save intermediate pipeline results"""
        
        # Save enhanced results as JSON
        enhanced_file = self.results_dir / "enhanced_ocr_results.json"
        with open(enhanced_file, 'w', encoding='utf-8') as f:
            json.dump(pipeline_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save structured table as CSV if available
        if 'structured_table' in pipeline_results:
            structured_data = pipeline_results['structured_table']
            if 'table_matrix' in structured_data and structured_data['table_matrix']:
                csv_file = self.results_dir / "enhanced_structured_table.csv"
                import csv
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(structured_data['table_matrix'])
                print(f"📁 Structured table saved to: {csv_file}")
        
        print(f"📁 Pipeline results saved to: {self.results_dir}")
    
    def _create_final_summary(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create final pipeline summary"""
        
        enhanced_results = pipeline_results.get('enhanced_cell_results', [])
        structured_table = pipeline_results.get('structured_table', {})
        
        summary = {
            'pipeline_completed': True,
            'processing_steps': pipeline_results.get('processing_steps', []),
            'total_cells_detected': len(enhanced_results),
            'cells_with_text': len([r for r in enhanced_results if r.get('text', '').strip()]),
            'average_confidence': np.mean([r.get('confidence', 0) for r in enhanced_results]) if enhanced_results else 0,
            'ocr_engines_used': list(set(r.get('ocr_engine', 'unknown') for r in enhanced_results)),
            'table_dimensions': {
                'rows': len(structured_table.get('table_matrix', [])),
                'columns': max(len(row) for row in structured_table.get('table_matrix', [])) if structured_table.get('table_matrix') else 0
            },
            'processing_quality': self._assess_processing_quality(enhanced_results)
        }
        
        return summary
    
    def _assess_processing_quality(self, enhanced_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality of OCR processing"""
        
        if not enhanced_results:
            return {'quality_score': 0, 'assessment': 'no_data'}
        
        # Calculate quality metrics
        total_cells = len(enhanced_results)
        cells_with_text = len([r for r in enhanced_results if r.get('text', '').strip()])
        avg_confidence = np.mean([r.get('confidence', 0) for r in enhanced_results])
        
        text_coverage = cells_with_text / total_cells if total_cells > 0 else 0
        
        # Quality score (0-1)
        quality_score = (text_coverage * 0.6) + (avg_confidence * 0.4)
        
        # Assessment level
        if quality_score >= 0.8:
            assessment = 'excellent'
        elif quality_score >= 0.6:
            assessment = 'good'
        elif quality_score >= 0.4:
            assessment = 'fair'
        else:
            assessment = 'poor'
        
        return {
            'quality_score': round(quality_score, 3),
            'assessment': assessment,
            'text_coverage': round(text_coverage, 3),
            'average_confidence': round(avg_confidence, 3),
            'total_cells': total_cells,
            'cells_with_text': cells_with_text
        }

# Integration function for the main system
def process_table_with_advanced_ocr(table_image: np.ndarray, 
                                   cell_detections: Dict[str, Any],
                                   table_info: Dict[str, Any] = None,
                                   save_results: bool = True) -> Dict[str, Any]:
    """
    Main integration function for processing tables with advanced OCR pipeline
    
    This function integrates the OCR_cell.ipynb pipeline into the main system
    
    Args:
        table_image: Cropped table image from table detection
        cell_detections: Cell detection results from YOLO cell model
        table_info: Additional table metadata
        save_results: Whether to save intermediate results
        
    Returns:
        Complete OCR processing results with structured data
    """
    
    print("🚀 Starting Advanced OCR Pipeline Integration...")
    
    # Initialize pipeline
    pipeline = TableOCRPipeline(save_intermediate_results=save_results)
    
    # Process table with enhanced OCR
    results = pipeline.process_table_cells(table_image, cell_detections, table_info)
    
    print("✅ Advanced OCR Pipeline Integration completed!")
    return results

if __name__ == "__main__":
    # Test the pipeline with sample data
    print("🧪 Testing Advanced OCR Pipeline...")
    
    # Create test table image
    test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_image, (50, 50), (350, 150), (0, 0, 0), 2)
    cv2.putText(test_image, "Sample Table", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Mock cell detections
    test_detections = {
        'cells': [
            [50, 50, 200, 100],
            [200, 50, 350, 100],
            [50, 100, 200, 150],
            [200, 100, 350, 150]
        ]
    }
    
    # Test processing
    results = process_table_with_advanced_ocr(test_image, test_detections)
    
    print("🎉 Pipeline test completed!")
    print(f"📊 Summary: {results['final_summary']}")
