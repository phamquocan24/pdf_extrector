#!/usr/bin/env python3
"""
Debug script to test PDF table extraction without the web service.
"""
import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our utils
import utils

def test_extraction(pdf_path):
    """Test PDF extraction with debug output"""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    print(f"Testing extraction on: {pdf_path}")
    print("=" * 50)
    
    try:
        # Process the PDF
        results = utils.process_pdf(pdf_path)
        
        print("\n" + "=" * 50)
        print("FINAL RESULTS:")
        print(f"Total tables found: {len(results)}")
        
        for i, table_info in enumerate(results):
            print(f"\nTable {i+1}:")
            print(f"  Page: {table_info['page']}")
            print(f"  Method: {table_info.get('method', 'unknown')}")
            print(f"  Rows: {len(table_info['data'])}")
            if table_info['data']:
                print(f"  Columns: {len(table_info['data'][0])}")
                print("  Sample data:")
                for j, row in enumerate(table_info['data'][:3]):  # Show first 3 rows
                    print(f"    Row {j+1}: {row}")
            else:
                print("  No data extracted")
                
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # The script is in backend/python_service, so project root is two levels up.
    project_root = Path(__file__).resolve().parent.parent
    
    # Default PDF path is in the project root.
    pdf_path = project_root / "test_extractor.pdf"
    
    # Allow overriding with a command-line argument.
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    
    test_extraction(str(pdf_path))