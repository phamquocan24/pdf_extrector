#!/usr/bin/env python3
"""
Test CSV output format from JSON data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import process_pdf
import json
import csv
from io import StringIO

def test_csv_format():
    """Test CSV formatting from OCR results"""
    print("🧪 Testing CSV Output Format...")
    
    try:
        # Process PDF
        results = process_pdf('../../test_extractor.pdf')
        
        if not results:
            print("❌ No results from PDF processing")
            return False
        
        # Get first table data
        first_table = results[0]
        table_data = first_table.get('data', [])
        
        print(f"📊 Table data: {len(table_data)} rows")
        
        # Create CSV format
        csv_output = StringIO()
        csv_writer = csv.writer(csv_output)
        
        # Write data to CSV
        for row in table_data:
            csv_writer.writerow(row)
        
        csv_string = csv_output.getvalue()
        
        print("\n📋 CSV Output (first 10 lines):")
        lines = csv_string.strip().split('\n')
        for i, line in enumerate(lines[:10]):
            print(f"  {i+1}: {line}")
        
        print(f"\n📈 Total CSV lines: {len(lines)}")
        
        # Test JSON format
        print(f"\n🔤 JSON Sample (first 3 rows):")
        sample_data = table_data[:3]
        print(json.dumps(sample_data, indent=2, ensure_ascii=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_csv_format()
    sys.exit(0 if success else 1)
