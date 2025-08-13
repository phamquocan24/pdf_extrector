# 🚀 Enhanced Text Extraction Setup Guide

## 📋 Tổng quan cải tiến

Hệ thống đã được nâng cấp với **3 phương thức trích xuất text** để cải thiện độ chính xác:

### 🔧 Phương thức trích xuất:
1. **PyMuPDF** - Cho PDF có text layer (nhanh nhất)
2. **EasyOCR** - Cho nội dung dạng ảnh/scan (chính xác)  
3. **LLMs** - Làm sạch và cải thiện text (thông minh)

### 🧠 LLM Models hỗ trợ:
- **Groq (Llama 3.1)** - Miễn phí, nhanh
- **OpenAI (GPT-3.5)** - Backup option

## 🛠️ Cài đặt Dependencies

### 1. Cài đặt Python packages
```bash
cd backend/python_service
pip install -r requirements.txt
```

### 2. Cài đặt EasyOCR (có thể mất vài phút)
```bash
# EasyOCR sẽ tự động download language models
# Vietnamese + English models (~100MB)
```

### 3. Cấu hình API Keys

Tạo file `.env` trong `backend/python_service/`:
```bash
# LLM API Keys
GROQ_API_KEY=gsk_15Q8kI09YgYz7JywxXdv6MGdyb3FY1jTJRVI1QLCPPuhzZzXlv1AP
OPENAI_API_KEY=your_openai_key_here

# Optional: Add other AI service API keys
# ANTHROPIC_API_KEY=
# HUGGINGFACE_API_KEY=
```

## 🧪 Kiểm tra cài đặt

```bash
cd backend/python_service
python test_enhanced_extraction.py
```

Expected output:
```
🧪 Testing Enhanced Text Extraction
==================================================
✅ Utils module imported successfully
✅ EasyOCR reader initialized
✅ Groq LLM client initialized
ℹ️  OpenAI client not initialized (no API key)

🔧 Available Text Extraction Methods:
1. PyMuPDF (text-based PDFs)
2. EasyOCR (image-based content)  
3. LLM text processing (Groq/OpenAI)

🧠 Testing LLM text processing...
Input: '2002\\n \\n2001\\n \\n2000\\n \\benefit cost\\n \\n'
Output: '2002 2001 2000 benefit cost'
✅ LLM text processing working

🎯 Enhanced Extraction Features:
✅ Multi-method fallback strategy
✅ EasyOCR for image-based content
✅ LLM-powered text cleaning
✅ Vietnamese + English language support
✅ Confidence-based filtering
```

## 🔄 Workflow mới

### 1. Table Detection (không đổi)
- Sử dụng YOLO model `best(table).pt`
- Phát hiện vùng bảng trong PDF

### 2. Cell Detection (không đổi)  
- Sử dụng YOLO model `best(cell).pt`
- Phát hiện từng cell riêng lẻ

### 3. **Enhanced Text Extraction** (MỚI)
```python
def extract_text_enhanced(page, table_box, cell_box):
    # Method 1: PyMuPDF (fastest for text PDFs)
    pymupdf_text = extract_text_from_cell_region(page, table_box, cell_box)
    if pymupdf_text and len(pymupdf_text.strip()) > 2:
        return process_text_with_llm(pymupdf_text, "pdf_text")
    
    # Method 2: EasyOCR (for image content)
    if ocr_reader:
        ocr_text = extract_text_easyocr(cell_img)
        if ocr_text:
            return process_text_with_llm(ocr_text, "ocr_text")
    
    # Method 3: Return cleaned PyMuPDF result
    return process_text_with_llm(pymupdf_text, "pdf_text")
```

### 4. **LLM Text Processing** (MỚI)
```python
def process_text_with_llm(raw_text, context):
    # Groq (primary) - Free and fast
    if groq_client:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    
    # OpenAI (fallback) - If Groq fails
    # ... similar implementation
```

## 📊 Cải thiện chất lượng

### Trước (chỉ PyMuPDF):
```csv
"2002\n \n2001\n \n2000\n \benefit cost\n \n"
"", "", ""
"207\n\n175\n\n108", "", ""
```

### Sau (Enhanced Multi-Method):
```csv
"2002","2001","2000","benefit cost"
"","","",""  
"207","175","108",""
```

## 🐛 Troubleshooting

### EasyOCR không hoạt động
```bash
# Kiểm tra CUDA (nếu có GPU)
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall EasyOCR
pip uninstall easyocr
pip install easyocr
```

### Groq API lỗi
```bash
# Kiểm tra API key
echo $GROQ_API_KEY

# Test API connection
curl -X POST "https://api.groq.com/openai/v1/chat/completions" \
  -H "Authorization: Bearer $GROQ_API_KEY" \
  -H "Content-Type: application/json"
```

### Memory issues với EasyOCR
```python
# Trong utils.py, line 104
ocr_reader = easyocr.Reader(['vi', 'en'], gpu=False)  # Force CPU
```

## 🚀 Khởi động service

```bash
# Terminal 1: Python Service (Enhanced)
cd backend/python_service
python app.py

# Terminal 2: Node.js Backend  
cd backend
npm start

# Terminal 3: Frontend
cd frontend
npm run dev
```

## 📈 Performance Monitoring

Monitor logs cho:
- `PyMuPDF extraction` - text-based content
- `EasyOCR extraction` - image-based content  
- `LLM processed` - text cleaning results
- Extraction method fallbacks

## 🔮 Future Enhancements

- [ ] **Fine-tuned OCR models** cho Vietnamese financial tables
- [ ] **Local LLM** deployment (Llama, Mistral)
- [ ] **Confidence scoring** cho extraction quality
- [ ] **Batch processing** cho multiple cells
- [ ] **Custom prompt templates** cho different table types
