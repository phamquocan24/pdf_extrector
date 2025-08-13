# Hướng dẫn khởi động hệ thống PDF Extractor

## 🚀 Khởi động tất cả services

### 1. Python Service (Port 8005)
```bash
cd backend/python_service
python app.py
```

### 2. Node.js Backend (Port 8080)
```bash
cd backend
npm start
```

### 3. Frontend (Port 5173)
```bash
cd frontend
npm run dev
```

## 📋 Kiểm tra services

### Kiểm tra Python Service:
- URL: http://localhost:8005
- Test endpoint: POST /api/extract

### Kiểm tra Node.js Backend:
- URL: http://localhost:8080
- Test endpoint: POST /api/upload

### Kiểm tra Frontend:
- URL: http://localhost:5173

## 🧪 Test luồng dữ liệu

1. **Mở Frontend**: http://localhost:5173
2. **Upload PDF file** từ Dashboard
3. **Kiểm tra response** trong Network tab (F12)
4. **Xem Preview** với dữ liệu nhận được

## 🐛 Troubleshooting

### Nếu không nhận được dữ liệu:

1. **Kiểm tra logs** của từng service
2. **Xem Network tab** để check API calls
3. **Verify file path**: Đảm bảo test file PDF tồn tại
4. **Check console errors** trong browser

### Expected Response Structure:
```json
{
  "data": [
    {
      "page": 1,
      "table": 1,
      "data": [...],
      "method": "ai_model",
      "table_detection": {
        "confidence": 0.8,
        "bbox": [...]
      },
      "structure_detection": {
        "rows_detected": 5,
        "cols_detected": 3,
        "rows_confidence": [...],
        "cols_confidence": [...]
      },
      "cell_detection": {
        "cells_detected": 15,
        "cells_confidence": [...],
        "method": "ai_model"
      }
    }
  ]
}
```

## 📁 Test Files

Sử dụng file test: `test_extractor.pdf` trong project root để test.
