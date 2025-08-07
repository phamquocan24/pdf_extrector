# 📄 PDF Table Extractor - AI-Powered Table Detection & Extraction

## 🌟 Tổng quan dự án

**PDF Table Extractor** là một hệ thống thông minh sử dụng AI để tự động phát hiện, phân tích và trích xuất dữ liệu từ các bảng trong file PDF. Dự án kết hợp 3 models AI tiên tiến để đạt độ chính xác cao trong việc nhận dạng và xử lý bảng.

### ✨ Tính năng chính

- 🤖 **AI-Powered Detection**: Sử dụng 3 models YOLO chuyên biệt
- 📊 **Table Recognition**: Phát hiện vùng bảng với độ chính xác 66-88%
- 🔍 **Structure Analysis**: Nhận dạng rows/columns với confidence 80-90%
- 📱 **Cell-Level Extraction**: Trích xuất text từng cell với độ chính xác cao
- 💾 **Multiple Export Formats**: CSV, TXT, JSON
- 🎨 **Modern UI**: Interface đẹp với Material-UI
- 📈 **Confidence Visualization**: Hiển thị độ tin cậy chi tiết

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │  Node.js API    │    │ Python Service  │
│   (React)       │◄──►│   (Express)     │◄──►│    (FastAPI)    │
│   Port: 5173    │    │   Port: 8080    │    │   Port: 8001    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                              ┌─────────────────┐
                                              │   AI Models     │
                                              │ ┌─────────────┐ │
                                              │ │Table Model  │ │
                                              │ │ Cell Model  │ │
                                              │ │OCR Extraction│ │
                                              │ └─────────────┘ │
                                              └─────────────────┘
```

### 🧠 AI Models Pipeline (3-Phase Workflow)

1. **Phase 1: Table Detection Model** (`best(table).pt`)
   - Phát hiện vùng bảng trong PDF
   - Confidence: 66-88%
   - Output: Bounding boxes của các bảng

2. **Phase 2: Cell Detection Model** (`best(cell).pt`)
   - Phát hiện và segment từng cell riêng lẻ
   - Confidence: 30-90%
   - Output: Boundaries chính xác của cells

3. **Phase 3: OCR Text Extraction**
   - Trích xuất text từ từng cell đã segment
   - Method: PyMuPDF text extraction
   - Output: Text content cho từng cell

## 🛠️ Tech Stack

### Frontend
- **React 18** - UI Framework
- **Material-UI (MUI)** - Component Library
- **React Router** - Navigation
- **Axios** - HTTP Client
- **Vite** - Build Tool

### Backend
- **Node.js** - Runtime Environment
- **Express.js** - Web Framework
- **Multer** - File Upload Handling
- **MongoDB** - Database
- **Mongoose** - ODM

### AI Service
- **Python 3.11** - Programming Language
- **FastAPI** - API Framework
- **PyTorch** - Deep Learning Framework
- **Ultralytics YOLO** - Object Detection
- **PyMuPDF** - PDF Processing
- **OpenCV** - Image Processing

## 📁 Cấu trúc dự án

```
Final_term/
├── 📁 frontend/                 # React Frontend
│   ├── src/
│   │   ├── components/         # Reusable components
│   │   ├── pages/             # Page components
│   │   ├── layouts/           # Layout components
│   │   └── contexts/          # React contexts
│   ├── public/                # Static files
│   └── package.json
│
├── 📁 backend/                  # Node.js Backend
│   ├── controllers/           # Route controllers
│   ├── models/               # AI model files
│   │   ├── best(table).pt    # Table detection model
│   │   └── best(cell).pt     # Cell detection model
│   ├── python_service/       # Python AI service
│   │   ├── app.py           # FastAPI application
│   │   ├── utils.py         # AI processing logic
│   │   └── requirements.txt # Python dependencies
│   ├── routes/              # API routes
│   ├── services/            # Business logic
│   └── server.js           # Express server
│
├── 📄 test_extractor.pdf       # Sample test file
├── 📄 README.md               # Project documentation
└── 📄 START_GUIDE.md          # Setup instructions
```

## 🚀 Cài đặt và chạy dự án

### Yêu cầu hệ thống
- **Node.js** >= 16.0.0
- **Python** >= 3.8
- **npm** hoặc **yarn**
- **MongoDB** (local hoặc cloud)

### 1. Clone repository
```bash
git clone <repository-url>
cd Final_term
```

### 2. Cài đặt Frontend
```bash
cd frontend
npm install
```

### 3. Cài đặt Backend
```bash
cd ../backend
npm install
```

### 4. Cài đặt Python Service
```bash
cd python_service
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 5. Khởi động tất cả services

#### Terminal 1: Python Service
```bash
cd backend/python_service
python app.py
```

#### Terminal 2: Node.js Backend
```bash
cd backend
npm start
```

#### Terminal 3: Frontend
```bash
cd frontend
npm run dev
```

### 6. Truy cập ứng dụng
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8080
- **Python Service**: http://localhost:8001

## 📋 Hướng dẫn sử dụng

### 1. Upload PDF File
1. Mở trình duyệt và truy cập http://localhost:5173
2. Đăng nhập vào hệ thống
3. Tại Dashboard, click vào vùng upload hoặc kéo thả file PDF
4. Chờ hệ thống xử lý (thường 10-30 giây)

### 2. Xem kết quả
1. Sau khi xử lý xong, click "Preview" để xem chi tiết
2. Kiểm tra thông tin detection:
   - **Table Detection**: Confidence và bounding boxes
   - **Structure Detection**: Số rows/columns và confidence
   - **Cell Detection**: Số cells và accuracy

### 3. Export dữ liệu
- **CSV**: Xuất dữ liệu bảng dạng comma-separated
- **TXT**: Xuất text thuần từ các cells
- **JSON**: Xuất toàn bộ metadata và confidence scores

## 🎯 Hiệu suất AI Models (3-Phase Workflow)

| Phase | Model | Chức năng | Confidence Range | Performance |
|-------|-------|-----------|------------------|-------------|
| 1 | Table Detection | Phát hiện vùng bảng | 66-88% | ⭐⭐⭐⭐ |
| 2 | Cell Detection | Segment cells | 30-90% | ⭐⭐⭐⭐ |
| 3 | OCR Extraction | Trích xuất text | N/A | ⭐⭐⭐ |

## 🔧 API Documentation

### Backend API Endpoints

#### POST `/api/upload`
Upload và xử lý file PDF
```bash
curl -X POST -F "file=@document.pdf" http://localhost:8080/api/upload
```

#### Response Format
```json
{
  "data": [
    {
      "page": 1,
      "table": 1,
      "data": [["cell1", "cell2"], ["cell3", "cell4"]],
      "method": "ai_model",
      "table_detection": {
        "confidence": 0.85,
        "bbox": [100, 200, 500, 600]
      },
      "cell_detection": {
        "cells_detected": 4,
        "method": "ai_model",
        "cells_confidence": [0.8, 0.9, 0.7, 0.85]
      }
    }
  ]
}
```

### Python Service API

#### POST `/api/extract`
Xử lý file PDF với AI models
```python
import requests
files = {'file': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8001/api/extract', files=files)
```

## 🐛 Troubleshooting

### Lỗi thường gặp

#### 1. Models không load được
```bash
# Kiểm tra file models
ls backend/models/
# Đảm bảo có 2 files: best(table).pt, best(cell).pt
```

#### 2. Python service lỗi
```bash
# Kiểm tra dependencies
cd backend/python_service
pip list
# Reinstall nếu cần
pip install -r requirements.txt
```

#### 3. Frontend không connect được backend
```bash
# Kiểm tra ports
netstat -an | findstr "5173 8080 8001"
```

#### 4. Không extract được text
- Đảm bảo PDF không phải scan image
- Thử với PDF có text layer
- Kiểm tra logs Python service

## 📊 Monitoring & Logs

### Backend Logs
```bash
# Node.js logs
cd backend && npm start

# Python service logs  
cd backend/python_service && python app.py
```

### Frontend Logs
- Mở Developer Tools (F12)
- Kiểm tra Console tab
- Xem Network tab cho API calls

## 🤝 Contributing

### Development Workflow
1. Fork repository
2. Tạo feature branch
3. Implement changes
4. Test thoroughly
5. Submit pull request

### Code Style
- **Frontend**: ESLint + Prettier
- **Backend**: StandardJS
- **Python**: PEP8 + Black formatter

## 📈 Future Enhancements

- [ ] **OCR Integration**: Hỗ trợ PDF scan images
- [ ] **Batch Processing**: Xử lý nhiều files cùng lúc
- [ ] **Cloud Storage**: Tích hợp AWS S3/Google Cloud
- [ ] **Real-time Processing**: WebSocket updates
- [ ] **Model Fine-tuning**: Cải thiện accuracy
- [ ] **Mobile App**: React Native version

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Developer**: "Xay Dung He Thong Tu Dong Trich Xuat Bang Du Lieu Tai Chinh"
- **Institution**: CMC University
- **Course**: Computer Vision - Final Term Project

## 📞 Support

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra [START_GUIDE.md](START_GUIDE.md)
2. Xem phần Troubleshooting ở trên
3. Tạo issue trên GitHub
4. Liên hệ qua email: [bit220006@st.cmcu.edu.vn]

---

⭐ **Nếu project này hữu ích, hãy cho chúng tôi một star!** ⭐
