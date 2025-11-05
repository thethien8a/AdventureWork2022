# 🚀 AdventureWorks Sales Revenue Prediction

Dự án máy học dự đoán doanh thu đơn hàng cho AdventureWorks - công ty bán lẻ xe đạp và phụ kiện. Sử dụng mô hình XGBoost để dự đoán `TotalDue` (tổng doanh thu) dựa trên các thông tin khách hàng, sản phẩm và đơn hàng.

## 📋 Tổng quan

### 🎯 Mục tiêu
- Dự đoán doanh thu đơn hàng (`TotalDue`) để hỗ trợ:
  - Dự báo doanh số bán hàng
  - Lập kế hoạch cung ứng
  - Định giá sản phẩm
  - Phân tích hiệu suất kinh doanh

### 📊 Dữ liệu
- **Nguồn gốc**: AdventureWorks (bộ dữ liệu mẫu từ Microsoft)
- **Lĩnh vực**: Bán lẻ xe đạp và phụ kiện
- **Đặc trưng chính**:
  - Loại khách hàng (`PersonType`): Store Contact (SC), Individual (IN), Sales Person (SP), Employee (EM), Vendor Contact (VC), Government Contact (GC)
  - Số lượng đặt hàng (`OrderQty`): 1-1000
  - Tên sản phẩm (`Name`): 260+ SKU từ catalog sản phẩm (danh sách đầy đủ trong `sample.md`)
  - Dòng sản phẩm (`ProductLine`): Mountain (M), Road (R), Touring (T), Standard/Accessories (S)
  - Thông tin địa lý:
    - Khu vực lãnh thổ (`Name_territory`): Northeast, Northwest, Southeast, Southwest, Central, Canada, France, Germany, Australia, United Kingdom
    - Mã quốc gia (`CountryRegionCode`): US, CA, FR, AU, GB, DE
    - Nhóm (`Group`): North America, Europe, Pacific
  - Ngày đặt hàng (`OrderDate`): Format YYYY-MM-DD

## 🏗️ Kiến trúc hệ thống

### Backend API (FastAPI)
- **Framework**: FastAPI với Pydantic v2
- **Server**: Uvicorn
- **Endpoints**:
  - `POST /predict`: Dự đoán doanh thu cho 1 đơn hàng
  - `POST /predict/batch`: Dự đoán hàng loạt
  - `GET /health`: Kiểm tra sức khỏe hệ thống
  - `GET /`: Root endpoint với thông tin API
- **Model Management**: Quản lý preprocessing và XGBoost model thông qua `ModelManager`

### Frontend (Static Web App)
- **Ngôn ngữ**: Giao diện hoàn toàn bằng tiếng Việt
- **UI**: HTML/CSS/JavaScript thuần (không framework)
- **Tính năng**:
  - Form nhập liệu với dropdown options và autocomplete
  - Validation client-side theo quy tắc API
  - Dark/Light theme toggle (lưu trong localStorage)
  - Hiển thị kết quả dự đoán dạng tiền tệ VND/USD
  - Error handling với thông báo chi tiết
- **Data Source**: `options.json` chứa danh sách dropdown values
- **Responsive**: Thiết kế responsive cho mobile và desktop

### Model & Data Pipeline
- **Algorithm**: XGBoost Regressor
- **Preprocessing**: Pipeline xử lý dữ liệu tự động
- **Artifacts**: Lưu trữ dưới dạng `.joblib` files trong thư mục `models/`

## 📁 Cấu trúc dự án

```
AdventureWork2022/
├── 📊 dashboard/
│   └── dashboard.pbix              # Power BI Dashboard
├── 📋 database diagram/
│   └── Database Diagram.png        # Sơ đồ quan hệ database
├── 📁 data/                        # Raw data sources
│   ├── Customer.xlsx
│   ├── OrderDetail.xlsx
│   ├── OrderHeader.xlsx
│   ├── Person.xlsx
│   ├── Product.xlsx
│   └── Territory.xlsx
├── 🤖 models/                      # Trained model artifacts
│   ├── xgboost_model_model.joblib
│   └── xgboost_model_preprocessing.joblib
├── 📜 src/
│   ├── 📓 notebook/
│   │   ├── explore.ipynb           # Exploratory Data Analysis
│   │   └── train.ipynb             # Model training notebook
│   └── 📜 scripts/
│       ├── model_manager.py        # Model loading & prediction
│       ├── train_and_save.py       # Training pipeline
│       └── README.md
├── 🌐 web/
│   ├── 🔧 backend/
│   │   ├── config.py               # App configuration
│   │   ├── main.py                 # FastAPI app
│   │   └── models.py               # Pydantic schemas
│   └── 🎨 frontend/
│       ├── index.html              # Main UI
│       ├── styles.css              # Styling & themes
│       ├── app.js                  # Frontend logic
│       └── options.json            # Dropdown options
├── 📋 requirements.txt             # Python dependencies
├── 📋 sample.md                    # Sample data for dropdowns
└── 📋 README.md                    # This file
```

## 🛠️ Tech Stack

### Backend
- **Python 3.11+**
- **FastAPI 0.109.0**: Modern API framework
- **Uvicorn 0.27.0**: ASGI server
- **Pydantic 2.5.3**: Data validation
- **XGBoost 2.0.3**: Machine learning model
- **Pandas 2.1.4**: Data manipulation
- **Scikit-learn 1.3.2**: ML utilities

### Frontend
- **Vanilla HTML/CSS/JS**: No frameworks
- **Fetch API**: HTTP requests
- **Local Storage**: Theme persistence

### Development Tools
- **Power BI**: Dashboard creation
- **Jupyter Notebook**: Data exploration & training

## 🚀 Cài đặt và Chạy

### 1. Chuẩn bị môi trường
```bash
# Đảm bảo Python 3.11+
python --version

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Khởi động Backend API
```bash
# Từ thư mục gốc dự án
cd web/backend

# Chạy server với auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Kiểm tra API hoạt động:**
- Truy cập: http://localhost:8000/docs (Swagger UI)
- Health check: http://localhost:8000/health

### 3. Mở Frontend
```bash
# Từ thư mục gốc dự án
cd web/frontend

# Sử dụng Python HTTP server
python -m http.server 8080

# Hoặc mở trực tiếp index.html trong browser
```

**Truy cập**: http://localhost:8080/index.html

### 4. Train Model (nếu cần)
```bash
# Từ thư mục gốc
python src/scripts/train_and_save.py
```

## 📊 Dashboard Power BI

[Chèn ảnh dashboard Power BI ở đây]

## 🌐 Website Dự đoán Doanh thu

[Chèn ảnh website dự đoán doanh thu ở đây]

## 📋 API Documentation

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "PersonType": "Individual",
  "OrderQty": 1,
  "Name": "Mountain-100 Black, 42",
  "ProductLine": "Mountain",
  "Name_territory": "Northeast",
  "CountryRegionCode": "US",
  "Group": "North America",
  "OrderDate": "2013-01-01"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": 3374.99,
  "input_data": {
    "PersonType": "Individual",
    "OrderQty": 1,
    "Name": "Mountain-100 Black, 42",
    "ProductLine": "Mountain",
    "Name_territory": "Northeast",
    "CountryRegionCode": "US",
    "Group": "North America",
    "OrderDate": "2013-01-01"
  },
  "timestamp": "2025-11-04T10:30:00",
  "model_name": "xgboost_model"
}
```

### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "data": [
    {
      "PersonType": "SC",
      "OrderQty": 5,
      "Name": "Mountain-200 Black, 38",
      "ProductLine": "M",
      "Name_territory": "Southwest",
      "CountryRegionCode": "US",
      "Group": "North America",
      "OrderDate": "2013-07-01"
    },
    {
      "PersonType": "IN",
      "OrderQty": 10,
      "Name": "Road-350-W Yellow, 48",
      "ProductLine": "R",
      "Name_territory": "Canada",
      "CountryRegionCode": "CA",
      "Group": "North America",
      "OrderDate": "2013-08-15"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "total_records": 2,
  "predictions": [
    {
      "index": 0,
      "prediction": 1234.56,
      "input_data": {
        "PersonType": "SC",
        "OrderQty": 5,
        "Name": "Mountain-200 Black, 38",
        "ProductLine": "M",
        "Name_territory": "Southwest",
        "CountryRegionCode": "US",
        "Group": "North America",
        "OrderDate": "2013-07-01"
      }
    },
    {
      "index": 1,
      "prediction": 2468.12,
      "input_data": {
        "PersonType": "IN",
        "OrderQty": 10,
        "Name": "Road-350-W Yellow, 48",
        "ProductLine": "R",
        "Name_territory": "Canada",
        "CountryRegionCode": "CA",
        "Group": "North America",
        "OrderDate": "2013-08-15"
      }
    }
  ],
  "timestamp": "2025-11-04T10:30:00"
}
```

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "xgboost_model",
  "timestamp": "2025-11-04T10:30:00"
}
```

## 🔧 Development Scripts

### Kiểm tra model
```bash
python -c "from src.scripts.model_manager import ModelManager; m = ModelManager(); print('Model loaded successfully')"
```

## 📈 Performance & Accuracy

- **Model**: XGBoost Regressor với hyperparameter tuning
- **Metrics**: MAE, RMSE, R² Score (được tính trong training)
- **Preprocessing**: Feature engineering tự động
- **Validation**: Cross-validation trong quá trình training

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Tạo Pull Request

## 📄 License

This project is for educational purposes.

## 📞 Support

Nếu có câu hỏi hoặc vấn đề, vui lòng tạo issue trong repository này.

---

*Built with ❤️ for AdventureWorks sales analytics*
