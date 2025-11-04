#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration settings for FastAPI backend
"""

from pathlib import Path
from typing import List


class Settings:
    """Application settings"""
    
    # Application Info
    APP_TITLE: str = "AdventureWorks Revenue Prediction API"
    APP_DESCRIPTION: str = """
    🎯 API dự đoán doanh thu cho AdventureWorks
    
    ## Features
    
    * **Single Prediction**: Dự đoán cho 1 đơn hàng
    * **Batch Prediction**: Dự đoán cho nhiều đơn hàng cùng lúc
    * **Health Check**: Kiểm tra trạng thái server và model
    * **Auto Documentation**: Swagger UI và ReDoc
    
    ## Model Information
    
    * Model: XGBoost Regressor
    * Target: TotalDue (Doanh thu)
    * Features: 8 features (PersonType, OrderQty, Name, ProductLine, Territory, Country, Group, OrderDate)
    """
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # CORS Settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React default
        "http://localhost:5173",  # Vite default
        "http://localhost:8501",  # Streamlit default
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8501",
        "*"  # Allow all (only for development!)
    ]
    
    # Model Settings
    BASE_DIR: Path = Path(__file__).parent.parent.parent  # Project root
    MODEL_PATH: Path = BASE_DIR / "models"
    MODEL_NAME: str = "xgboost_model"
    
    # API Settings
    API_PREFIX: str = "/api/v1"
    MAX_BATCH_SIZE: int = 100
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    def __init__(self):
        """Initialize and validate settings"""
        # Ensure model path exists
        if not self.MODEL_PATH.exists():
            print(f"⚠️  Warning: Model path does not exist: {self.MODEL_PATH}")
            print(f"📁 Creating models directory...")
            self.MODEL_PATH.mkdir(parents=True, exist_ok=True)

settings = Settings()