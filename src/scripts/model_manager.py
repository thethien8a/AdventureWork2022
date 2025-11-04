#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Manager - Lưu, Tải và Dự đoán với Mô hình ML
Author: AI Assistant
Date: 2025-11-03
"""

import joblib
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import boxcox
import warnings
warnings.filterwarnings('ignore')


class ModelManager:
    """
    Class quản lý việc lưu, tải và dự đoán với mô hình machine learning
    """
    
    def __init__(self, model_dir="../../models"):
        """
        Khởi tạo ModelManager
        
        Parameters:
        -----------
        model_dir : str
            Thư mục lưu trữ mô hình (relative to scripts folder)
        """
        # Get absolute path from current script location
        script_dir = Path(__file__).parent
        self.model_dir = (script_dir / model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.ohe = None
        self.fitted_lambda = None
        self.product_target_mean = None
        self.overall_mean = None
    
    
    def save_model_pickle(self, model, filename="model.pkl"):
        """
        Lưu mô hình bằng pickle
        
        Parameters:
        -----------
        model : object
            Mô hình đã được huấn luyện
        filename : str
            Tên file để lưu
        """
        filepath = self.model_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Đã lưu mô hình bằng pickle tại: {filepath}")
    
    
    def load_model_pickle(self, filename="model.pkl"):
        """
        Tải mô hình từ file pickle
        
        Parameters:
        -----------
        filename : str
            Tên file cần tải
            
        Returns:
        --------
        model : object
            Mô hình đã được tải
        """
        filepath = self.model_dir / filename
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Đã tải mô hình từ: {filepath}")
        return model
    
    
    def save_model_joblib(self, model, filename="model.joblib"):
        """
        Lưu mô hình bằng joblib (KHUYÊN DÙNG cho Scikit-learn)
        
        Parameters:
        -----------
        model : object
            Mô hình đã được huấn luyện
        filename : str
            Tên file để lưu
        """
        filepath = self.model_dir / filename
        joblib.dump(model, filepath)
        print(f"✅ Đã lưu mô hình bằng joblib tại: {filepath}")
    
    
    def load_model_joblib(self, filename="model.joblib"):
        """
        Tải mô hình từ file joblib
        
        Parameters:
        -----------
        filename : str
            Tên file cần tải
            
        Returns:
        --------
        model : object
            Mô hình đã được tải
        """
        filepath = self.model_dir / filename
        model = joblib.load(filepath)
        print(f"✅ Đã tải mô hình từ: {filepath}")
        return model
    
    
    def save_preprocessing_components(self, ohe, fitted_lambda, product_target_mean, 
                                     overall_mean, filename="preprocessing.joblib"):
        """
        Lưu các thành phần preprocessing (QUAN TRỌNG!)
        
        Parameters:
        -----------
        ohe : OneHotEncoder
            One-Hot Encoder đã được fit
        fitted_lambda : float
            Lambda value từ Box-Cox transformation
        product_target_mean : dict or Series
            Target encoding mean cho cột Name
        overall_mean : float
            Mean tổng thể của TotalDue
        filename : str
            Tên file để lưu
        """
        preprocessing_data = {
            'ohe': ohe,
            'fitted_lambda': fitted_lambda,
            'product_target_mean': product_target_mean,
            'overall_mean': overall_mean
        }
        
        filepath = self.model_dir / filename
        joblib.dump(preprocessing_data, filepath)
        print(f"✅ Đã lưu preprocessing components tại: {filepath}")
    
    
    def load_preprocessing_components(self, filename="preprocessing.joblib"):
        """
        Tải các thành phần preprocessing
        
        Parameters:
        -----------
        filename : str
            Tên file cần tải
            
        Returns:
        --------
        dict : Dictionary chứa các components
        """
        filepath = self.model_dir / filename
        preprocessing_data = joblib.load(filepath)
        
        self.ohe = preprocessing_data['ohe']
        self.fitted_lambda = preprocessing_data['fitted_lambda']
        self.product_target_mean = preprocessing_data['product_target_mean']
        self.overall_mean = preprocessing_data['overall_mean']
        
        print(f"✅ Đã tải preprocessing components từ: {filepath}")
        return preprocessing_data
    
    
    def save_complete_pipeline(self, model, ohe, fitted_lambda, product_target_mean, 
                              overall_mean, model_name="complete_pipeline"):
        """
        Lưu toàn bộ pipeline (mô hình + preprocessing)
        
        Parameters:
        -----------
        model : object
            Mô hình đã được huấn luyện
        ohe : OneHotEncoder
            One-Hot Encoder đã được fit
        fitted_lambda : float
            Lambda value từ Box-Cox transformation
        product_target_mean : dict or Series
            Target encoding mean cho cột Name
        overall_mean : float
            Mean tổng thể của TotalDue
        model_name : str
            Tên base cho các file
        """
        # Lưu mô hình
        self.save_model_joblib(model, f"{model_name}_model.joblib")
        
        # Lưu preprocessing
        self.save_preprocessing_components(
            ohe, fitted_lambda, product_target_mean, overall_mean,
            f"{model_name}_preprocessing.joblib"
        )
        
        print(f"\n🎉 Đã lưu toàn bộ pipeline với tên: {model_name}")
    
    
    def load_complete_pipeline(self, model_name="complete_pipeline"):
        """
        Tải toàn bộ pipeline (mô hình + preprocessing)
        
        Parameters:
        -----------
        model_name : str
            Tên base của pipeline
            
        Returns:
        --------
        tuple : (model, preprocessing_components)
        """
        # Tải mô hình
        self.model = self.load_model_joblib(f"{model_name}_model.joblib")
        
        # Tải preprocessing
        preprocessing = self.load_preprocessing_components(f"{model_name}_preprocessing.joblib")
        
        print(f"\n🎉 Đã tải toàn bộ pipeline: {model_name}")
        return self.model, preprocessing
    
    
    def preprocess_new_data(self, new_data_df):
        """
        Xử lý dữ liệu mới giống như dữ liệu training
        
        Parameters:
        -----------
        new_data_df : DataFrame
            Dữ liệu mới cần dự đoán (phải có các cột giống training)
            
        Returns:
        --------
        DataFrame : Dữ liệu đã được xử lý
        """
        if self.ohe is None or self.fitted_lambda is None:
            raise ValueError("⚠️ Chưa load preprocessing components! Hãy gọi load_complete_pipeline() trước.")
        
        df = new_data_df.copy()
        # 1. Feature Extraction: Extract date features
        if 'OrderDate' in df.columns:
            df["OrderDate"] = pd.to_datetime(df["OrderDate"])
            df["Year"] = df["OrderDate"].dt.year
            df["Month"] = df["OrderDate"].dt.month
            df["Day"] = df["OrderDate"].dt.day
            df.drop(columns=["OrderDate"], inplace=True)
        
        # 2. Fill missing values
        if "ProductLine" in df.columns:
            df["ProductLine"] = df["ProductLine"].fillna("Unidentified")
        
        # 3. Box-Cox transformation cho OrderQty
        if "OrderQty" in df.columns:
            df["OrderQty_boxcox"] = boxcox(df["OrderQty"], lmbda=self.fitted_lambda)
            df.drop(columns=["OrderQty"], inplace=True)
        
        # 4. One-Hot Encoding
        ohe_cols = ["PersonType", "ProductLine", "Name_territory", "CountryRegionCode", "Group"]
        available_ohe_cols = [col for col in ohe_cols if col in df.columns]
        
        if available_ohe_cols:
            encoded_array = self.ohe.transform(df[available_ohe_cols])
            encoded_columns = self.ohe.get_feature_names_out(input_features=available_ohe_cols)
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=encoded_columns,
                index=df.index
            )
            df = pd.concat([df.drop(columns=available_ohe_cols), encoded_df], axis=1)
        
        # 5. Target Encoding cho cột Name
        if "Name" in df.columns:
            df['Name_target_encoded'] = df['Name'].map(self.product_target_mean)
            df['Name_target_encoded'].fillna(self.overall_mean, inplace=True)
            df.drop(columns=["Name"], inplace=True)
        
        return df
    
    
    def predict(self, new_data_df):
        """
        Dự đoán với dữ liệu mới
        
        Parameters:
        -----------
        new_data_df : DataFrame
            Dữ liệu mới cần dự đoán
            
        Returns:
        --------
        array : Kết quả dự đoán
        """
        if self.model is None:
            raise ValueError("⚠️ Chưa load mô hình! Hãy gọi load_complete_pipeline() trước.")
        
        # Preprocess dữ liệu mới
        processed_data = self.preprocess_new_data(new_data_df)
        
        # Dự đoán
        predictions = self.model.predict(processed_data)
        
        print(f"✅ Đã dự đoán cho {len(predictions)} mẫu")
        return predictions
