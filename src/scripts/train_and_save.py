#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training và Lưu Mô hình
Sử dụng code từ train.py và lưu mô hình để dùng sau này
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
# Train XGBoost model
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

# Import ModelManager
from model_manager import ModelManager


def load_and_prepare_data():
    """Tải và chuẩn bị dữ liệu giống như train.py"""
    
    print("📂 Đang tải dữ liệu...")
    
    # Get the project root directory (2 levels up from scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir / "../.."
    data_dir = project_root / "data"
    
    # Load data
    customer_df = pd.read_excel(data_dir / "Customer.xlsx")
    person_df = pd.read_excel(data_dir / "Person.xlsx")
    order_detail_df = pd.read_excel(data_dir / "OrderDetail.xlsx")
    product_df = pd.read_excel(data_dir / "Product.xlsx")
    territory_df = pd.read_excel(data_dir / "Territory.xlsx")
    order_header_df = pd.read_excel(data_dir / "OrderHeader.xlsx")
    
    # Merge data
    full_df = pd.merge(order_detail_df, order_header_df, on="SalesOrderID", how="left", suffixes=("_detail", "_header"))
    full_df = pd.merge(full_df, customer_df, on="CustomerID", how="left", suffixes=("", "_customer"))
    full_df = pd.merge(full_df, person_df, left_on="PersonID", right_on="BusinessEntityID", how="left", suffixes=("", "_person"))
    full_df = pd.merge(full_df, product_df, on="ProductID", how="left", suffixes=("", "_product"))
    full_df = pd.merge(full_df, territory_df, on="TerritoryID", how="left", suffixes=("", "_territory"))
    
    # Select columns
    full_df_copy = full_df[[
        "PersonType",
        "OrderQty",
        "Name", 
        "ProductLine", 
        "Class", 
        "Style",
        "Name_territory",
        "CountryRegionCode", 
        "Group",
        "TotalDue",
        "OrderDate"
    ]].copy()
    
    # Drop duplicates
    full_df_copy.drop_duplicates(inplace=True)
    
    # Drop Class và Style
    full_df_copy = full_df_copy.drop(columns=["Class", "Style"])
    
    # Fill missing values
    full_df_copy["ProductLine"] = full_df_copy["ProductLine"].fillna("Unidentified")
    
    print(f"✅ Đã tải {len(full_df_copy)} dòng dữ liệu")
    
    return full_df_copy


def train_model():
    """Huấn luyện mô hình và lưu lại"""
    
    print("\n" + "="*70)
    print("BƯỚC 1: TẢI VÀ CHUẨN BỊ DỮ LIỆU")
    print("="*70)
    
    # Load data
    full_df_copy = load_and_prepare_data()
    
    print("\n" + "="*70)
    print("BƯỚC 2: FEATURE ENGINEERING")
    print("="*70)
    
    # Feature Extraction
    full_df_copy["OrderDate"] = pd.to_datetime(full_df_copy["OrderDate"])
    full_df_copy["Year"] = full_df_copy["OrderDate"].dt.year
    full_df_copy["Month"] = full_df_copy["OrderDate"].dt.month
    full_df_copy["Day"] = full_df_copy["OrderDate"].dt.day
    full_df_copy.drop(columns=["OrderDate"], inplace=True)
    
    # Train-test split
    train_df, test_df = train_test_split(full_df_copy, test_size=0.2, random_state=42)
    
    print(f"📊 Train set: {len(train_df)} mẫu")
    print(f"📊 Test set: {len(test_df)} mẫu")
    
    # Box-Cox transformation
    train_df["OrderQty_boxcox"], fitted_lambda = boxcox(train_df["OrderQty"])
    test_df["OrderQty_boxcox"] = boxcox(test_df["OrderQty"], lmbda=fitted_lambda)
    print(f"📈 Box-Cox lambda: {fitted_lambda:.4f}")
    
    train_df.drop(columns=["OrderQty"], inplace=True)
    test_df.drop(columns=["OrderQty"], inplace=True)
    
    # One-Hot Encoding
    ohe_cols = ["PersonType", "ProductLine", "Name_territory", "CountryRegionCode", "Group"]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(train_df[ohe_cols])
    
    train_encoded_array = ohe.transform(train_df[ohe_cols])
    test_encoded_array = ohe.transform(test_df[ohe_cols])
    
    encoded_columns = ohe.get_feature_names_out(input_features=ohe_cols)
    
    train_encoded_df = pd.DataFrame(train_encoded_array, columns=encoded_columns, index=train_df.index)
    test_encoded_df = pd.DataFrame(test_encoded_array, columns=encoded_columns, index=test_df.index)
    
    train_df = pd.concat([train_df.drop(columns=ohe_cols), train_encoded_df], axis=1)
    test_df = pd.concat([test_df.drop(columns=ohe_cols), test_encoded_df], axis=1)
    
    print(f"🔢 One-Hot Encoding: {len(encoded_columns)} features")
    
    # Target Encoding
    product_target_mean = train_df.groupby('Name')['TotalDue'].mean()
    overall_mean = train_df['TotalDue'].mean()
    
    train_df['Name_target_encoded'] = train_df['Name'].map(product_target_mean)
    test_df['Name_target_encoded'] = test_df['Name'].map(product_target_mean)
    
    test_df['Name_target_encoded'].fillna(overall_mean, inplace=True)
    train_df['Name_target_encoded'].fillna(overall_mean, inplace=True)
    
    train_df.drop(columns=["Name"], inplace=True)
    test_df.drop(columns=["Name"], inplace=True)
    
    print(f"🎯 Target Encoding: Overall mean = {overall_mean:.2f}")
    
    print("\n" + "="*70)
    print("BƯỚC 3: HUẤN LUYỆN MÔ HÌNH XGBOOST")
    print("="*70)
    
    # Prepare X, y
    X_train, y_train = train_df.drop(columns=["TotalDue"]), train_df["TotalDue"]
    X_test, y_test = test_df.drop(columns=["TotalDue"]), test_df["TotalDue"]
    

    xgb_model = XGBRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    print("⏳ Đang huấn luyện mô hình XGBoost...")
    xgb_model.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = xgb_model.predict(X_train)
    y_test_pred = xgb_model.predict(X_test)
    
    print("\n📊 KẾT QUẢ TRAIN SET:")
    print(f"   MSE:  {mean_squared_error(y_train, y_train_pred):.2f}")
    print(f"   R2:   {r2_score(y_train, y_train_pred):.4f}")
    print(f"   RMSE: {root_mean_squared_error(y_train, y_train_pred):.2f}")
    
    print("\n📊 KẾT QUẢ TEST SET:")
    print(f"   MSE:  {mean_squared_error(y_test, y_test_pred):.2f}")
    print(f"   R2:   {r2_score(y_test, y_test_pred):.4f}")
    print(f"   RMSE: {root_mean_squared_error(y_test, y_test_pred):.2f}")
    
    print("\n" + "="*70)
    print("BƯỚC 4: LƯU MÔ HÌNH VÀ PREPROCESSING")
    print("="*70)
    
    # Lưu mô hình (default path: ../../models from scripts folder)
    manager = ModelManager()
    manager.save_complete_pipeline(
        model=xgb_model,
        ohe=ohe,
        fitted_lambda=fitted_lambda,
        product_target_mean=product_target_mean,
        overall_mean=overall_mean,
        model_name="xgboost_model"
    )
    
    print("\n✅ HOÀN THÀNH! Mô hình đã được lưu trong thư mục 'models/'")
    print("💡 Sử dụng file predict_new_data.py để dự đoán với dữ liệu mới")
    

if __name__ == "__main__":
    train_model()
