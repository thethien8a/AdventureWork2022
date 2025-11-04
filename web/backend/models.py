#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class PredictionInput(BaseModel):
    """Input model for single prediction"""
    
    PersonType: str = Field(
        ...,
        description="Customer type",
        example="SC"
    )
    OrderQty: int = Field(
        ...,
        ge=1,
        le=1000,
        description="Order quantity",
        example=5
    )
    Name: str = Field(
        ...,
        description="Product name",
        example="Mountain-200 Black, 38"
    )
    ProductLine: str = Field(
        ...,
        description="Product line (M, R, T, S)",
        example="M"
    )
    Name_territory: str = Field(
        ...,
        description="Territory name",
        example="Southwest"
    )
    CountryRegionCode: str = Field(
        ...,
        description="Country code",
        example="US"
    )
    Group: str = Field(
        ...,
        description="Geographic group",
        example="North America"
    )
    OrderDate: str = Field(
        ...,
        description="Order date (YYYY-MM-DD)",
        example="2013-07-01"
    )
    
    @validator('PersonType')
    def validate_person_type(cls, v):
        valid_types = ['SC', 'IN', 'SP', 'EM', 'VC', 'GC']
        if v not in valid_types:
            raise ValueError(f'PersonType must be one of {valid_types}')
        return v
    
    @validator('ProductLine')
    def validate_product_line(cls, v):
        valid_lines = ['M', 'R', 'T', 'S']
        if v and v not in valid_lines:  # Allow None/empty
            raise ValueError(f'ProductLine must be one of {valid_lines}')
        return v
    
    @validator('OrderDate')
    def validate_order_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('OrderDate must be in YYYY-MM-DD format')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "PersonType": "SC",
                "OrderQty": 5,
                "Name": "Mountain-200 Black, 38",
                "ProductLine": "M",
                "Name_territory": "Southwest",
                "CountryRegionCode": "US",
                "Group": "North America",
                "OrderDate": "2013-07-01"
            }
        }


class PredictionOutput(BaseModel):
    """Output model for single prediction"""
    
    success: bool = Field(..., description="Whether prediction was successful")
    prediction: float = Field(..., description="Predicted revenue")
    input_data: Dict[str, Any] = Field(..., description="Input data used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_name: Optional[str] = Field(None, description="Model name used")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "prediction": 1234.56,
                "input_data": {
                    "PersonType": "SC",
                    "OrderQty": 5,
                    "Name": "Mountain-200 Black, 38"
                },
                "timestamp": "2025-11-04T10:30:00",
                "model_name": "xgboost_model"
            }
        }


class BatchPredictionInput(BaseModel):
    """Input model for batch prediction"""
    
    data: List[PredictionInput] = Field(
        ...,
        description="List of prediction inputs",
        min_items=1,
        max_items=100
    )
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class BatchPredictionOutput(BaseModel):
    """Output model for batch prediction"""
    
    success: bool = Field(..., description="Whether batch prediction was successful")
    total_records: int = Field(..., description="Total number of records processed")
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "total_records": 2,
                "predictions": [
                    {
                        "index": 0,
                        "prediction": 1234.56,
                        "input_data": {"PersonType": "SC", "OrderQty": 5}
                    },
                    {
                        "index": 1,
                        "prediction": 2468.12,
                        "input_data": {"PersonType": "IN", "OrderQty": 10}
                    }
                ],
                "timestamp": "2025-11-04T10:30:00"
            }
        }


class HealthResponse(BaseModel):
    """Health check response model"""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")
    timestamp: str = Field(..., description="Check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "xgboost_model",
                "timestamp": "2025-11-04T10:30:00"
            }
        }
