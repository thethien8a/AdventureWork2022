from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging
import uvicorn
from config import settings
from models import PredictionInput, PredictionOutput, BatchPredictionInput, BatchPredictionOutput, HealthResponse

# Add project root to path
project_root = Path(__file__).parent / "../.."
sys.path.append(str(project_root / "src" / "scripts"))
from model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Modern replacement for deprecated @app.on_event()
    """
    # Startup: Code before yield runs on startup
    try:
        logger.info("🚀 Starting FastAPI server...")
        logger.info(f"📁 Model path: {settings.MODEL_PATH}")
        
        # Initialize model manager and store in app state
        app.state.model_manager = ModelManager(model_dir=str(settings.MODEL_PATH))
        app.state.model_manager.load_complete_pipeline(model_name=settings.MODEL_NAME)
        
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise
    
    yield  # Application is ready to handle requests
    
    # Shutdown: Code after yield runs on shutdown
    logger.info("🛑 Shutting down FastAPI server...")
    # Clean up resources if needed
    if hasattr(app.state, 'model_manager'):
        app.state.model_manager = None
    logger.info("✅ Shutdown complete!")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # Pass lifespan context manager
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AdventureWorks Revenue Prediction API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    model_manager = getattr(app.state, 'model_manager', None)
    is_healthy = model_manager is not None and model_manager.model is not None
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": is_healthy,
        "model_name": settings.MODEL_NAME if is_healthy else None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_revenue(input_data: PredictionInput):
    """
    Predict revenue for a single order
    
    - **PersonType**: Customer type (SC, IN, SP, EM, VC)
    - **OrderQty**: Order quantity (1-1000)
    - **Name**: Product name
    - **ProductLine**: Product line (M, R, T, S)
    - **Name_territory**: Territory name
    - **CountryRegionCode**: Country code (US, CA, FR, etc.)
    - **Group**: Geographic group
    - **OrderDate**: Order date (YYYY-MM-DD)
    """
    try:
        # Get model manager from app state
        model_manager = getattr(app.state, 'model_manager', None)
        
        # Validate model is loaded
        if model_manager is None or model_manager.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please check server logs."
            )
        
        logger.info(f"📊 Prediction request: {input_data.PersonType}, Qty: {input_data.OrderQty}")
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data.dict()])
        
        # Predict
        prediction = model_manager.predict(df)[0]
        
        logger.info(f"✅ Prediction successful: ${prediction:,.2f}")
        
        return {
            "success": True,
            "prediction": float(prediction),
            "input_data": input_data.dict(),
            "timestamp": datetime.now().isoformat(),
            "model_name": settings.MODEL_NAME
        }
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_revenue_batch(batch_input: BatchPredictionInput):
    """
    Predict revenue for multiple orders
    
    Accepts an array of prediction inputs and returns predictions for all
    """
    try:
        # Get model manager from app state
        model_manager = getattr(app.state, 'model_manager', None)
        
        # Validate model is loaded
        if model_manager is None or model_manager.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please check server logs."
            )
        
        logger.info(f"📊 Batch prediction request: {len(batch_input.data)} records")
        
        # Convert to DataFrame
        data_dicts = [item.dict() for item in batch_input.data]
        df = pd.DataFrame(data_dicts)
        
        # Predict
        predictions = model_manager.predict(df)
        
        # Format results
        results = []
        for i, (input_item, pred) in enumerate(zip(batch_input.data, predictions)):
            results.append({
                "index": i,
                "prediction": float(pred),
                "input_data": input_item.dict()
            })
        
        logger.info(f"✅ Batch prediction successful: {len(results)} records")
        
        return {
            "success": True,
            "total_records": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
