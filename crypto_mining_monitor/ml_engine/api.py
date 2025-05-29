"""
API for the ML recommendation engine.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import json

from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ml_engine.api_vnish_endpoints import router as vnish_router

from ml_engine.feature_engineering import FeatureEngineeringPipeline, MockDataGenerator
from ml_engine.recommender import RecommendationEngine
from ml_engine.config import API_CONFIG
from ml_engine.utils.constants import MODEL_DIR
from ml_engine.utils.logging_config import logger
from ml_engine.utils.validation import ValidationError


# Define API models
class MinerTelemetry(BaseModel):
    miner_id: str
    timestamp: str
    hashrate_th_s: float
    power_consumption_w: float
    avg_chip_temp_c: float
    max_chip_temp_c: Optional[float] = None
    fan_speed_percent: Optional[float] = None
    accepted_shares: Optional[int] = None
    rejected_shares: Optional[int] = None
    overclock_profile: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "miner_id": "miner_001",
                "timestamp": "2025-05-20T12:00:00",
                "hashrate_th_s": 95.5,
                "power_consumption_w": 3200,
                "avg_chip_temp_c": 65.2,
                "max_chip_temp_c": 68.7,
                "fan_speed_percent": 70.0,
                "accepted_shares": 1250,
                "rejected_shares": 5,
                "overclock_profile": "normal"
            }
        }


class PoolPerformance(BaseModel):
    worker_id: str
    timestamp: str
    effective_hashrate_th_s: float
    hashrate_th_s: Optional[float] = None
    earnings_usd_24h: float
    primary_coin: str
    merge_mining_enabled: Optional[bool] = False
    
    class Config:
        schema_extra = {
            "example": {
                "worker_id": "worker_001",
                "timestamp": "2025-05-20T12:00:00",
                "effective_hashrate_th_s": 92.3,
                "hashrate_th_s": 95.5,
                "earnings_usd_24h": 12.75,
                "primary_coin": "BTC",
                "merge_mining_enabled": False
            }
        }


class MarketData(BaseModel):
    coin_id: str
    timestamp: str
    price_usd: float
    market_cap_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    price_change_24h_percent: Optional[float] = None
    network_difficulty: float
    block_reward_usd: float
    transaction_fees_percent: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "coin_id": "BTC",
                "timestamp": "2025-05-20T12:00:00",
                "price_usd": 55000.0,
                "market_cap_usd": 1050000000000.0,
                "volume_24h_usd": 25000000000.0,
                "price_change_24h_percent": 2.5,
                "network_difficulty": 30000000000000.0,
                "block_reward_usd": 343750.0,
                "transaction_fees_percent": 10.0
            }
        }


class UserPreferences(BaseModel):
    min_improvement_threshold: Optional[float] = 0.05
    confidence_threshold: Optional[float] = 0.7
    cooldown_period_hours: Optional[int] = 12
    min_efficiency_improvement: Optional[float] = 0.03
    max_hashrate_reduction: Optional[float] = 0.05
    
    class Config:
        schema_extra = {
            "example": {
                "min_improvement_threshold": 0.05,
                "confidence_threshold": 0.7,
                "cooldown_period_hours": 12,
                "min_efficiency_improvement": 0.03,
                "max_hashrate_reduction": 0.05
            }
        }


class RecommendationRequest(BaseModel):
    miner_telemetry: List[MinerTelemetry]
    pool_performance: List[PoolPerformance]
    market_data: List[MarketData]
    user_preferences: Optional[UserPreferences] = None
    
    class Config:
        schema_extra = {
            "example": {
                "miner_telemetry": [
                    {
                        "miner_id": "miner_001",
                        "timestamp": "2025-05-20T12:00:00",
                        "hashrate_th_s": 95.5,
                        "power_consumption_w": 3200,
                        "avg_chip_temp_c": 65.2,
                        "max_chip_temp_c": 68.7,
                        "fan_speed_percent": 70.0,
                        "accepted_shares": 1250,
                        "rejected_shares": 5,
                        "overclock_profile": "normal"
                    }
                ],
                "pool_performance": [
                    {
                        "worker_id": "worker_001",
                        "timestamp": "2025-05-20T12:00:00",
                        "effective_hashrate_th_s": 92.3,
                        "hashrate_th_s": 95.5,
                        "earnings_usd_24h": 12.75,
                        "primary_coin": "BTC",
                        "merge_mining_enabled": False
                    }
                ],
                "market_data": [
                    {
                        "coin_id": "BTC",
                        "timestamp": "2025-05-20T12:00:00",
                        "price_usd": 55000.0,
                        "network_difficulty": 30000000000000.0,
                        "block_reward_usd": 343750.0
                    },
                    {
                        "coin_id": "ETH",
                        "timestamp": "2025-05-20T12:00:00",
                        "price_usd": 2500.0,
                        "network_difficulty": 10000000000000.0,
                        "block_reward_usd": 5000.0
                    }
                ],
                "user_preferences": {
                    "min_improvement_threshold": 0.05,
                    "confidence_threshold": 0.7,
                    "cooldown_period_hours": 12
                }
            }
        }


class RecommendationResponse(BaseModel):
    recommendations: Dict[str, List[Dict[str, Any]]]
    timestamp: str
    request_id: str


class FeedbackRequest(BaseModel):
    recommendation_id: str
    feedback_type: str = Field(..., description="One of: accepted, rejected, implemented, not_implemented")
    details: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "recommendation_id": "123e4567-e89b-12d3-a456-426614174000",
                "feedback_type": "implemented",
                "details": "Implemented the recommendation and saw a 4.8% improvement in profitability."
            }
        }


# Initialize the API
app = FastAPI(
    title="Crypto Mining ML Recommendation API",
    description="API for generating ML-based recommendations for cryptocurrency mining optimization",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the Vnish router
app.include_router(vnish_router)

# Global variables
feature_pipeline = FeatureEngineeringPipeline()
recommendation_engine = None
mock_data_generator = None
feedback_store = {}


# Dependency to get the recommendation engine
def get_recommendation_engine():
    global recommendation_engine
    if recommendation_engine is None:
        # Find the latest model files
        profit_model_path = None
        power_model_path = None
        
        if os.path.exists(MODEL_DIR):
            # Find latest profit model
            profit_models = [f for f in os.listdir(MODEL_DIR) if f.startswith('profit_model_') and f.endswith('.joblib')]
            if profit_models:
                profit_models.sort(reverse=True)
                profit_model_path = os.path.join(MODEL_DIR, profit_models[0])
                logger.info(f"Found profit model: {profit_model_path}")
            
            # Find latest power model
            power_models = [f for f in os.listdir(MODEL_DIR) if f.startswith('power_optimizer_') and f.endswith('.joblib')]
            if power_models:
                power_models.sort(reverse=True)
                power_model_path = os.path.join(MODEL_DIR, power_models[0])
                logger.info(f"Found power model: {power_model_path}")
        
        # Initialize the recommendation engine
        recommendation_engine = RecommendationEngine(profit_model_path, power_model_path)
    
    return recommendation_engine


# Dependency to get mock data generator
def get_mock_data_generator():
    global mock_data_generator
    if mock_data_generator is None:
        mock_data_generator = MockDataGenerator(num_miners=5, num_coins=3, days=7, interval_minutes=60)
    
    return mock_data_generator


@app.get("/")
def read_root():
    """API root endpoint."""
    return {"message": "Crypto Mining ML Recommendation API", "version": "0.1.0"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/recommendations", response_model=RecommendationResponse)
def generate_recommendations(
    request: RecommendationRequest,
    recommendation_engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Generate recommendations based on provided mining telemetry, pool performance, and market data.
    """
    try:
        logger.info(f"Received recommendation request with {len(request.miner_telemetry)} miners, {len(request.pool_performance)} pool records, {len(request.market_data)} market records")
        
        # Convert request data to pandas DataFrames
        miner_df = pd.DataFrame([m.dict() for m in request.miner_telemetry])
        pool_df = pd.DataFrame([p.dict() for p in request.pool_performance])
        market_df = pd.DataFrame([m.dict() for m in request.market_data])
        
        # Process features
        try:
            processed_miner = feature_pipeline.process_miner_telemetry(miner_df)
            processed_pool = feature_pipeline.process_pool_performance(pool_df)
            processed_market = feature_pipeline.process_market_data(market_df)
        except ValidationError as e:
            logger.warning(f"Validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid request data: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing features: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing features: {str(e)}")
        
        # Convert user preferences
        user_prefs = request.user_preferences.dict() if request.user_preferences else None
        
        # Generate recommendations
        try:
            recommendations = recommendation_engine.generate_all_recommendations(
                processed_miner, processed_pool, processed_market, user_prefs
            )
            
            # Generate dynamic hashrate tuning recommendations if model is loaded
            if recommendation_engine.hashrate_tuner is not None:
                try:
                    hashrate_recommendations = recommendation_engine.generate_dynamic_hashrate_recommendations(
                        processed_miner, processed_pool, processed_market, None, user_prefs
                    )
                    recommendations['dynamic_hashrate_tuning'] = hashrate_recommendations
                    logger.info(f"Generated {len(hashrate_recommendations)} dynamic hashrate tuning recommendations")
                except Exception as e:
                    logger.error(f"Failed to generate dynamic hashrate tuning recommendations: {str(e)}")
                    recommendations['dynamic_hashrate_tuning'] = []
            
            # Generate intelligent overclocking recommendations if model is loaded
            if recommendation_engine.overclocker is not None:
                try:
                    overclocking_recommendations = recommendation_engine.generate_intelligent_overclocking_recommendations(
                        processed_miner, processed_pool, processed_market, user_prefs
                    )
                    recommendations['intelligent_overclocking'] = overclocking_recommendations
                    logger.info(f"Generated {len(overclocking_recommendations)} intelligent overclocking recommendations")
                except Exception as e:
                    logger.error(f"Failed to generate intelligent overclocking recommendations: {str(e)}")
                    recommendations['intelligent_overclocking'] = []
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
        
        # Create response
        request_id = str(uuid.uuid4())
        response = {
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
        
        logger.info(f"Generated recommendations with request_id: {request_id}")
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.post("/recommendations/async", status_code=202)
async def generate_recommendations_async(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    recommendation_engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Generate recommendations asynchronously.
    """
    try:
        # Generate a request ID
        request_id = str(uuid.uuid4())
        
        # Add the task to the background
        background_tasks.add_task(
            process_recommendation_request,
            request_id,
            request,
            recommendation_engine
        )
        
        # Return immediate response
        return {
            "request_id": request_id,
            "status": "processing",
            "message": "Recommendation request is being processed",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error starting async recommendation task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting async recommendation task: {str(e)}")


# Background task function
async def process_recommendation_request(request_id: str, request: RecommendationRequest, recommendation_engine: RecommendationEngine):
    """Process recommendation request in the background."""
    try:
        # Convert request data to pandas DataFrames
        miner_df = pd.DataFrame([m.dict() for m in request.miner_telemetry])
        pool_df = pd.DataFrame([p.dict() for p in request.pool_performance])
        market_df = pd.DataFrame([m.dict() for m in request.market_data])
        
        # Process features
        processed_miner = feature_pipeline.process_miner_telemetry(miner_df)
        processed_pool = feature_pipeline.process_pool_performance(pool_df)
        processed_market = feature_pipeline.process_market_data(market_df)
        
        # Convert user preferences
        user_prefs = request.user_preferences.dict() if request.user_preferences else None
        
        # Generate recommendations
        recommendations = recommendation_engine.generate_all_recommendations(
            processed_miner, processed_pool, processed_market, user_prefs
        )
        
        # Generate dynamic hashrate tuning recommendations if model is loaded
        if recommendation_engine.hashrate_tuner is not None:
            try:
                hashrate_recommendations = recommendation_engine.generate_dynamic_hashrate_recommendations(
                    processed_miner, processed_pool, processed_market, None, user_prefs
                )
                recommendations['dynamic_hashrate_tuning'] = hashrate_recommendations
                logger.info(f"Generated {len(hashrate_recommendations)} dynamic hashrate tuning recommendations")
            except Exception as e:
                logger.error(f"Failed to generate dynamic hashrate tuning recommendations: {str(e)}")
                recommendations['dynamic_hashrate_tuning'] = []
        
        # Generate intelligent overclocking recommendations if model is loaded
        if recommendation_engine.overclocker is not None:
            try:
                overclocking_recommendations = recommendation_engine.generate_intelligent_overclocking_recommendations(
                    processed_miner, processed_pool, processed_market, user_prefs
                )
                recommendations['intelligent_overclocking'] = overclocking_recommendations
                logger.info(f"Generated {len(overclocking_recommendations)} intelligent overclocking recommendations")
            except Exception as e:
                logger.error(f"Failed to generate intelligent overclocking recommendations: {str(e)}")
                recommendations['intelligent_overclocking'] = []
        
        # Store results
        store_recommendation_results(request_id, recommendations)
        
        logger.info(f"Completed async recommendation request: {request_id}")
    
    except Exception as e:
        logger.error(f"Error processing async recommendation request: {str(e)}", exc_info=True)
        store_recommendation_error(request_id, str(e))


# In-memory storage for async results (in production, use a database)
async_results = {}

def store_recommendation_results(request_id: str, recommendations: Dict):
    """Store recommendation results."""
    async_results[request_id] = {
        "status": "completed",
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat()
    }

def store_recommendation_error(request_id: str, error: str):
    """Store recommendation error."""
    async_results[request_id] = {
        "status": "failed",
        "error": error,
        "timestamp": datetime.now().isoformat()
    }

def get_request_status(request_id: str):
    """Get status of an asynchronous request."""
    if request_id not in async_results:
        return {
            "status": "not_found",
            "message": f"Request ID {request_id} not found",
            "timestamp": datetime.now().isoformat()
        }
    
    return async_results[request_id]


@app.get("/recommendations/{request_id}")
async def get_recommendation_status(request_id: str):
    """Get status and results of an asynchronous recommendation request."""
    status = get_request_status(request_id)
    
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Request ID {request_id} not found")
    
    return status


@app.get("/mock-data")
def get_mock_data(
    mock_data_generator: MockDataGenerator = Depends(get_mock_data_generator)
):
    """
    Generate mock data for testing the recommendation engine.
    """
    try:
        # Generate mock data
        miner_data, pool_data, market_data = mock_data_generator.generate_all_data()
        
        # Convert to dictionaries for JSON response
        miner_records = miner_data.to_dict(orient='records')
        pool_records = pool_data.to_dict(orient='records')
        market_records = market_data.to_dict(orient='records')
        
        logger.info(f"Generated mock data: {len(miner_records)} miner records, {len(pool_records)} pool records, {len(market_records)} market records")
        
        return {
            "miner_telemetry": miner_records,
            "pool_performance": pool_records,
            "market_data": market_records
        }
    
    except Exception as e:
        logger.error(f"Error generating mock data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating mock data: {str(e)}")


@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback on a recommendation.
    """
    try:
        # Validate feedback type
        valid_feedback_types = ["accepted", "rejected", "implemented", "not_implemented"]
        if feedback.feedback_type not in valid_feedback_types:
            raise ValidationError(f"Invalid feedback type. Must be one of: {', '.join(valid_feedback_types)}")
        
        # Store feedback
        feedback_id = str(uuid.uuid4())
        feedback_data = {
            "id": feedback_id,
            "recommendation_id": feedback.recommendation_id,
            "feedback_type": feedback.feedback_type,
            "details": feedback.details,
            "timestamp": datetime.now().isoformat()
        }
        
        feedback_store[feedback_id] = feedback_data
        
        logger.info(f"Received feedback for recommendation {feedback.recommendation_id}: {feedback.feedback_type}")
        
        return {
            "message": "Feedback received successfully",
            "feedback_id": feedback_id
        }
    
    except ValidationError as e:
        logger.warning(f"Validation error in feedback: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")


@app.get("/models")
def list_models():
    """
    List available ML models.
    """
    try:
        models = []
        
        if os.path.exists(MODEL_DIR):
            for filename in os.listdir(MODEL_DIR):
                if filename.endswith('.joblib'):
                    file_path = os.path.join(MODEL_DIR, filename)
                    file_stats = os.stat(file_path)
                    
                    model_type = "unknown"
                    if filename.startswith('profit_model_'):
                        model_type = "profit_prediction"
                    elif filename.startswith('power_optimizer_'):
                        model_type = "power_optimization"
                    
                    models.append({
                        "filename": filename,
                        "type": model_type,
                        "size_bytes": file_stats.st_size,
                        "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                    })
        
        logger.info(f"Listed {len(models)} available models")
        return {"models": models}
    
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


def start_server():
    """Start the API server."""
    logger.info(f"Starting API server on {API_CONFIG['host']}:{API_CONFIG['port']}")
    uvicorn.run(
        "ml_engine.api:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"]
    )


if __name__ == "__main__":
    start_server()
