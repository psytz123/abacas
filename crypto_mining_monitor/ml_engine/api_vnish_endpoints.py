"""
API endpoints for Vnish firmware integration.

This module provides FastAPI endpoints for integrating the ML recommendation engine
with Vnish firmware, allowing ML recommendations to be applied to miners.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import json

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, Body
from pydantic import BaseModel, Field

from ml_engine.vnish_integration import VnishMLIntegration, VnishIntegrationError
from ml_engine.recommender import RecommendationEngine
from ml_engine.utils.logging_config import logger

# Import API clients
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_clients.credential_store import VnishCredentialManager


# Define API models
class MinerIdentifier(BaseModel):
    miner_id: str
    miner_ip: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "miner_id": "miner_001",
                "miner_ip": "192.168.1.101"
            }
        }


class VnishCredentials(BaseModel):
    miner_ip: str
    username: str
    password: str
    is_default: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "miner_ip": "192.168.1.101",
                "username": "admin",
                "password": "admin",
                "is_default": False
            }
        }


class HashrateTuningRequest(BaseModel):
    miner_id: str
    miner_ip: Optional[str] = None
    hashrate_percent: float = Field(..., ge=50.0, le=100.0)
    dry_run: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "miner_id": "miner_001",
                "miner_ip": "192.168.1.101",
                "hashrate_percent": 80.0,
                "dry_run": False
            }
        }


class PowerOptimizationRequest(BaseModel):
    miner_id: str
    miner_ip: Optional[str] = None
    power_limit_factor: float = Field(..., gt=0.0, le=1.0)
    dry_run: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "miner_id": "miner_001",
                "miner_ip": "192.168.1.101",
                "power_limit_factor": 0.85,
                "dry_run": False
            }
        }


class OverclockingRequest(BaseModel):
    miner_id: str
    miner_ip: Optional[str] = None
    core_clock_offset: int = Field(..., ge=-100, le=200)
    memory_clock_offset: int = Field(..., ge=-500, le=1500)
    power_limit_percent: float = Field(..., ge=70.0, le=100.0)
    core_voltage_offset: int = Field(..., ge=-50, le=50)
    dry_run: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "miner_id": "miner_001",
                "miner_ip": "192.168.1.101",
                "core_clock_offset": 50,
                "memory_clock_offset": 200,
                "power_limit_percent": 85.0,
                "core_voltage_offset": 10,
                "dry_run": False
            }
        }


class ApplyRecommendationRequest(BaseModel):
    recommendation_id: str
    miner_ip: Optional[str] = None
    dry_run: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "recommendation_id": "123e4567-e89b-12d3-a456-426614174000",
                "miner_ip": "192.168.1.101",
                "dry_run": False
            }
        }


class GenerateAndApplyRequest(BaseModel):
    miner_ips: List[str]
    recommendation_types: Optional[List[str]] = None
    dry_run: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "miner_ips": ["192.168.1.101", "192.168.1.102"],
                "recommendation_types": ["power_optimization", "dynamic_hashrate_tuning"],
                "dry_run": False
            }
        }


# Create router
router = APIRouter(
    prefix="/vnish",
    tags=["vnish"],
    responses={404: {"description": "Not found"}},
)


# Dependency to get the Vnish ML integration
def get_vnish_integration(recommendation_engine: RecommendationEngine = Depends()):
    return VnishMLIntegration(recommendation_engine=recommendation_engine)


@router.post("/credentials", response_model=Dict[str, Any])
async def save_vnish_credentials(
    credentials: VnishCredentials,
    credential_manager: VnishCredentialManager = Depends()
):
    """
    Save Vnish firmware credentials.
    """
    try:
        success = credential_manager.save_credentials(
            miner_ip=credentials.miner_ip,
            username=credentials.username,
            password=credentials.password,
            is_default=credentials.is_default
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Credentials saved for miner {credentials.miner_ip}",
                "is_default": credentials.is_default
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save credentials")
    
    except Exception as e:
        logger.error(f"Error saving Vnish credentials: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving Vnish credentials: {str(e)}")


@router.delete("/credentials/{miner_ip}", response_model=Dict[str, Any])
async def delete_vnish_credentials(
    miner_ip: str,
    credential_manager: VnishCredentialManager = Depends()
):
    """
    Delete Vnish firmware credentials.
    """
    try:
        success = credential_manager.delete_credentials(miner_ip)
        
        if success:
            return {
                "status": "success",
                "message": f"Credentials deleted for miner {miner_ip}"
            }
        else:
            raise HTTPException(status_code=404, detail=f"No credentials found for miner {miner_ip}")
    
    except Exception as e:
        logger.error(f"Error deleting Vnish credentials: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting Vnish credentials: {str(e)}")


@router.get("/telemetry/{miner_ip}", response_model=Dict[str, Any])
async def get_miner_telemetry(
    miner_ip: str,
    vnish_integration: VnishMLIntegration = Depends(get_vnish_integration)
):
    """
    Get telemetry data from a miner.
    """
    try:
        telemetry = vnish_integration.get_miner_telemetry(miner_ip)
        
        return {
            "status": "success",
            "miner_ip": miner_ip,
            "telemetry": telemetry,
            "timestamp": datetime.now().isoformat()
        }
    
    except VnishIntegrationError as e:
        logger.error(f"Vnish integration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting miner telemetry: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting miner telemetry: {str(e)}")


@router.post("/hashrate-tuning", response_model=Dict[str, Any])
async def apply_hashrate_tuning(
    request: HashrateTuningRequest,
    vnish_integration: VnishMLIntegration = Depends(get_vnish_integration)
):
    """
    Apply hashrate tuning to a miner.
    """
    try:
        # Create a mock recommendation
        recommendation = {
            "id": str(uuid.uuid4()),
            "type": "dynamic_hashrate_tuning",
            "miner_id": request.miner_id,
            "recommended_hashrate_percent": request.hashrate_percent
        }
        
        result = vnish_integration.apply_hashrate_tuning_recommendation(
            recommendation=recommendation,
            miner_ip=request.miner_ip,
            dry_run=request.dry_run
        )
        
        return result
    
    except VnishIntegrationError as e:
        logger.error(f"Vnish integration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error applying hashrate tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error applying hashrate tuning: {str(e)}")


@router.post("/power-optimization", response_model=Dict[str, Any])
async def apply_power_optimization(
    request: PowerOptimizationRequest,
    vnish_integration: VnishMLIntegration = Depends(get_vnish_integration)
):
    """
    Apply power optimization to a miner.
    """
    try:
        # Calculate power reduction percentage
        power_reduction_percent = (1.0 - request.power_limit_factor) * 100.0
        
        # Create a mock recommendation
        recommendation = {
            "id": str(uuid.uuid4()),
            "type": "power_optimization",
            "miner_id": request.miner_id,
            "power_reduction_percent": power_reduction_percent
        }
        
        result = vnish_integration.apply_power_optimization_recommendation(
            recommendation=recommendation,
            miner_ip=request.miner_ip,
            dry_run=request.dry_run
        )
        
        return result
    
    except VnishIntegrationError as e:
        logger.error(f"Vnish integration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error applying power optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error applying power optimization: {str(e)}")


@router.post("/overclocking", response_model=Dict[str, Any])
async def apply_overclocking(
    request: OverclockingRequest,
    vnish_integration: VnishMLIntegration = Depends(get_vnish_integration)
):
    """
    Apply overclocking to a miner.
    """
    try:
        # Create a mock recommendation
        recommendation = {
            "id": str(uuid.uuid4()),
            "type": "intelligent_overclocking",
            "miner_id": request.miner_id,
            "core_clock_offset": request.core_clock_offset,
            "memory_clock_offset": request.memory_clock_offset,
            "power_limit_percent": request.power_limit_percent,
            "core_voltage_offset": request.core_voltage_offset
        }
        
        result = vnish_integration.apply_overclocking_recommendation(
            recommendation=recommendation,
            miner_ip=request.miner_ip,
            dry_run=request.dry_run
        )
        
        return result
    
    except VnishIntegrationError as e:
        logger.error(f"Vnish integration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error applying overclocking: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error applying overclocking: {str(e)}")


@router.post("/apply-recommendation", response_model=Dict[str, Any])
async def apply_recommendation(
    request: ApplyRecommendationRequest,
    recommendation_store: Dict[str, Any] = Depends(),
    vnish_integration: VnishMLIntegration = Depends(get_vnish_integration)
):
    """
    Apply a specific recommendation to a miner.
    """
    try:
        # In a real implementation, you would retrieve the recommendation from a database
        # For simplicity, we'll use a mock recommendation store
        recommendation = recommendation_store.get(request.recommendation_id)
        
        if not recommendation:
            raise HTTPException(status_code=404, detail=f"Recommendation {request.recommendation_id} not found")
        
        result = vnish_integration.apply_recommendation(
            recommendation=recommendation,
            miner_ip=request.miner_ip,
            dry_run=request.dry_run
        )
        
        return result
    
    except VnishIntegrationError as e:
        logger.error(f"Vnish integration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error applying recommendation: {str(e)}")


@router.post("/generate-and-apply", response_model=Dict[str, Any])
async def generate_and_apply_recommendations(
    request: GenerateAndApplyRequest,
    vnish_integration: VnishMLIntegration = Depends(get_vnish_integration)
):
    """
    Generate and apply recommendations for multiple miners.
    """
    try:
        result = vnish_integration.generate_and_apply_recommendations(
            miner_ips=request.miner_ips,
            recommendation_types=request.recommendation_types,
            dry_run=request.dry_run
        )
        
        return result
    
    except VnishIntegrationError as e:
        logger.error(f"Vnish integration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating and applying recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating and applying recommendations: {str(e)}")
