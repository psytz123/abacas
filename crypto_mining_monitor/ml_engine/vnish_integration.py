"""
Vnish Firmware Integration for ML Engine

This module provides integration between the ML recommendation engine and
Vnish firmware, allowing ML recommendations to be applied to miners.
"""

import logging
import time
import os
import json
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import ML engine components
from ml_engine.recommender import RecommendationEngine
from ml_engine.utils.logging_config import logger
from ml_engine.utils.validation import ValidationError

# Import API clients
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api_clients.vnish_firmware_client import VnishFirmwareClient
from api_clients.credential_store import VnishCredentialManager
from api_clients.config_validation import (
    validate_hashrate_tuning,
    validate_power_optimization,
    validate_overclocking,
    ValidationError as ConfigValidationError
)


class VnishIntegrationError(Exception):
    """Exception raised for Vnish integration errors."""
    pass


class VnishMLIntegration:
    """
    Integration between ML recommendation engine and Vnish firmware.
    
    This class provides methods for applying ML recommendations to miners
    running Vnish firmware, including hashrate tuning, power optimization,
    and overclocking.
    """
    
    def __init__(
        self,
        recommendation_engine: Optional[RecommendationEngine] = None,
        credential_manager: Optional[VnishCredentialManager] = None
    ):
        """
        Initialize the Vnish ML integration.
        
        Args:
            recommendation_engine: RecommendationEngine instance to use. If None, uses the global instance.
            credential_manager: VnishCredentialManager instance to use. If None, creates a new one.
        """
        self.recommendation_engine = recommendation_engine
        self.credential_manager = credential_manager or VnishCredentialManager()
        self.clients = {}  # Cache of VnishFirmwareClient instances
    
    def get_client(self, miner_ip: Optional[str] = None) -> VnishFirmwareClient:
        """
        Get a VnishFirmwareClient instance for a miner.
        
        Args:
            miner_ip: IP address of the miner. If None, uses default credentials.
        
        Returns:
            VnishFirmwareClient instance
        
        Raises:
            VnishIntegrationError: If credentials are not found
        """
        try:
            # Get credentials
            ip, username, password = self.credential_manager.get_credentials(miner_ip)
            
            # Check if we already have a client for this miner
            if ip in self.clients:
                return self.clients[ip]
            
            # Create a new client
            client = VnishFirmwareClient(
                miner_ip=ip,
                username=username,
                password=password
            )
            
            # Cache the client
            self.clients[ip] = client
            
            return client
        
        except Exception as e:
            logger.error(f"Error getting Vnish firmware client: {str(e)}")
            raise VnishIntegrationError(f"Error getting Vnish firmware client: {str(e)}")
    
    def apply_hashrate_tuning_recommendation(
        self,
        recommendation: Dict[str, Any],
        miner_ip: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Apply a hashrate tuning recommendation to a miner.
        
        Args:
            recommendation: Hashrate tuning recommendation dictionary
            miner_ip: IP address of the miner. If None, extracts from recommendation.
            dry_run: If True, only validates the recommendation without applying it
        
        Returns:
            Dictionary with the result of applying the recommendation
        
        Raises:
            VnishIntegrationError: If the recommendation cannot be applied
        """
        try:
            # Extract miner ID from recommendation if not provided
            if miner_ip is None:
                miner_id = recommendation.get("miner_id")
                if not miner_id:
                    raise VnishIntegrationError("Miner ID not found in recommendation")
                
                # Extract IP address from miner ID (assuming format like "miner_192.168.1.100")
                # In a real implementation, you would have a mapping from miner_id to IP
                miner_ip = miner_id.split("_")[1] if "_" in miner_id else miner_id
            
            # Extract hashrate percentage from recommendation
            hashrate_percent = recommendation.get("recommended_hashrate_percent")
            if hashrate_percent is None:
                raise VnishIntegrationError("Hashrate percentage not found in recommendation")
            
            # Validate the configuration
            try:
                validated_config = validate_hashrate_tuning({"hashrate_percent": hashrate_percent})
                hashrate_percent = validated_config["hashrate_percent"]
            except ConfigValidationError as e:
                raise VnishIntegrationError(f"Invalid hashrate tuning configuration: {str(e)}")
            
            # If this is a dry run, return without applying
            if dry_run:
                return {
                    "status": "success",
                    "message": f"Dry run: Would apply hashrate tuning of {hashrate_percent}% to miner {miner_ip}",
                    "recommendation_id": recommendation.get("id"),
                    "miner_ip": miner_ip,
                    "hashrate_percent": hashrate_percent,
                    "dry_run": True
                }
            
            # Get the client
            client = self.get_client(miner_ip)
            
            # Apply the recommendation
            result = client.apply_hashrate_tuning(hashrate_percent)
            
            # Log the result
            logger.info(f"Applied hashrate tuning recommendation to miner {miner_ip}: {hashrate_percent}%")
            
            # Return the result with additional information
            return {
                "status": result.get("status", "error"),
                "message": result.get("message", "Unknown error"),
                "recommendation_id": recommendation.get("id"),
                "miner_ip": miner_ip,
                "hashrate_percent": hashrate_percent,
                "original_frequency": result.get("original_frequency"),
                "target_frequency": result.get("target_frequency"),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error applying hashrate tuning recommendation: {str(e)}")
            raise VnishIntegrationError(f"Error applying hashrate tuning recommendation: {str(e)}")
    
    def apply_power_optimization_recommendation(
        self,
        recommendation: Dict[str, Any],
        miner_ip: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Apply a power optimization recommendation to a miner.
        
        Args:
            recommendation: Power optimization recommendation dictionary
            miner_ip: IP address of the miner. If None, extracts from recommendation.
            dry_run: If True, only validates the recommendation without applying it
        
        Returns:
            Dictionary with the result of applying the recommendation
        
        Raises:
            VnishIntegrationError: If the recommendation cannot be applied
        """
        try:
            # Extract miner ID from recommendation if not provided
            if miner_ip is None:
                miner_id = recommendation.get("miner_id")
                if not miner_id:
                    raise VnishIntegrationError("Miner ID not found in recommendation")
                
                # Extract IP address from miner ID (assuming format like "miner_192.168.1.100")
                # In a real implementation, you would have a mapping from miner_id to IP
                miner_ip = miner_id.split("_")[1] if "_" in miner_id else miner_id
            
            # Extract power limit factor from recommendation
            power_reduction_percent = recommendation.get("power_reduction_percent")
            if power_reduction_percent is None:
                raise VnishIntegrationError("Power reduction percentage not found in recommendation")
            
            # Convert power reduction percentage to power limit factor
            power_limit_factor = 1.0 - (power_reduction_percent / 100.0)
            
            # Validate the configuration
            try:
                validated_config = validate_power_optimization({"power_limit_factor": power_limit_factor})
                power_limit_factor = validated_config["power_limit_factor"]
            except ConfigValidationError as e:
                raise VnishIntegrationError(f"Invalid power optimization configuration: {str(e)}")
            
            # If this is a dry run, return without applying
            if dry_run:
                return {
                    "status": "success",
                    "message": f"Dry run: Would apply power limit factor of {power_limit_factor:.2f} to miner {miner_ip}",
                    "recommendation_id": recommendation.get("id"),
                    "miner_ip": miner_ip,
                    "power_limit_factor": power_limit_factor,
                    "power_reduction_percent": power_reduction_percent,
                    "dry_run": True
                }
            
            # Get the client
            client = self.get_client(miner_ip)
            
            # Apply the recommendation
            result = client.apply_power_optimization(power_limit_factor)
            
            # Log the result
            logger.info(f"Applied power optimization recommendation to miner {miner_ip}: {power_limit_factor:.2f}")
            
            # Return the result with additional information
            return {
                "status": result.get("status", "error"),
                "message": result.get("message", "Unknown error"),
                "recommendation_id": recommendation.get("id"),
                "miner_ip": miner_ip,
                "power_limit_factor": power_limit_factor,
                "power_reduction_percent": power_reduction_percent,
                "original_power": result.get("original_power"),
                "target_power": result.get("target_power"),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error applying power optimization recommendation: {str(e)}")
            raise VnishIntegrationError(f"Error applying power optimization recommendation: {str(e)}")
    
    def apply_overclocking_recommendation(
        self,
        recommendation: Dict[str, Any],
        miner_ip: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Apply an overclocking recommendation to a miner.
        
        Args:
            recommendation: Overclocking recommendation dictionary
            miner_ip: IP address of the miner. If None, extracts from recommendation.
            dry_run: If True, only validates the recommendation without applying it
        
        Returns:
            Dictionary with the result of applying the recommendation
        
        Raises:
            VnishIntegrationError: If the recommendation cannot be applied
        """
        try:
            # Extract miner ID from recommendation if not provided
            if miner_ip is None:
                miner_id = recommendation.get("miner_id")
                if not miner_id:
                    raise VnishIntegrationError("Miner ID not found in recommendation")
                
                # Extract IP address from miner ID (assuming format like "miner_192.168.1.100")
                # In a real implementation, you would have a mapping from miner_id to IP
                miner_ip = miner_id.split("_")[1] if "_" in miner_id else miner_id
            
            # Extract overclocking parameters from recommendation
            core_clock_offset = recommendation.get("core_clock_offset")
            memory_clock_offset = recommendation.get("memory_clock_offset")
            power_limit_percent = recommendation.get("power_limit_percent")
            core_voltage_offset = recommendation.get("core_voltage_offset")
            
            if any(param is None for param in [core_clock_offset, memory_clock_offset, power_limit_percent, core_voltage_offset]):
                raise VnishIntegrationError("Overclocking parameters not found in recommendation")
            
            # Validate the configuration
            try:
                validated_config = validate_overclocking({
                    "core_clock_offset": core_clock_offset,
                    "memory_clock_offset": memory_clock_offset,
                    "power_limit_percent": power_limit_percent,
                    "core_voltage_offset": core_voltage_offset
                })
                
                core_clock_offset = validated_config["core_clock_offset"]
                memory_clock_offset = validated_config["memory_clock_offset"]
                power_limit_percent = validated_config["power_limit_percent"]
                core_voltage_offset = validated_config["core_voltage_offset"]
            except ConfigValidationError as e:
                raise VnishIntegrationError(f"Invalid overclocking configuration: {str(e)}")
            
            # If this is a dry run, return without applying
            if dry_run:
                return {
                    "status": "success",
                    "message": f"Dry run: Would apply overclocking settings to miner {miner_ip}",
                    "recommendation_id": recommendation.get("id"),
                    "miner_ip": miner_ip,
                    "core_clock_offset": core_clock_offset,
                    "memory_clock_offset": memory_clock_offset,
                    "power_limit_percent": power_limit_percent,
                    "core_voltage_offset": core_voltage_offset,
                    "dry_run": True
                }
            
            # Get the client
            client = self.get_client(miner_ip)
            
            # Apply the recommendation
            result = client.apply_overclocking(
                core_clock_offset=core_clock_offset,
                memory_clock_offset=memory_clock_offset,
                power_limit_percent=power_limit_percent,
                core_voltage_offset=core_voltage_offset
            )
            
            # Log the result
            logger.info(f"Applied overclocking recommendation to miner {miner_ip}")
            
            # Return the result with additional information
            return {
                "status": result.get("status", "error"),
                "message": result.get("message", "Unknown error"),
                "recommendation_id": recommendation.get("id"),
                "miner_ip": miner_ip,
                "core_clock_offset": core_clock_offset,
                "memory_clock_offset": memory_clock_offset,
                "power_limit_percent": power_limit_percent,
                "core_voltage_offset": core_voltage_offset,
                "original_settings": result.get("original_settings"),
                "target_settings": result.get("target_settings"),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error applying overclocking recommendation: {str(e)}")
            raise VnishIntegrationError(f"Error applying overclocking recommendation: {str(e)}")
    
    def apply_recommendation(
        self,
        recommendation: Dict[str, Any],
        miner_ip: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Apply a recommendation to a miner.
        
        Args:
            recommendation: Recommendation dictionary
            miner_ip: IP address of the miner. If None, extracts from recommendation.
            dry_run: If True, only validates the recommendation without applying it
        
        Returns:
            Dictionary with the result of applying the recommendation
        
        Raises:
            VnishIntegrationError: If the recommendation cannot be applied
        """
        try:
            # Determine the type of recommendation
            recommendation_type = recommendation.get("type")
            
            if recommendation_type == "dynamic_hashrate_tuning":
                return self.apply_hashrate_tuning_recommendation(recommendation, miner_ip, dry_run)
            elif recommendation_type == "power_optimization":
                return self.apply_power_optimization_recommendation(recommendation, miner_ip, dry_run)
            elif recommendation_type == "intelligent_overclocking":
                return self.apply_overclocking_recommendation(recommendation, miner_ip, dry_run)
            else:
                raise VnishIntegrationError(f"Unsupported recommendation type: {recommendation_type}")
        
        except Exception as e:
            logger.error(f"Error applying recommendation: {str(e)}")
            raise VnishIntegrationError(f"Error applying recommendation: {str(e)}")
    
    def apply_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        miner_ip: Optional[str] = None,
        dry_run: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Apply multiple recommendations to a miner.
        
        Args:
            recommendations: List of recommendation dictionaries
            miner_ip: IP address of the miner. If None, extracts from recommendations.
            dry_run: If True, only validates the recommendations without applying them
        
        Returns:
            List of dictionaries with the results of applying the recommendations
        """
        results = []
        
        for recommendation in recommendations:
            try:
                result = self.apply_recommendation(recommendation, miner_ip, dry_run)
                results.append(result)
            except Exception as e:
                logger.error(f"Error applying recommendation: {str(e)}")
                results.append({
                    "status": "error",
                    "message": str(e),
                    "recommendation_id": recommendation.get("id"),
                    "miner_ip": miner_ip,
                    "timestamp": datetime.now().isoformat()
                })
        
        return results
    
    def get_miner_telemetry(self, miner_ip: str) -> Dict[str, Any]:
        """
        Get telemetry data from a miner.
        
        Args:
            miner_ip: IP address of the miner
        
        Returns:
            Dictionary with telemetry data
        
        Raises:
            VnishIntegrationError: If telemetry data cannot be retrieved
        """
        try:
            # Get the client
            client = self.get_client(miner_ip)
            
            # Get telemetry data
            telemetry = client.get_telemetry()
            
            return telemetry
        
        except Exception as e:
            logger.error(f"Error getting miner telemetry: {str(e)}")
            raise VnishIntegrationError(f"Error getting miner telemetry: {str(e)}")
    
    def get_miners_telemetry(self, miner_ips: List[str]) -> List[Dict[str, Any]]:
        """
        Get telemetry data from multiple miners.
        
        Args:
            miner_ips: List of miner IP addresses
        
        Returns:
            List of dictionaries with telemetry data
        """
        telemetry_data = []
        
        for miner_ip in miner_ips:
            try:
                telemetry = self.get_miner_telemetry(miner_ip)
                telemetry_data.append(telemetry)
            except Exception as e:
                logger.error(f"Error getting telemetry for miner {miner_ip}: {str(e)}")
        
        return telemetry_data
    
    def generate_and_apply_recommendations(
        self,
        miner_ips: List[str],
        recommendation_types: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Generate and apply recommendations for multiple miners.
        
        Args:
            miner_ips: List of miner IP addresses
            recommendation_types: List of recommendation types to generate and apply.
                If None, generates all types.
            dry_run: If True, only validates the recommendations without applying them
        
        Returns:
            Dictionary with the results of generating and applying recommendations
        
        Raises:
            VnishIntegrationError: If recommendations cannot be generated or applied
        """
        try:
            # Check if recommendation engine is available
            if self.recommendation_engine is None:
                raise VnishIntegrationError("Recommendation engine not available")
            
            # Get telemetry data from miners
            telemetry_data = self.get_miners_telemetry(miner_ips)
            
            if not telemetry_data:
                raise VnishIntegrationError("No telemetry data available")
            
            # Convert telemetry data to DataFrame
            telemetry_df = pd.DataFrame(telemetry_data)
            
            # Generate recommendations
            # In a real implementation, you would also need pool and market data
            # For simplicity, we'll use the telemetry data for all inputs
            recommendations = self.recommendation_engine.batch_generate_recommendations(telemetry_data)
            
            # Filter recommendations by type if specified
            if recommendation_types:
                filtered_recommendations = {}
                for rec_type, recs in recommendations.items():
                    if rec_type in recommendation_types:
                        filtered_recommendations[rec_type] = recs
                recommendations = filtered_recommendations
            
            # Apply recommendations
            results = {}
            for rec_type, recs in recommendations.items():
                if recs:
                    results[rec_type] = self.apply_recommendations(recs, dry_run=dry_run)
            
            return {
                "status": "success",
                "message": f"Generated and applied recommendations for {len(miner_ips)} miners",
                "recommendations": recommendations,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating and applying recommendations: {str(e)}")
            raise VnishIntegrationError(f"Error generating and applying recommendations: {str(e)}")
