
"""
Prohashing API Client

This module provides a client for interacting with the Prohashing.com API
to fetch mining pool data, profitability metrics, and worker performance.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Union
import requests
import asyncio
from urllib.parse import urljoin

from .base import BaseAPIClient
from .schemas import ProhashingPoolPerformance, ProhashingAlgorithmProfitability

logger = logging.getLogger(__name__)

class ProhashingClient(BaseAPIClient):
    """
    Client for interacting with the Prohashing.com API.
    
    This client provides methods for fetching mining pool data, profitability metrics,
    and worker performance from Prohashing.com.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://prohashing.com/api/v1",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the Prohashing API client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries
        )
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the Prohashing API.
        
        Returns:
            Dict containing the API status
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        response = self.get(
            endpoint="/status",
            headers=self.headers
        )
        
        return response.json()
    
    def get_profitability(self) -> Dict[str, Any]:
        """
        Get current profitability metrics for various algorithms.
        
        Returns:
            Dict containing profitability data for each algorithm
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        response = self.get(
            endpoint="/profitability",
            headers=self.headers
        )
        
        return response.json()
    
    def get_worker_performance(self, worker_id: str) -> Dict[str, Any]:
        """
        Get performance data for a specific worker.
        
        Args:
            worker_id: ID of the worker
        
        Returns:
            Dict containing worker performance data
        
        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If the worker ID is invalid
        """
        if not worker_id:
            raise ValueError("Worker ID is required")
        
        response = self.get(
            endpoint=f"/workers/{worker_id}",
            headers=self.headers
        )
        
        return response.json()
    
    def get_algorithm_profitability(self, algorithm_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get profitability data for a specific algorithm.
        
        Args:
            algorithm_id: ID of the algorithm (e.g., 1 for Scrypt, 2 for SHA-256)
        
        Returns:
            Dict containing algorithm profitability data
        
        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If the algorithm ID is invalid
        """
        profitability_data = self.get_profitability()
        
        algorithm_id_str = str(algorithm_id)
        if algorithm_id_str not in profitability_data:
            raise ValueError(f"Invalid algorithm ID: {algorithm_id}")
        
        algorithm_data = profitability_data[algorithm_id_str]
        
        # Validate and transform the data
        try:
            profitability = ProhashingAlgorithmProfitability(**algorithm_data)
            return profitability.transform()
        except Exception as e:
            logger.error(f"Error validating profitability data: {str(e)}")
            raise ValueError(f"Invalid profitability data: {str(e)}")
    
    def get_all_algorithm_profitability(self) -> List[Dict[str, Any]]:
        """
        Get profitability data for all available algorithms.
        
        Returns:
            List of dicts containing algorithm profitability data
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        profitability_data = self.get_profitability()
        
        result = []
        for algorithm_id, algorithm_data in profitability_data.items():
            try:
                profitability = ProhashingAlgorithmProfitability(**algorithm_data)
                result.append(profitability.transform())
            except Exception as e:
                logger.warning(f"Error validating profitability data for algorithm {algorithm_id}: {str(e)}")
                # Skip invalid data
        
        return result
    
    def get_worker_pool_performance(self, worker_id: str, algorithm: str = "Scrypt") -> Dict[str, Any]:
        """
        Get comprehensive pool performance data for a worker.
        
        This method fetches worker performance data and transforms it into the expected schema.
        
        Args:
            worker_id: ID of the worker
            algorithm: Mining algorithm (e.g., "Scrypt", "SHA-256")
        
        Returns:
            Dict containing the transformed pool performance data
        
        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If the worker ID is invalid
        """
        worker_data = self.get_worker_performance(worker_id)
        
        if "error" in worker_data:
            raise ValueError(f"Error fetching worker data: {worker_data['error']}")
        
        # Get algorithm profitability data
        algorithm_id = self._get_algorithm_id(algorithm)
        profitability_data = None
        
        try:
            profitability_data = self.get_algorithm_profitability(algorithm_id)
        except Exception as e:
            logger.warning(f"Error fetching profitability data: {str(e)}")
            # Continue without profitability data
        
        # Extract data from the response
        # Note: This is a simplified example and would need to be adjusted
        # based on the actual structure of the Prohashing API response
        
        # Prepare the pool performance data
        performance_data = {
            "timestamp": int(time.time()),
            "pool_id": "prohashing",
            "worker_id": worker_id,
            "algorithm": algorithm,
            "hashrate": {
                "reported": worker_data.get("hashrate", {}).get("reported", 0.0),
                "effective": worker_data.get("hashrate", {}).get("effective", 0.0),
                "unit": worker_data.get("hashrate", {}).get("unit", "MH/s")
            },
            "shares": {
                "accepted": worker_data.get("shares", {}).get("accepted", 0),
                "rejected": worker_data.get("shares", {}).get("rejected", 0),
                "stale": worker_data.get("shares", {}).get("stale", 0),
                "last_share_time": worker_data.get("last_share_time", int(time.time()))
            },
            "earnings": {
                "amount": worker_data.get("earnings", {}).get("amount", 0.0),
                "currency": worker_data.get("earnings", {}).get("currency", "USD"),
                "time_period": "24h"
            },
            "coins_mined": self._extract_coins_mined(worker_data),
            "profitability": {
                "per_hash_rate": profitability_data["profitability"]["current_usd"] if profitability_data else 0.0,
                "unit": "USD/TH/day",
                "time_period": "24h"
            },
            "difficulty": worker_data.get("difficulty", 0.0),
            "status": worker_data.get("status", "active")
        }
        
        # Validate and transform the data
        try:
            pool_performance = ProhashingPoolPerformance(**performance_data)
            return pool_performance.transform()
        except Exception as e:
            logger.error(f"Error validating pool performance data: {str(e)}")
            raise ValueError(f"Invalid pool performance data: {str(e)}")
    
    def _get_algorithm_id(self, algorithm_name: str) -> str:
        """
        Get the algorithm ID from its name.
        
        Args:
            algorithm_name: Name of the algorithm (e.g., "Scrypt", "SHA-256")
        
        Returns:
            Algorithm ID as a string
        """
        algorithm_map = {
            "Scrypt": "1",
            "SHA-256": "2",
            "X11": "4",
            # Add more algorithms as needed
        }
        
        return algorithm_map.get(algorithm_name, "1")  # Default to Scrypt
    
    def _extract_coins_mined(self, worker_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract coins mined from worker data.
        
        Args:
            worker_data: Worker performance data
        
        Returns:
            List of dicts containing coin mining data
        """
        coins_mined = []
        
        if "coins" in worker_data:
            for coin_data in worker_data.get("coins", []):
                coins_mined.append({
                    "coin_id": coin_data.get("id", ""),
                    "symbol": coin_data.get("symbol", ""),
                    "amount": coin_data.get("amount", 0.0),
                    "usd_value": coin_data.get("usd_value", 0.0),
                    "mining_type": coin_data.get("mining_type", "primary")
                })
        
        # If no coins data is available, add a placeholder for the primary coin
        if not coins_mined:
            coins_mined.append({
                "coin_id": "ltc",
                "symbol": "LTC",
                "amount": 0.0,
                "usd_value": 0.0,
                "mining_type": "primary"
            })
        
        return coins_mined
