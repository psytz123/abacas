
"""
Vnish Firmware API Client

This module provides a client for interacting with the Vnish firmware API
to fetch telemetry data from ASIC miners.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
import requests
from requests.auth import HTTPBasicAuth

from .base import BaseAPIClient
from .schemas import VnishMinerTelemetry

logger = logging.getLogger(__name__)

class VnishClient(BaseAPIClient):
    """
    Client for interacting with the Vnish firmware API.
    
    This client provides methods for fetching telemetry data from ASIC miners
    running Vnish firmware, including hashrate, temperature, power consumption,
    and operational status.
    """
    
    def __init__(
        self,
        miner_ip: str,
        username: str,
        password: str,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the Vnish API client.
        
        Args:
            miner_ip: IP address of the miner
            username: Username for authentication
            password: Password for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        base_url = f"http://{miner_ip}"
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries
        )
        self.auth = HTTPBasicAuth(username, password)
        self.miner_ip = miner_ip
        self.miner_id = None  # Will be populated from the first API call
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get real-time statistics from the miner.
        
        Returns:
            Dict containing the miner statistics
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        response = self.get(
            endpoint="/cgi-bin/stats.cgi",
            auth=self.auth
        )
        
        stats = response.json()
        
        # Store the miner ID for future reference
        if self.miner_id is None and "data" in stats and "miner_id" in stats["data"]:
            self.miner_id = stats["data"]["miner_id"]
        
        return stats
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the miner.
        
        Returns:
            Dict containing the miner configuration
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        response = self.get(
            endpoint="/cgi-bin/config.cgi",
            auth=self.auth
        )
        
        return response.json()
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Get comprehensive telemetry data from the miner.
        
        This method fetches both stats and configuration data and combines them
        into a single telemetry object that matches the expected schema.
        
        Returns:
            Dict containing the transformed telemetry data
        
        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If the response data is invalid
        """
        stats = self.get_stats()
        config = self.get_config()
        
        if "status" not in stats or stats["status"] != "success":
            raise ValueError(f"Invalid stats response: {stats}")
        
        if "status" not in config or config["status"] != "success":
            raise ValueError(f"Invalid config response: {config}")
        
        # Extract data from the responses
        stats_data = stats["data"]
        config_data = config["data"]
        
        # Prepare the telemetry data
        telemetry_data = {
            "timestamp": int(time.time()),
            "miner_id": stats_data.get("miner_id", f"miner_{self.miner_ip}"),
            "ip_address": self.miner_ip,
            "model": self._extract_model(stats_data, config_data),
            "firmware_version": self._extract_firmware_version(stats_data, config_data),
            "hashrate": {
                "total": stats_data.get("hashrate", 0.0),
                "unit": stats_data.get("hashrate_unit", "TH/s"),
                "per_hashboard": self._extract_hashboard_data(stats_data)
            },
            "temperature": self._extract_temperature_data(stats_data),
            "power": {
                "consumption": stats_data.get("power", {}).get("consumption", 0.0),
                "efficiency": stats_data.get("power", {}).get("efficiency", 0.0),
                "voltage": stats_data.get("power", {}).get("voltage", 0.0)
            },
            "fans": self._extract_fan_data(stats_data),
            "pool": {
                "url": stats_data.get("pool", ""),
                "user": stats_data.get("worker", ""),
                "status": "mining" if stats_data.get("status") == "mining" else "disconnected"
            },
            "shares": {
                "accepted": stats_data.get("accepted_shares", 0),
                "rejected": stats_data.get("rejected_shares", 0),
                "stale": 0,
                "last_share_time": int(time.time())
            },
            "status": {
                "mining_status": stats_data.get("status", "unknown"),
                "uptime": stats_data.get("uptime", 0),
                "errors": []
            },
            "config": {
                "frequency": config_data.get("overclock", {}).get("frequency", 0.0),
                "overclock_profile": config_data.get("overclock", {}).get("profile", "default"),
                "power_limit": config_data.get("overclock", {}).get("power_limit", 0.0)
            }
        }
        
        # Validate and transform the data
        try:
            telemetry = VnishMinerTelemetry(**telemetry_data)
            return telemetry.transform()
        except Exception as e:
            logger.error(f"Error validating telemetry data: {str(e)}")
            raise ValueError(f"Invalid telemetry data: {str(e)}")
    
    def _extract_model(self, stats_data: Dict[str, Any], config_data: Dict[str, Any]) -> str:
        """Extract the miner model from the API response."""
        # Try to extract from miner_id which often contains the model
        miner_id = stats_data.get("miner_id", "")
        if "ANTMINER" in miner_id:
            return miner_id.split("_")[0] + "_" + miner_id.split("_")[1]
        
        # Default to a generic model
        return "ASIC_MINER"
    
    def _extract_firmware_version(self, stats_data: Dict[str, Any], config_data: Dict[str, Any]) -> str:
        """Extract the firmware version from the API response."""
        # This information might be in different places depending on the firmware
        # For now, we'll return a default value
        return "Vnish Firmware"
    
    def _extract_hashboard_data(self, stats_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract hashboard data from the API response."""
        hashboard_data = []
        
        # Check if per-board hashrate is available
        if "hashrate_per_board" in stats_data:
            for board_id, hashrate in stats_data["hashrate_per_board"].items():
                hashboard_data.append({
                    "board_id": int(board_id.replace("board", "")),
                    "hashrate": float(hashrate),
                    "status": "active"
                })
        else:
            # If not available, estimate based on total hashrate
            total_hashrate = stats_data.get("hashrate", 0.0)
            num_boards = 3  # Typical for Antminers
            
            # Check temperature data to estimate number of boards
            if "temperature" in stats_data:
                board_temps = [k for k in stats_data["temperature"].keys() if k.startswith("board")]
                if board_temps:
                    num_boards = len(board_temps)
            
            # Distribute hashrate evenly across boards
            if num_boards > 0 and total_hashrate > 0:
                hashrate_per_board = total_hashrate / num_boards
                for i in range(1, num_boards + 1):
                    hashboard_data.append({
                        "board_id": i,
                        "hashrate": hashrate_per_board,
                        "status": "active"
                    })
        
        return hashboard_data
    
    def _extract_temperature_data(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temperature data from the API response."""
        temp_data = {
            "ambient": 0.0,
            "avg_chip": 0.0,
            "max_chip": 0.0
        }
        
        if "temperature" in stats_data:
            temp = stats_data["temperature"]
            
            # Extract average temperature if available
            if "average" in temp:
                temp_data["avg_chip"] = float(temp["average"])
            
            # Calculate average and max from board temperatures
            board_temps = []
            for key, value in temp.items():
                if key.startswith("board"):
                    board_id = int(key.replace("board", ""))
                    board_temp = float(value)
                    board_temps.append(board_temp)
                    
                    # Add per-board temperature data
                    temp_data[key] = {
                        "pcb": board_temp,
                        "chip": board_temp  # Assuming chip temp is the same if not specified
                    }
            
            if board_temps:
                temp_data["avg_chip"] = sum(board_temps) / len(board_temps)
                temp_data["max_chip"] = max(board_temps)
        
        return temp_data
    
    def _extract_fan_data(self, stats_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract fan data from the API response."""
        fan_data = []
        
        if "fan_speed" in stats_data:
            fan_speed = stats_data["fan_speed"]
            
            for key, value in fan_speed.items():
                if key.startswith("fan") and key != "unit":
                    fan_id = int(key.replace("fan", ""))
                    speed_percent = float(value)
                    
                    # Convert percentage to RPM (rough estimate)
                    speed_rpm = speed_percent * 60
                    
                    fan_data.append({
                        "fan_id": fan_id,
                        "speed": speed_rpm,
                        "speed_percent": speed_percent,
                        "status": "active"
                    })
        
        return fan_data
