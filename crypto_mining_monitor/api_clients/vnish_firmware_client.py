"""
Vnish Firmware API Client

This module provides a client for interacting with the Vnish firmware API
to control and configure ASIC miners, including applying ML-recommended settings.
"""

import logging
import time
import os
import json
import requests
from typing import Dict, List, Optional, Any, Union
from requests.auth import HTTPBasicAuth
import urllib3
from urllib3.exceptions import InsecureRequestWarning

from .base import BaseAPIClient
from .schemas import VnishMinerTelemetry

# Suppress insecure request warnings when verify=False is used
urllib3.disable_warnings(InsecureRequestWarning)

logger = logging.getLogger(__name__)

class VnishFirmwareClient(BaseAPIClient):
    """
    Client for interacting with the Vnish firmware API.
    
    This client provides methods for controlling and configuring ASIC miners
    running Vnish firmware, including applying ML-recommended settings for
    hashrate tuning, overclocking, and power optimization.
    """
    
    def __init__(
        self,
        miner_ip: str,
        username: str,
        password: str,
        timeout: int = 30,
        max_retries: int = 3,
        verify_ssl: bool = False
    ):
        """
        Initialize the Vnish Firmware API client.
        
        Args:
            miner_ip: IP address of the miner
            username: Username for authentication
            password: Password for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            verify_ssl: Whether to verify SSL certificates
        """
        base_url = f"http://{miner_ip}"
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries
        )
        self.auth = HTTPBasicAuth(username, password)
        self.miner_ip = miner_ip
        self.verify_ssl = verify_ssl
        self.miner_id = None  # Will be populated from the first API call
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the miner.
        
        Returns:
            Dict containing the miner summary information
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        response = self.get(
            endpoint="/summary",
            auth=self.auth,
            verify=self.verify_ssl
        )
        
        summary = response.json()
        
        # Store the miner ID for future reference if available
        if self.miner_id is None and "miner_id" in summary:
            self.miner_id = summary["miner_id"]
        
        return summary
    
    def get_chips_status(self) -> Dict[str, Any]:
        """
        Get detailed status information about all chips.
        
        Returns:
            Dict containing the chip status information
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        response = self.get(
            endpoint="/chips",
            auth=self.auth,
            verify=self.verify_ssl
        )
        
        return response.json()
    
    def get_pools(self) -> Dict[str, Any]:
        """
        Get information about configured mining pools.
        
        Returns:
            Dict containing the pool configuration
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        response = self.get(
            endpoint="/pools",
            auth=self.auth,
            verify=self.verify_ssl
        )
        
        return response.json()
    
    def get_profiles(self) -> Dict[str, Any]:
        """
        Get available overclocking profiles.
        
        Returns:
            Dict containing the available profiles
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        response = self.get(
            endpoint="/profiles",
            auth=self.auth,
            verify=self.verify_ssl
        )
        
        return response.json()
    
    def set_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Set the active overclocking profile.
        
        Args:
            profile_name: Name of the profile to activate
        
        Returns:
            Dict containing the response
        
        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If the profile name is invalid
        """
        # First, validate that the profile exists
        profiles = self.get_profiles()
        available_profiles = profiles.get("profiles", [])
        
        if not any(p.get("name") == profile_name for p in available_profiles):
            raise ValueError(f"Profile '{profile_name}' not found. Available profiles: {[p.get('name') for p in available_profiles]}")
        
        # Set the profile
        data = {
            "profile": profile_name
        }
        
        response = self.post(
            endpoint="/profile",
            json=data,
            auth=self.auth,
            verify=self.verify_ssl
        )
        
        return response.json()
    
    def set_frequency(self, frequency: int, board_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Set the frequency for all chips or a specific board.
        
        Args:
            frequency: Frequency in MHz
            board_id: Optional board ID to set frequency for a specific board
        
        Returns:
            Dict containing the response
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        data = {
            "frequency": frequency
        }
        
        if board_id is not None:
            data["board"] = board_id
        
        response = self.post(
            endpoint="/frequency",
            json=data,
            auth=self.auth,
            verify=self.verify_ssl
        )
        
        return response.json()
    
    def set_voltage(self, voltage: float, board_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Set the voltage for all chips or a specific board.
        
        Args:
            voltage: Voltage in volts (e.g., 1.2 for 1.2V)
            board_id: Optional board ID to set voltage for a specific board
        
        Returns:
            Dict containing the response
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        data = {
            "voltage": voltage
        }
        
        if board_id is not None:
            data["board"] = board_id
        
        response = self.post(
            endpoint="/voltage",
            json=data,
            auth=self.auth,
            verify=self.verify_ssl
        )
        
        return response.json()
    
    def set_power_limit(self, power_limit: int) -> Dict[str, Any]:
        """
        Set the power limit for the miner.
        
        Args:
            power_limit: Power limit in watts
        
        Returns:
            Dict containing the response
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        data = {
            "power_limit": power_limit
        }
        
        response = self.post(
            endpoint="/power",
            json=data,
            auth=self.auth,
            verify=self.verify_ssl
        )
        
        return response.json()
    
    def set_fan_speed(self, fan_speed: int) -> Dict[str, Any]:
        """
        Set the fan speed for the miner.
        
        Args:
            fan_speed: Fan speed percentage (0-100)
        
        Returns:
            Dict containing the response
        
        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If fan_speed is not between 0 and 100
        """
        if not 0 <= fan_speed <= 100:
            raise ValueError("Fan speed must be between 0 and 100")
        
        data = {
            "fan_speed": fan_speed
        }
        
        response = self.post(
            endpoint="/fan",
            json=data,
            auth=self.auth,
            verify=self.verify_ssl
        )
        
        return response.json()
    
    def set_max_fan_rpm(self, max_rpm: int) -> Dict[str, Any]:
        """
        Set the maximum fan RPM for the miner.
        
        Args:
            max_rpm: Maximum fan RPM
        
        Returns:
            Dict containing the response
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        data = {
            "max_rpm": max_rpm
        }
        
        response = self.post(
            endpoint="/fan/max",
            json=data,
            auth=self.auth,
            verify=self.verify_ssl
        )
        
        return response.json()
    
    def restart_miner(self) -> Dict[str, Any]:
        """
        Restart the miner.
        
        Returns:
            Dict containing the response
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        response = self.post(
            endpoint="/restart",
            auth=self.auth,
            verify=self.verify_ssl
        )
        
        return response.json()
    
    def apply_hashrate_tuning(self, hashrate_percent: float) -> Dict[str, Any]:
        """
        Apply hashrate tuning based on ML recommendations.
        
        Args:
            hashrate_percent: Target hashrate as a percentage of maximum (0-100)
        
        Returns:
            Dict containing the response and applied settings
        
        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If hashrate_percent is not between 0 and 100
        """
        if not 0 <= hashrate_percent <= 100:
            raise ValueError("Hashrate percentage must be between 0 and 100")
        
        # Get current settings
        summary = self.get_summary()
        chips_status = self.get_chips_status()
        
        # Extract current frequency and calculate target frequency
        current_frequency = None
        if "frequency" in summary:
            current_frequency = summary["frequency"]
        elif "chips" in chips_status and chips_status["chips"]:
            # Take the average frequency of all chips
            frequencies = [chip.get("frequency", 0) for chip in chips_status["chips"] if "frequency" in chip]
            if frequencies:
                current_frequency = sum(frequencies) / len(frequencies)
        
        if current_frequency is None:
            raise ValueError("Could not determine current frequency")
        
        # Calculate target frequency based on hashrate percentage
        # This is a simplified approach - in reality, the relationship between
        # frequency and hashrate is not perfectly linear
        target_frequency = int(current_frequency * (hashrate_percent / 100.0))
        
        # Apply the new frequency
        result = self.set_frequency(target_frequency)
        
        # Return the result with additional information
        return {
            "status": "success" if result.get("status") == "ok" else "error",
            "message": f"Applied hashrate tuning: {hashrate_percent}% of maximum",
            "original_frequency": current_frequency,
            "target_frequency": target_frequency,
            "api_response": result
        }
    
    def apply_power_optimization(self, power_limit_factor: float) -> Dict[str, Any]:
        """
        Apply power optimization based on ML recommendations.
        
        Args:
            power_limit_factor: Target power as a factor of current power (0.0-1.0)
        
        Returns:
            Dict containing the response and applied settings
        
        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If power_limit_factor is not between 0 and 1
        """
        if not 0 < power_limit_factor <= 1:
            raise ValueError("Power limit factor must be between 0 and 1")
        
        # Get current settings
        summary = self.get_summary()
        
        # Extract current power consumption
        current_power = None
        if "power_consumption" in summary:
            current_power = summary["power_consumption"]
        else:
            raise ValueError("Could not determine current power consumption")
        
        # Calculate target power limit
        target_power = int(current_power * power_limit_factor)
        
        # Apply the new power limit
        result = self.set_power_limit(target_power)
        
        # Return the result with additional information
        return {
            "status": "success" if result.get("status") == "ok" else "error",
            "message": f"Applied power optimization: {power_limit_factor:.2f} of current power",
            "original_power": current_power,
            "target_power": target_power,
            "api_response": result
        }
    
    def apply_overclocking(self, 
                          core_clock_offset: int, 
                          memory_clock_offset: int, 
                          power_limit_percent: float, 
                          core_voltage_offset: int) -> Dict[str, Any]:
        """
        Apply intelligent overclocking based on ML recommendations.
        
        Args:
            core_clock_offset: Core clock offset in MHz
            memory_clock_offset: Memory clock offset in MHz
            power_limit_percent: Power limit as a percentage of TDP (0-100)
            core_voltage_offset: Core voltage offset in mV
        
        Returns:
            Dict containing the response and applied settings
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        # Get current settings
        summary = self.get_summary()
        chips_status = self.get_chips_status()
        
        # Extract current frequency and voltage
        current_frequency = None
        current_voltage = None
        
        if "frequency" in summary:
            current_frequency = summary["frequency"]
        elif "chips" in chips_status and chips_status["chips"]:
            frequencies = [chip.get("frequency", 0) for chip in chips_status["chips"] if "frequency" in chip]
            if frequencies:
                current_frequency = sum(frequencies) / len(frequencies)
        
        if "voltage" in summary:
            current_voltage = summary["voltage"]
        elif "chips" in chips_status and chips_status["chips"]:
            voltages = [chip.get("voltage", 0) for chip in chips_status["chips"] if "voltage" in chip]
            if voltages:
                current_voltage = sum(voltages) / len(voltages)
        
        if current_frequency is None or current_voltage is None:
            raise ValueError("Could not determine current frequency or voltage")
        
        # Calculate target frequency and voltage
        target_frequency = int(current_frequency + core_clock_offset)
        target_voltage = current_voltage + (core_voltage_offset / 1000.0)  # Convert mV to V
        
        # Calculate target power limit
        current_power = summary.get("power_consumption", 0)
        target_power = int(current_power * (power_limit_percent / 100.0))
        
        # Apply the new settings
        frequency_result = self.set_frequency(target_frequency)
        voltage_result = self.set_voltage(target_voltage)
        power_result = self.set_power_limit(target_power)
        
        # Return the result with additional information
        return {
            "status": "success" if all(r.get("status") == "ok" for r in [frequency_result, voltage_result, power_result]) else "error",
            "message": f"Applied overclocking settings",
            "original_settings": {
                "frequency": current_frequency,
                "voltage": current_voltage,
                "power": current_power
            },
            "target_settings": {
                "frequency": target_frequency,
                "voltage": target_voltage,
                "power": target_power
            },
            "api_responses": {
                "frequency": frequency_result,
                "voltage": voltage_result,
                "power": power_result
            }
        }
    
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Get comprehensive telemetry data from the miner.
        
        This method fetches both summary and chip status data and combines them
        into a single telemetry object that matches the expected schema.
        
        Returns:
            Dict containing the transformed telemetry data
        
        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If the response data is invalid
        """
        summary = self.get_summary()
        chips_status = self.get_chips_status()
        pools = self.get_pools()
        
        if "status" in summary and summary["status"] != "ok":
            raise ValueError(f"Invalid summary response: {summary}")
        
        # Extract data from the responses
        # This is a simplified implementation - in a real scenario, you would
        # need to handle different firmware versions and response formats
        
        # Prepare the telemetry data
        telemetry_data = {
            "timestamp": int(time.time()),
            "miner_id": summary.get("miner_id", f"miner_{self.miner_ip}"),
            "ip_address": self.miner_ip,
            "model": summary.get("model", "Unknown"),
            "firmware_version": summary.get("firmware_version", "Vnish Firmware"),
            "hashrate": {
                "total": summary.get("hashrate", 0.0),
                "unit": summary.get("hashrate_unit", "TH/s"),
                "per_hashboard": self._extract_hashboard_data(chips_status)
            },
            "temperature": self._extract_temperature_data(summary, chips_status),
            "power": {
                "consumption": summary.get("power_consumption", 0.0),
                "efficiency": summary.get("efficiency", 0.0),
                "voltage": summary.get("voltage", 0.0)
            },
            "fans": self._extract_fan_data(summary),
            "pool": self._extract_pool_data(pools),
            "shares": {
                "accepted": summary.get("accepted_shares", 0),
                "rejected": summary.get("rejected_shares", 0),
                "stale": 0,
                "last_share_time": int(time.time())
            },
            "status": {
                "mining_status": summary.get("status", "unknown"),
                "uptime": summary.get("uptime", 0),
                "errors": []
            },
            "config": {
                "frequency": summary.get("frequency", 0.0),
                "overclock_profile": summary.get("profile", "default"),
                "power_limit": summary.get("power_limit", 0.0)
            }
        }
        
        # Validate and transform the data
        try:
            telemetry = VnishMinerTelemetry(**telemetry_data)
            return telemetry.transform()
        except Exception as e:
            logger.error(f"Error validating telemetry data: {str(e)}")
            raise ValueError(f"Invalid telemetry data: {str(e)}")
    
    def _extract_hashboard_data(self, chips_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract hashboard data from the chips status response."""
        hashboard_data = []
        
        # Check if per-board hashrate is available
        if "boards" in chips_status:
            for board in chips_status["boards"]:
                board_id = board.get("id", 0)
                hashrate = board.get("hashrate", 0.0)
                status = board.get("status", "unknown")
                
                hashboard_data.append({
                    "board_id": board_id,
                    "hashrate": float(hashrate),
                    "status": "active" if status == "ok" else status
                })
        elif "chips" in chips_status:
            # Group chips by board
            boards = {}
            for chip in chips_status["chips"]:
                board_id = chip.get("board", 0)
                if board_id not in boards:
                    boards[board_id] = {
                        "chips": [],
                        "hashrate": 0.0,
                        "status": "active"
                    }
                
                boards[board_id]["chips"].append(chip)
                
                # If any chip is not ok, mark the board as not active
                if chip.get("status", "ok") != "ok":
                    boards[board_id]["status"] = "error"
            
            # Calculate hashrate per board (simplified)
            for board_id, board in boards.items():
                # In a real implementation, you would calculate the hashrate
                # based on the frequency and number of chips
                hashboard_data.append({
                    "board_id": board_id,
                    "hashrate": 0.0,  # Placeholder
                    "status": board["status"]
                })
        
        return hashboard_data
    
    def _extract_temperature_data(self, summary: Dict[str, Any], chips_status: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temperature data from the summary and chips status responses."""
        temp_data = {
            "ambient": 0.0,
            "avg_chip": 0.0,
            "max_chip": 0.0
        }
        
        # Extract temperature data from summary
        if "temperature" in summary:
            temp = summary["temperature"]
            if isinstance(temp, dict):
                temp_data["ambient"] = temp.get("ambient", 0.0)
                temp_data["avg_chip"] = temp.get("chip", 0.0)
                temp_data["max_chip"] = temp.get("chip_max", 0.0)
            else:
                # If temperature is a single value, use it for all fields
                temp_data["avg_chip"] = float(temp)
                temp_data["max_chip"] = float(temp)
        
        # Extract per-board temperature data from chips status
        if "boards" in chips_status:
            board_temps = []
            for board in chips_status["boards"]:
                board_id = board.get("id", 0)
                temp = board.get("temperature", 0.0)
                board_temps.append(temp)
                
                # Add per-board temperature data
                temp_data[f"board{board_id}"] = {
                    "pcb": temp,
                    "chip": temp  # Assuming chip temp is the same if not specified
                }
            
            if board_temps:
                temp_data["avg_chip"] = sum(board_temps) / len(board_temps)
                temp_data["max_chip"] = max(board_temps)
        
        return temp_data
    
    def _extract_fan_data(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract fan data from the summary response."""
        fan_data = []
        
        if "fans" in summary:
            fans = summary["fans"]
            if isinstance(fans, list):
                for i, fan in enumerate(fans):
                    if isinstance(fan, dict):
                        fan_id = fan.get("id", i)
                        speed = fan.get("speed", 0)
                        speed_percent = fan.get("percent", 0)
                    else:
                        # If fan is a single value, assume it's the speed
                        fan_id = i
                        speed = float(fan)
                        speed_percent = 0  # Unknown
                    
                    fan_data.append({
                        "fan_id": fan_id,
                        "speed": speed,
                        "speed_percent": speed_percent,
                        "status": "active"
                    })
            elif isinstance(fans, dict):
                for fan_id, fan_info in fans.items():
                    if isinstance(fan_info, dict):
                        speed = fan_info.get("speed", 0)
                        speed_percent = fan_info.get("percent", 0)
                    else:
                        # If fan_info is a single value, assume it's the speed
                        speed = float(fan_info)
                        speed_percent = 0  # Unknown
                    
                    fan_data.append({
                        "fan_id": int(fan_id.replace("fan", "")),
                        "speed": speed,
                        "speed_percent": speed_percent,
                        "status": "active"
                    })
        
        return fan_data
    
    def _extract_pool_data(self, pools: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pool data from the pools response."""
        pool_data = {
            "url": "",
            "user": "",
            "status": "disconnected"
        }
        
        if "pools" in pools and pools["pools"]:
            # Get the first active pool
            active_pool = None
            for pool in pools["pools"]:
                if pool.get("status", "") == "active":
                    active_pool = pool
                    break
            
            # If no active pool, use the first one
            if active_pool is None and pools["pools"]:
                active_pool = pools["pools"][0]
            
            if active_pool:
                pool_data["url"] = active_pool.get("url", "")
                pool_data["user"] = active_pool.get("user", "")
                pool_data["status"] = "mining" if active_pool.get("status", "") == "active" else "disconnected"
        
        return pool_data
