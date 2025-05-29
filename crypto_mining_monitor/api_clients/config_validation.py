"""
Configuration Validation for API Clients

This module provides validation functions for API client configurations,
ensuring that settings are within safe and acceptable ranges before
being applied to hardware.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator, root_validator
import json

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


class HashrateTuningConfig(BaseModel):
    """Validation model for hashrate tuning configuration."""
    
    hashrate_percent: float = Field(..., ge=50.0, le=100.0, description="Target hashrate as a percentage of maximum (50-100)")
    
    @validator('hashrate_percent')
    def validate_hashrate_percent(cls, v):
        """Validate hashrate percentage is within safe limits."""
        if v < 50.0:
            logger.warning(f"Hashrate percentage {v}% is below recommended minimum (50%). This may cause instability.")
        return v


class PowerOptimizationConfig(BaseModel):
    """Validation model for power optimization configuration."""
    
    power_limit_factor: float = Field(..., gt=0.0, le=1.0, description="Target power as a factor of current power (0-1)")
    
    @validator('power_limit_factor')
    def validate_power_limit_factor(cls, v):
        """Validate power limit factor is within safe limits."""
        if v < 0.7:
            logger.warning(f"Power limit factor {v} is below recommended minimum (0.7). This may cause instability.")
        return v


class OverclockingConfig(BaseModel):
    """Validation model for overclocking configuration."""
    
    core_clock_offset: int = Field(..., ge=-100, le=200, description="Core clock offset in MHz (-100 to 200)")
    memory_clock_offset: int = Field(..., ge=-500, le=1500, description="Memory clock offset in MHz (-500 to 1500)")
    power_limit_percent: float = Field(..., ge=70.0, le=100.0, description="Power limit as a percentage of TDP (70-100)")
    core_voltage_offset: int = Field(..., ge=-50, le=50, description="Core voltage offset in mV (-50 to 50)")
    
    @root_validator
    def validate_overclocking_safety(cls, values):
        """Validate that the overclocking configuration is safe."""
        core_clock_offset = values.get('core_clock_offset', 0)
        memory_clock_offset = values.get('memory_clock_offset', 0)
        power_limit_percent = values.get('power_limit_percent', 100.0)
        core_voltage_offset = values.get('core_voltage_offset', 0)
        
        # Check for potentially unsafe combinations
        if core_clock_offset > 100 and core_voltage_offset < 0:
            logger.warning("High core clock offset with negative voltage offset may cause instability")
        
        if core_clock_offset > 150 and power_limit_percent < 90:
            logger.warning("High core clock offset with low power limit may cause throttling or instability")
        
        if memory_clock_offset > 1000 and power_limit_percent < 85:
            logger.warning("High memory clock offset with low power limit may cause throttling or instability")
        
        return values


class MinerConfig(BaseModel):
    """Validation model for miner configuration."""
    
    miner_ip: str
    username: str
    password: str
    timeout: int = Field(30, ge=5, le=120, description="Request timeout in seconds (5-120)")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum number of retries for failed requests (0-10)")
    verify_ssl: bool = Field(False, description="Whether to verify SSL certificates")


class ProfileConfig(BaseModel):
    """Validation model for profile configuration."""
    
    profile_name: str = Field(..., min_length=1, max_length=50, description="Name of the profile to activate")


class FrequencyConfig(BaseModel):
    """Validation model for frequency configuration."""
    
    frequency: int = Field(..., ge=100, le=1500, description="Frequency in MHz (100-1500)")
    board_id: Optional[int] = Field(None, ge=0, description="Optional board ID to set frequency for a specific board")


class VoltageConfig(BaseModel):
    """Validation model for voltage configuration."""
    
    voltage: float = Field(..., ge=0.8, le=1.5, description="Voltage in volts (0.8-1.5)")
    board_id: Optional[int] = Field(None, ge=0, description="Optional board ID to set voltage for a specific board")


class PowerLimitConfig(BaseModel):
    """Validation model for power limit configuration."""
    
    power_limit: int = Field(..., ge=500, le=5000, description="Power limit in watts (500-5000)")


class FanSpeedConfig(BaseModel):
    """Validation model for fan speed configuration."""
    
    fan_speed: int = Field(..., ge=0, le=100, description="Fan speed percentage (0-100)")


class MaxFanRpmConfig(BaseModel):
    """Validation model for maximum fan RPM configuration."""
    
    max_rpm: int = Field(..., ge=1000, le=7000, description="Maximum fan RPM (1000-7000)")


def validate_hashrate_tuning(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate hashrate tuning configuration.
    
    Args:
        config: Dictionary containing hashrate tuning configuration
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    try:
        validated = HashrateTuningConfig(**config)
        return validated.dict()
    except Exception as e:
        logger.error(f"Hashrate tuning validation error: {str(e)}")
        raise ValidationError(f"Invalid hashrate tuning configuration: {str(e)}")


def validate_power_optimization(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate power optimization configuration.
    
    Args:
        config: Dictionary containing power optimization configuration
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    try:
        validated = PowerOptimizationConfig(**config)
        return validated.dict()
    except Exception as e:
        logger.error(f"Power optimization validation error: {str(e)}")
        raise ValidationError(f"Invalid power optimization configuration: {str(e)}")


def validate_overclocking(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate overclocking configuration.
    
    Args:
        config: Dictionary containing overclocking configuration
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    try:
        validated = OverclockingConfig(**config)
        return validated.dict()
    except Exception as e:
        logger.error(f"Overclocking validation error: {str(e)}")
        raise ValidationError(f"Invalid overclocking configuration: {str(e)}")


def validate_miner_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate miner configuration.
    
    Args:
        config: Dictionary containing miner configuration
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    try:
        validated = MinerConfig(**config)
        return validated.dict()
    except Exception as e:
        logger.error(f"Miner configuration validation error: {str(e)}")
        raise ValidationError(f"Invalid miner configuration: {str(e)}")


def validate_profile_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate profile configuration.
    
    Args:
        config: Dictionary containing profile configuration
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    try:
        validated = ProfileConfig(**config)
        return validated.dict()
    except Exception as e:
        logger.error(f"Profile configuration validation error: {str(e)}")
        raise ValidationError(f"Invalid profile configuration: {str(e)}")


def validate_frequency_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate frequency configuration.
    
    Args:
        config: Dictionary containing frequency configuration
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    try:
        validated = FrequencyConfig(**config)
        return validated.dict()
    except Exception as e:
        logger.error(f"Frequency configuration validation error: {str(e)}")
        raise ValidationError(f"Invalid frequency configuration: {str(e)}")


def validate_voltage_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate voltage configuration.
    
    Args:
        config: Dictionary containing voltage configuration
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    try:
        validated = VoltageConfig(**config)
        return validated.dict()
    except Exception as e:
        logger.error(f"Voltage configuration validation error: {str(e)}")
        raise ValidationError(f"Invalid voltage configuration: {str(e)}")


def validate_power_limit_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate power limit configuration.
    
    Args:
        config: Dictionary containing power limit configuration
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    try:
        validated = PowerLimitConfig(**config)
        return validated.dict()
    except Exception as e:
        logger.error(f"Power limit configuration validation error: {str(e)}")
        raise ValidationError(f"Invalid power limit configuration: {str(e)}")


def validate_fan_speed_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate fan speed configuration.
    
    Args:
        config: Dictionary containing fan speed configuration
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    try:
        validated = FanSpeedConfig(**config)
        return validated.dict()
    except Exception as e:
        logger.error(f"Fan speed configuration validation error: {str(e)}")
        raise ValidationError(f"Invalid fan speed configuration: {str(e)}")


def validate_max_fan_rpm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate maximum fan RPM configuration.
    
    Args:
        config: Dictionary containing maximum fan RPM configuration
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    try:
        validated = MaxFanRpmConfig(**config)
        return validated.dict()
    except Exception as e:
        logger.error(f"Maximum fan RPM configuration validation error: {str(e)}")
        raise ValidationError(f"Invalid maximum fan RPM configuration: {str(e)}")


def validate_json_config(json_str: str) -> Dict[str, Any]:
    """
    Validate a JSON configuration string.
    
    Args:
        json_str: JSON string containing configuration
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    try:
        config = json.loads(json_str)
        
        # Determine the type of configuration and validate accordingly
        if "hashrate_percent" in config:
            return validate_hashrate_tuning(config)
        elif "power_limit_factor" in config:
            return validate_power_optimization(config)
        elif "core_clock_offset" in config:
            return validate_overclocking(config)
        elif "miner_ip" in config:
            return validate_miner_config(config)
        elif "profile_name" in config:
            return validate_profile_config(config)
        elif "frequency" in config:
            return validate_frequency_config(config)
        elif "voltage" in config:
            return validate_voltage_config(config)
        elif "power_limit" in config:
            return validate_power_limit_config(config)
        elif "fan_speed" in config:
            return validate_fan_speed_config(config)
        elif "max_rpm" in config:
            return validate_max_fan_rpm_config(config)
        else:
            raise ValidationError("Unknown configuration type")
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        raise ValidationError(f"Invalid JSON: {str(e)}")
    except ValidationError:
        # Re-raise ValidationError
        raise
    except Exception as e:
        logger.error(f"Configuration validation error: {str(e)}")
        raise ValidationError(f"Invalid configuration: {str(e)}")
