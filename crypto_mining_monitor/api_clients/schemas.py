
"""
Data schemas for API responses and transformations.

This module defines Pydantic models for validating and transforming data
from Vnish firmware and Prohashing.com APIs.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator, root_validator

class HashboardData(BaseModel):
    """Hashboard data from a miner."""
    board_id: int
    hashrate: float
    status: str = "active"

class TemperatureData(BaseModel):
    """Temperature data for a hashboard."""
    board_id: int
    pcb_temp: float
    chip_temp: float

class FanData(BaseModel):
    """Fan data from a miner."""
    fan_id: int
    speed: float
    speed_percent: float
    status: str = "active"

class ErrorData(BaseModel):
    """Error data from a miner."""
    code: str
    message: str
    severity: str

class PoolData(BaseModel):
    """Pool configuration data."""
    url: str
    user: str
    status: str = "connected"

class SharesData(BaseModel):
    """Shares data from a miner."""
    accepted: int
    rejected: int
    stale: int = 0
    last_share_time: Optional[int] = None

class PowerData(BaseModel):
    """Power consumption data."""
    consumption: float
    efficiency: float
    voltage: Optional[float] = None

class HashrateData(BaseModel):
    """Hashrate data from a miner."""
    total: float
    unit: str
    per_hashboard: List[HashboardData] = []

class StatusData(BaseModel):
    """Operational status data."""
    mining_status: str
    uptime: int
    errors: List[ErrorData] = []

class ConfigData(BaseModel):
    """Miner configuration data."""
    frequency: float
    overclock_profile: str
    power_limit: Optional[float] = None

class VnishMinerTelemetry(BaseModel):
    """
    Telemetry data from a Vnish firmware miner.
    
    This model matches the expected schema for miner telemetry data
    as defined in the data pipeline design.
    """
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    miner_id: str
    ip_address: str
    model: str
    firmware_version: str
    hashrate: HashrateData
    temperature: Dict[str, Any]  # Complex structure handled in transform method
    power: PowerData
    fans: List[FanData] = []
    pool: PoolData
    shares: SharesData
    status: StatusData
    config: ConfigData

    def transform(self) -> Dict[str, Any]:
        """
        Transform the telemetry data to match the expected schema.
        
        Returns:
            Dict containing the transformed data
        """
        # Extract temperature data in the expected format
        temp_data = {
            "ambient": self.temperature.get("ambient", 0.0),
            "avg_chip": self.temperature.get("avg_chip", 0.0),
            "max_chip": self.temperature.get("max_chip", 0.0),
            "per_hashboard": []
        }
        
        # Process per-hashboard temperature data if available
        for board_id, temps in self.temperature.items():
            if isinstance(temps, dict) and "pcb" in temps and "chip" in temps:
                temp_data["per_hashboard"].append({
                    "board_id": int(board_id.replace("board", "")),
                    "pcb_temp": temps["pcb"],
                    "chip_temp": temps["chip"]
                })
        
        return {
            "timestamp": self.timestamp,
            "miner_id": self.miner_id,
            "ip_address": self.ip_address,
            "model": self.model,
            "firmware_version": self.firmware_version,
            "hashrate": {
                "total": self.hashrate.total,
                "unit": self.hashrate.unit,
                "per_hashboard": [hb.dict() for hb in self.hashrate.per_hashboard]
            },
            "temperature": temp_data,
            "power": self.power.dict(),
            "fans": [fan.dict() for fan in self.fans],
            "pool": self.pool.dict(),
            "shares": self.shares.dict(),
            "status": self.status.dict(),
            "config": self.config.dict()
        }

class CoinMined(BaseModel):
    """Data about a coin being mined."""
    coin_id: str
    symbol: str
    amount: float
    usd_value: float
    mining_type: str  # "primary" or "merge-mined"

class PoolHashrateData(BaseModel):
    """Hashrate data from a mining pool."""
    reported: float
    effective: float
    unit: str

class PoolEarningsData(BaseModel):
    """Earnings data from a mining pool."""
    amount: float
    currency: str
    time_period: str

class PoolProfitabilityData(BaseModel):
    """Profitability data from a mining pool."""
    per_hash_rate: float
    unit: str
    time_period: str

class ProhashingPoolPerformance(BaseModel):
    """
    Pool performance data from Prohashing.
    
    This model matches the expected schema for mining pool data
    as defined in the data pipeline design.
    """
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    pool_id: str = "prohashing"
    worker_id: str
    algorithm: str
    hashrate: PoolHashrateData
    shares: SharesData
    earnings: PoolEarningsData
    coins_mined: List[CoinMined] = []
    profitability: PoolProfitabilityData
    difficulty: float
    status: str = "active"

    def transform(self) -> Dict[str, Any]:
        """
        Transform the pool performance data to match the expected schema.
        
        Returns:
            Dict containing the transformed data
        """
        return {
            "timestamp": self.timestamp,
            "pool_id": self.pool_id,
            "worker_id": self.worker_id,
            "algorithm": self.algorithm,
            "hashrate": self.hashrate.dict(),
            "shares": self.shares.dict(),
            "earnings": self.earnings.dict(),
            "coins_mined": [coin.dict() for coin in self.coins_mined],
            "profitability": self.profitability.dict(),
            "difficulty": self.difficulty,
            "status": self.status
        }

class ProhashingAlgorithmProfitability(BaseModel):
    """Profitability data for a mining algorithm from Prohashing."""
    algorithm_name: str
    usd: float
    btc: float
    max_usd: float
    max_btc: float
    percentile_usd: float
    percentile_btc: float
    data_timestamp: float
    server_timestamp: float
    server_id: int

    def transform(self) -> Dict[str, Any]:
        """
        Transform the algorithm profitability data.
        
        Returns:
            Dict containing the transformed data
        """
        return {
            "algorithm": self.algorithm_name,
            "profitability": {
                "current_usd": self.usd,
                "current_btc": self.btc,
                "max_usd": self.max_usd,
                "max_btc": self.max_btc,
                "percentile_usd": self.percentile_usd,
                "percentile_btc": self.percentile_btc
            },
            "timestamp": int(self.data_timestamp),
            "server_info": {
                "timestamp": int(self.server_timestamp),
                "server_id": self.server_id
            }
        }
