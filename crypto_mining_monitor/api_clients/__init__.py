
"""
Cryptocurrency Mining API Clients

This package provides API clients for interacting with Vnish firmware and Prohashing.com
to fetch mining telemetry data, pool performance, and profitability metrics.
"""

from .vnish_client import VnishClient
from .prohash_client import ProhashingClient

__all__ = ['VnishClient', 'ProhashingClient']
