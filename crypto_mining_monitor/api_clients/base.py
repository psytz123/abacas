
"""
Base API client functionality for cryptocurrency mining APIs.

This module provides common functionality for API clients, including:
- Authentication handling
- Rate limiting and retry logic
- Error handling
- Response validation
"""

import logging
import time
from typing import Any, Dict, Optional, Union, Callable
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)

class BaseAPIClient:
    """Base class for API clients with common functionality."""
    
    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        status_forcelist: list = None
    ):
        """
        Initialize the base API client.
        
        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retries
            status_forcelist: HTTP status codes to retry on
        """
        self.base_url = base_url
        self.timeout = timeout
        
        # Configure session with retry logic
        self.session = requests.Session()
        
        if status_forcelist is None:
            status_forcelist = [429, 500, 502, 503, 504]
            
        retries = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["GET", "POST"]
        )
        
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
        json: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        auth: Any = None
    ) -> requests.Response:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Form data
            json: JSON data
            headers: HTTP headers
            auth: Authentication
            
        Returns:
            Response object
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json,
                headers=headers,
                auth=auth,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def get(
        self,
        endpoint: str,
        params: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        auth: Any = None
    ) -> requests.Response:
        """
        Make a GET request to the API with retry logic.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: HTTP headers
            auth: Authentication
            
        Returns:
            Response object
        """
        return self._make_request("GET", endpoint, params=params, headers=headers, auth=auth)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def post(
        self,
        endpoint: str,
        data: Dict[str, Any] = None,
        json: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        auth: Any = None
    ) -> requests.Response:
        """
        Make a POST request to the API with retry logic.
        
        Args:
            endpoint: API endpoint
            data: Form data
            json: JSON data
            headers: HTTP headers
            auth: Authentication
            
        Returns:
            Response object
        """
        return self._make_request("POST", endpoint, data=data, json=json, headers=headers, auth=auth)
    
    def close(self):
        """Close the session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
