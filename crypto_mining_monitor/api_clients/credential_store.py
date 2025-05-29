"""
Credential Store for API Clients

This module provides secure storage and retrieval of credentials for API clients,
including Vnish firmware access credentials.
"""

import os
import json
import logging
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class CredentialStore:
    """
    Secure storage and retrieval of API credentials.
    
    This class provides methods for storing and retrieving credentials
    for various API clients, with optional encryption for sensitive data.
    """
    
    def __init__(self, config_dir: Optional[str] = None, use_encryption: bool = True):
        """
        Initialize the credential store.
        
        Args:
            config_dir: Directory to store credential files. If None, uses ~/.crypto_mining_monitor
            use_encryption: Whether to encrypt stored credentials
        """
        if config_dir is None:
            self.config_dir = os.path.expanduser("~/.crypto_mining_monitor")
        else:
            self.config_dir = config_dir
        
        # Create config directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        self.credentials_file = os.path.join(self.config_dir, "credentials.json")
        self.use_encryption = use_encryption
        self._encryption_key = None
        
        # Initialize encryption if needed
        if self.use_encryption:
            self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption with a key derived from an environment variable or config file."""
        # Try to get encryption key from environment variable
        env_key = os.environ.get("CRYPTO_MINING_MONITOR_KEY")
        
        if env_key:
            # Use the environment variable as the encryption key
            key_bytes = self._derive_key(env_key)
        else:
            # Check if a key file exists
            key_file = os.path.join(self.config_dir, ".encryption_key")
            
            if os.path.exists(key_file):
                # Read the key from the file
                with open(key_file, "rb") as f:
                    key_bytes = base64.urlsafe_b64decode(f.read())
            else:
                # Generate a new key and save it
                key_bytes = Fernet.generate_key()
                with open(key_file, "wb") as f:
                    f.write(base64.urlsafe_b64encode(key_bytes))
                
                # Set restrictive permissions on the key file
                os.chmod(key_file, 0o600)
        
        # Create the Fernet cipher
        self._encryption_key = Fernet(key_bytes)
    
    def _derive_key(self, password: str) -> bytes:
        """Derive an encryption key from a password."""
        salt = b'crypto_mining_monitor_salt'  # In production, use a secure random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def _encrypt(self, data: str) -> str:
        """Encrypt a string."""
        if not self.use_encryption or self._encryption_key is None:
            return data
        
        encrypted = self._encryption_key.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def _decrypt(self, data: str) -> str:
        """Decrypt a string."""
        if not self.use_encryption or self._encryption_key is None:
            return data
        
        decrypted = self._encryption_key.decrypt(base64.urlsafe_b64decode(data.encode()))
        return decrypted.decode()
    
    def save_credentials(self, service: str, credentials: Dict[str, Any]) -> bool:
        """
        Save credentials for a service.
        
        Args:
            service: Service identifier (e.g., 'vnish_firmware')
            credentials: Dictionary of credentials
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing credentials
            all_credentials = self.load_all_credentials()
            
            # Encrypt sensitive fields if encryption is enabled
            if self.use_encryption and self._encryption_key is not None:
                encrypted_credentials = {}
                for key, value in credentials.items():
                    if isinstance(value, str):
                        encrypted_credentials[key] = self._encrypt(value)
                    else:
                        encrypted_credentials[key] = value
                
                all_credentials[service] = encrypted_credentials
            else:
                all_credentials[service] = credentials
            
            # Save to file
            with open(self.credentials_file, "w") as f:
                json.dump(all_credentials, f)
            
            # Set restrictive permissions on the credentials file
            os.chmod(self.credentials_file, 0o600)
            
            logger.info(f"Saved credentials for service: {service}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving credentials for service {service}: {str(e)}")
            return False
    
    def load_credentials(self, service: str) -> Dict[str, Any]:
        """
        Load credentials for a service.
        
        Args:
            service: Service identifier (e.g., 'vnish_firmware')
        
        Returns:
            Dictionary of credentials, or empty dict if not found
        """
        try:
            all_credentials = self.load_all_credentials()
            
            if service not in all_credentials:
                logger.warning(f"No credentials found for service: {service}")
                return {}
            
            credentials = all_credentials[service]
            
            # Decrypt sensitive fields if encryption is enabled
            if self.use_encryption and self._encryption_key is not None:
                decrypted_credentials = {}
                for key, value in credentials.items():
                    if isinstance(value, str):
                        try:
                            decrypted_credentials[key] = self._decrypt(value)
                        except Exception:
                            # If decryption fails, assume the value is not encrypted
                            decrypted_credentials[key] = value
                    else:
                        decrypted_credentials[key] = value
                
                return decrypted_credentials
            else:
                return credentials
        
        except Exception as e:
            logger.error(f"Error loading credentials for service {service}: {str(e)}")
            return {}
    
    def load_all_credentials(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all stored credentials.
        
        Returns:
            Dictionary of all credentials, keyed by service
        """
        if not os.path.exists(self.credentials_file):
            return {}
        
        try:
            with open(self.credentials_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading credentials file: {str(e)}")
            return {}
    
    def delete_credentials(self, service: str) -> bool:
        """
        Delete credentials for a service.
        
        Args:
            service: Service identifier (e.g., 'vnish_firmware')
        
        Returns:
            True if successful, False otherwise
        """
        try:
            all_credentials = self.load_all_credentials()
            
            if service in all_credentials:
                del all_credentials[service]
                
                # Save to file
                with open(self.credentials_file, "w") as f:
                    json.dump(all_credentials, f)
                
                logger.info(f"Deleted credentials for service: {service}")
                return True
            else:
                logger.warning(f"No credentials found for service: {service}")
                return False
        
        except Exception as e:
            logger.error(f"Error deleting credentials for service {service}: {str(e)}")
            return False


class VnishCredentialManager:
    """
    Manager for Vnish firmware credentials.
    
    This class provides methods for storing and retrieving Vnish firmware
    credentials, with support for environment variables and credential files.
    """
    
    def __init__(self, credential_store: Optional[CredentialStore] = None):
        """
        Initialize the Vnish credential manager.
        
        Args:
            credential_store: CredentialStore instance to use. If None, creates a new one.
        """
        self.credential_store = credential_store or CredentialStore()
        self.service_name = "vnish_firmware"
    
    def get_credentials(self, miner_ip: Optional[str] = None) -> Tuple[str, str, str]:
        """
        Get credentials for a Vnish firmware miner.
        
        Args:
            miner_ip: IP address of the miner. If None, returns default credentials.
        
        Returns:
            Tuple of (miner_ip, username, password)
        
        Raises:
            ValueError: If credentials are not found and not available in environment variables
        """
        # First, try to get credentials from environment variables
        env_ip = os.environ.get("VNISH_HOST")
        env_username = os.environ.get("VNISH_USERNAME")
        env_password = os.environ.get("VNISH_PASSWORD")
        
        # If miner_ip is provided, use it; otherwise use the environment variable
        if miner_ip is None:
            miner_ip = env_ip
        
        # If we have all environment variables and no specific miner_ip was requested,
        # or the requested miner_ip matches the environment variable, use the environment credentials
        if env_ip and env_username and env_password and (miner_ip is None or miner_ip == env_ip):
            logger.info(f"Using Vnish credentials from environment variables for {env_ip}")
            return env_ip, env_username, env_password
        
        # If we have a specific miner_ip, try to load credentials for it
        if miner_ip:
            # Try to load credentials for this specific miner
            credentials = self.credential_store.load_credentials(f"{self.service_name}_{miner_ip}")
            
            if credentials and "username" in credentials and "password" in credentials:
                logger.info(f"Using stored Vnish credentials for {miner_ip}")
                return miner_ip, credentials["username"], credentials["password"]
        
        # Try to load default credentials
        credentials = self.credential_store.load_credentials(self.service_name)
        
        if credentials and "miner_ip" in credentials and "username" in credentials and "password" in credentials:
            logger.info(f"Using default stored Vnish credentials for {credentials['miner_ip']}")
            return credentials["miner_ip"], credentials["username"], credentials["password"]
        
        # If we have a miner_ip but no username/password, check if we have environment username/password
        if miner_ip and env_username and env_password:
            logger.info(f"Using environment username/password for {miner_ip}")
            return miner_ip, env_username, env_password
        
        # If we still don't have credentials, raise an error
        raise ValueError(
            "Vnish credentials not found. Please set VNISH_HOST, VNISH_USERNAME, and VNISH_PASSWORD "
            "environment variables, or store credentials using save_credentials()."
        )
    
    def save_credentials(self, miner_ip: str, username: str, password: str, is_default: bool = False) -> bool:
        """
        Save credentials for a Vnish firmware miner.
        
        Args:
            miner_ip: IP address of the miner
            username: Username for authentication
            password: Password for authentication
            is_default: Whether these credentials should be used as the default
        
        Returns:
            True if successful, False otherwise
        """
        credentials = {
            "miner_ip": miner_ip,
            "username": username,
            "password": password
        }
        
        # Save as miner-specific credentials
        success = self.credential_store.save_credentials(f"{self.service_name}_{miner_ip}", credentials)
        
        # If this is the default, also save as default credentials
        if is_default:
            success = success and self.credential_store.save_credentials(self.service_name, credentials)
        
        return success
    
    def delete_credentials(self, miner_ip: str) -> bool:
        """
        Delete credentials for a Vnish firmware miner.
        
        Args:
            miner_ip: IP address of the miner
        
        Returns:
            True if successful, False otherwise
        """
        return self.credential_store.delete_credentials(f"{self.service_name}_{miner_ip}")
