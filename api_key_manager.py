#!/usr/bin/env python3
"""
API Key Management System for ChainBreak
Handles API key rotation, validation, and fallback mechanisms.
"""

import logging
import time
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class APIKey:
    """Represents an API key with metadata."""
    key: str
    provider: str
    is_active: bool = True
    last_used: Optional[datetime] = None
    error_count: int = 0
    max_errors: int = 5
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class APIKeyManager:
    """Manages API keys for multiple providers with rotation and validation."""
    
    def __init__(self, config_file: str = "api_keys.json"):
        self.config_file = config_file
        self.api_keys: Dict[str, List[APIKey]] = {}
        self.key_cache: Dict[str, APIKey] = {}
        self.load_config()
        
    def load_config(self):
        """Load API keys from configuration file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    for provider, keys in config.items():
                        self.api_keys[provider] = []
                        for key_data in keys:
                            api_key = APIKey(
                                key=key_data['key'],
                                provider=provider,
                                is_active=key_data.get('is_active', True),
                                error_count=key_data.get('error_count', 0),
                                max_errors=key_data.get('max_errors', 5)
                            )
                            self.api_keys[provider].append(api_key)
                logger.info(f"Loaded API keys for {len(self.api_keys)} providers")
            except Exception as e:
                logger.error(f"Error loading API keys: {e}")
        else:
            # Create default configuration
            self.create_default_config()
    
    def create_default_config(self):
        """Create default API key configuration."""
        default_config = {
            "chainalysis": [
                {
                    "key": "db373a00f1f63693d7ccf144ee781787865310acda3870ca8abfb09135cbfc58",
                    "is_active": True,
                    "error_count": 0,
                    "max_errors": 5
                }
            ],
            "blockchain_info": [
                {
                    "key": "free_tier",
                    "is_active": True,
                    "error_count": 0,
                    "max_errors": 10
                }
            ]
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        logger.info("Created default API key configuration")
    
    def get_active_key(self, provider: str) -> Optional[APIKey]:
        """Get an active API key for the specified provider."""
        if provider not in self.api_keys:
            logger.warning(f"No API keys configured for provider: {provider}")
            return None
        
        # Check cache first
        if provider in self.key_cache:
            cached_key = self.key_cache[provider]
            if cached_key.is_active and cached_key.error_count < cached_key.max_errors:
                return cached_key
        
        # Find active key
        for api_key in self.api_keys[provider]:
            if api_key.is_active and api_key.error_count < api_key.max_errors:
                self.key_cache[provider] = api_key
                return api_key
        
        logger.warning(f"No active API keys available for provider: {provider}")
        return None
    
    def validate_key(self, provider: str, api_key: APIKey) -> bool:
        """Validate an API key by making a test request."""
        try:
            if provider == "chainalysis":
                # Test Chainalysis API
                url = "https://api.chainalysis.com/api/v1/addresses/1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
                headers = {
                    "Token": api_key.key,
                    "Content-Type": "application/json"
                }
                response = requests.get(url, headers=headers, timeout=10)
                
                # 410 means key is expired, 401 means invalid
                if response.status_code in [401, 410]:
                    return False
                elif response.status_code == 200:
                    return True
                else:
                    # Other errors might be temporary
                    return True
                    
            elif provider == "blockchain_info":
                # Test Blockchain.info API
                url = "https://blockchain.info/rawaddr/1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
                response = requests.get(url, timeout=10)
                return response.status_code in [200, 429]  # 429 is rate limit, not key issue
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating API key for {provider}: {e}")
            return False
    
    def report_error(self, provider: str, api_key: APIKey, error_type: str = "general"):
        """Report an error for an API key."""
        api_key.error_count += 1
        api_key.last_used = datetime.utcnow()
        
        logger.warning(f"API key error for {provider}: {error_type} (error count: {api_key.error_count})")
        
        # Deactivate key if too many errors
        if api_key.error_count >= api_key.max_errors:
            api_key.is_active = False
            logger.error(f"Deactivated API key for {provider} due to excessive errors")
            
            # Clear cache
            if provider in self.key_cache:
                del self.key_cache[provider]
        
        self.save_config()
    
    def report_success(self, provider: str, api_key: APIKey):
        """Report successful API key usage."""
        api_key.last_used = datetime.utcnow()
        api_key.error_count = max(0, api_key.error_count - 1)  # Reduce error count on success
        self.save_config()
    
    def save_config(self):
        """Save current API key configuration."""
        try:
            config = {}
            for provider, keys in self.api_keys.items():
                config[provider] = []
                for api_key in keys:
                    config[provider].append({
                        "key": api_key.key,
                        "is_active": api_key.is_active,
                        "error_count": api_key.error_count,
                        "max_errors": api_key.max_errors
                    })
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving API key configuration: {e}")
    
    def add_key(self, provider: str, key: str, max_errors: int = 5):
        """Add a new API key for a provider."""
        if provider not in self.api_keys:
            self.api_keys[provider] = []
        
        api_key = APIKey(
            key=key,
            provider=provider,
            max_errors=max_errors
        )
        
        # Validate the key
        if self.validate_key(provider, api_key):
            self.api_keys[provider].append(api_key)
            self.save_config()
            logger.info(f"Added and validated new API key for {provider}")
            return True
        else:
            logger.error(f"Failed to validate new API key for {provider}")
            return False
    
    def get_key_status(self) -> Dict[str, Dict]:
        """Get status of all API keys."""
        status = {}
        for provider, keys in self.api_keys.items():
            status[provider] = {
                "total_keys": len(keys),
                "active_keys": sum(1 for k in keys if k.is_active),
                "keys": []
            }
            
            for key in keys:
                status[provider]["keys"].append({
                    "is_active": key.is_active,
                    "error_count": key.error_count,
                    "max_errors": key.max_errors,
                    "last_used": key.last_used.isoformat() if key.last_used else None
                })
        
        return status

# Global instance
api_key_manager = APIKeyManager()

def get_api_key(provider: str) -> Optional[str]:
    """Get an active API key for the specified provider."""
    api_key = api_key_manager.get_active_key(provider)
    return api_key.key if api_key else None

def report_api_error(provider: str, error_type: str = "general"):
    """Report an API error for the current key."""
    api_key = api_key_manager.get_active_key(provider)
    if api_key:
        api_key_manager.report_error(provider, api_key, error_type)

def report_api_success(provider: str):
    """Report successful API usage."""
    api_key = api_key_manager.get_active_key(provider)
    if api_key:
        api_key_manager.report_success(provider, api_key)
