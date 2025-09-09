#!/usr/bin/env python3
"""
Performance Optimization System for ChainBreak
Implements caching, batch processing, and rate limiting.
"""

import logging
import time
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import threading
from collections import defaultdict, deque
import requests

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cached result."""
    data: Any
    timestamp: datetime
    ttl: int  # Time to live in seconds
    access_count: int = 0
    last_accessed: datetime = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.utcnow() > self.timestamp + timedelta(seconds=self.ttl)
    
    def access(self):
        """Record cache access."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()

@dataclass
class RateLimit:
    """Represents rate limiting configuration."""
    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int
    current_minute: deque
    current_hour: deque
    last_reset_minute: datetime
    last_reset_hour: datetime

class PerformanceOptimizer:
    """Optimizes performance through caching and rate limiting."""
    
    def __init__(self, cache_dir: str = "cache", max_cache_size: int = 1000):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, CacheEntry] = {}
        self.rate_limits: Dict[str, RateLimit] = {}
        self.batch_queues: Dict[str, List] = defaultdict(list)
        self.batch_lock = threading.Lock()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize rate limits
        self._init_rate_limits()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _init_rate_limits(self):
        """Initialize rate limits for different APIs."""
        self.rate_limits = {
            "chainalysis": RateLimit(
                requests_per_minute=60,
                requests_per_hour=1000,
                burst_limit=10,
                current_minute=deque(),
                current_hour=deque(),
                last_reset_minute=datetime.utcnow(),
                last_reset_hour=datetime.utcnow()
            ),
            "blockchain_info": RateLimit(
                requests_per_minute=30,
                requests_per_hour=500,
                burst_limit=5,
                current_minute=deque(),
                current_hour=deque(),
                last_reset_minute=datetime.utcnow(),
                last_reset_hour=datetime.utcnow()
            ),
            "bitcoinwhoswho": RateLimit(
                requests_per_minute=20,
                requests_per_hour=300,
                burst_limit=3,
                current_minute=deque(),
                current_hour=deque(),
                last_reset_minute=datetime.utcnow(),
                last_reset_hour=datetime.utcnow()
            )
        }
    
    def _start_cleanup_thread(self):
        """Start background thread for cache cleanup."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(300)  # Cleanup every 5 minutes
                    self._cleanup_cache()
                    self._cleanup_rate_limits()
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        expired_keys = []
        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        # Remove oldest entries if cache is too large
        if len(self.cache) > self.max_cache_size:
            sorted_entries = sorted(self.cache.items(), 
                                  key=lambda x: x[1].last_accessed)
            excess = len(self.cache) - self.max_cache_size
            for i in range(excess):
                del self.cache[sorted_entries[i][0]]
            
            logger.info(f"Removed {excess} oldest cache entries")
    
    def _cleanup_rate_limits(self):
        """Clean up rate limit tracking."""
        now = datetime.utcnow()
        
        for provider, rate_limit in self.rate_limits.items():
            # Clean up minute-based tracking
            if (now - rate_limit.last_reset_minute).total_seconds() >= 60:
                rate_limit.current_minute.clear()
                rate_limit.last_reset_minute = now
            
            # Clean up hour-based tracking
            if (now - rate_limit.last_reset_hour).total_seconds() >= 3600:
                rate_limit.current_hour.clear()
                rate_limit.last_reset_hour = now
    
    def _generate_cache_key(self, provider: str, address: str, operation: str = "check") -> str:
        """Generate a cache key for an operation."""
        key_data = f"{provider}:{address}:{operation}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_result(self, provider: str, address: str, operation: str = "check") -> Optional[Any]:
        """Get a cached result if available."""
        cache_key = self._generate_cache_key(provider, address, operation)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if not entry.is_expired():
                entry.access()
                logger.debug(f"Cache hit for {provider}:{address}:{operation}")
                return entry.data
            else:
                del self.cache[cache_key]
        
        logger.debug(f"Cache miss for {provider}:{address}:{operation}")
        return None
    
    def cache_result(self, provider: str, address: str, data: Any, 
                    ttl: int = 3600, operation: str = "check"):
        """Cache a result."""
        cache_key = self._generate_cache_key(provider, address, operation)
        
        entry = CacheEntry(
            data=data,
            timestamp=datetime.utcnow(),
            ttl=ttl
        )
        
        self.cache[cache_key] = entry
        logger.debug(f"Cached result for {provider}:{address}:{operation}")
    
    def check_rate_limit(self, provider: str) -> Tuple[bool, float]:
        """Check if request is within rate limits."""
        if provider not in self.rate_limits:
            return True, 0.0
        
        rate_limit = self.rate_limits[provider]
        now = datetime.utcnow()
        
        # Check minute-based limit
        if len(rate_limit.current_minute) >= rate_limit.requests_per_minute:
            # Calculate wait time
            oldest_request = rate_limit.current_minute[0]
            wait_time = 60 - (now - oldest_request).total_seconds()
            if wait_time > 0:
                return False, wait_time
        
        # Check hour-based limit
        if len(rate_limit.current_hour) >= rate_limit.requests_per_hour:
            # Calculate wait time
            oldest_request = rate_limit.current_hour[0]
            wait_time = 3600 - (now - oldest_request).total_seconds()
            if wait_time > 0:
                return False, wait_time
        
        return True, 0.0
    
    def record_request(self, provider: str):
        """Record a request for rate limiting."""
        if provider not in self.rate_limits:
            return
        
        rate_limit = self.rate_limits[provider]
        now = datetime.utcnow()
        
        # Add to minute tracking
        rate_limit.current_minute.append(now)
        
        # Add to hour tracking
        rate_limit.current_hour.append(now)
    
    def wait_for_rate_limit(self, provider: str, max_wait: float = 60.0) -> bool:
        """Wait for rate limit to reset if necessary."""
        can_proceed, wait_time = self.check_rate_limit(provider)
        
        if not can_proceed:
            if wait_time > max_wait:
                logger.warning(f"Rate limit wait time too long for {provider}: {wait_time}s")
                return False
            
            logger.info(f"Waiting {wait_time:.1f}s for rate limit reset for {provider}")
            time.sleep(wait_time)
        
        return True
    
    def batch_process(self, provider: str, addresses: List[str], 
                     process_func, batch_size: int = 10) -> Dict[str, Any]:
        """Process addresses in batches to optimize API usage."""
        results = {}
        
        with self.batch_lock:
            # Add to batch queue
            self.batch_queues[provider].extend(addresses)
            
            # Process if batch is full
            if len(self.batch_queues[provider]) >= batch_size:
                batch = self.batch_queues[provider][:batch_size]
                self.batch_queues[provider] = self.batch_queues[provider][batch_size:]
                
                # Process batch
                batch_results = process_func(batch)
                results.update(batch_results)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_stats = {
            "total_entries": len(self.cache),
            "hit_rate": 0.0,
            "total_accesses": 0
        }
        
        if self.cache:
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            cache_stats["total_accesses"] = total_accesses
            
            # Calculate hit rate (simplified)
            cache_stats["hit_rate"] = min(1.0, total_accesses / len(self.cache))
        
        rate_limit_stats = {}
        for provider, rate_limit in self.rate_limits.items():
            rate_limit_stats[provider] = {
                "requests_this_minute": len(rate_limit.current_minute),
                "requests_this_hour": len(rate_limit.current_hour),
                "minute_limit": rate_limit.requests_per_minute,
                "hour_limit": rate_limit.requests_per_hour
            }
        
        return {
            "cache": cache_stats,
            "rate_limits": rate_limit_stats,
            "batch_queues": {k: len(v) for k, v in self.batch_queues.items()}
        }
    
    def clear_cache(self, provider: Optional[str] = None):
        """Clear cache entries."""
        if provider:
            # Clear entries for specific provider
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(provider)]
            for key in keys_to_remove:
                del self.cache[key]
            logger.info(f"Cleared cache for provider: {provider}")
        else:
            # Clear all cache
            self.cache.clear()
            logger.info("Cleared all cache entries")

# Global instance
performance_optimizer = PerformanceOptimizer()

def get_cached_result(provider: str, address: str, operation: str = "check") -> Optional[Any]:
    """Get a cached result if available."""
    return performance_optimizer.get_cached_result(provider, address, operation)

def cache_result(provider: str, address: str, data: Any, ttl: int = 3600, operation: str = "check"):
    """Cache a result."""
    performance_optimizer.cache_result(provider, address, data, ttl, operation)

def check_rate_limit(provider: str) -> Tuple[bool, float]:
    """Check if request is within rate limits."""
    return performance_optimizer.check_rate_limit(provider)

def wait_for_rate_limit(provider: str, max_wait: float = 60.0) -> bool:
    """Wait for rate limit to reset if necessary."""
    return performance_optimizer.wait_for_rate_limit(provider, max_wait)
