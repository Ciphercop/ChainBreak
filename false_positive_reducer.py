#!/usr/bin/env python3
"""
False Positive Reduction System for ChainBreak
Implements whitelisting, context-aware scoring, and legitimate pattern recognition.
"""

import logging
import json
import os
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)

@dataclass
class WhitelistEntry:
    """Represents a whitelisted address or pattern."""
    identifier: str
    entry_type: str  # 'address', 'exchange', 'merchant', 'pattern'
    reason: str
    confidence: float
    added_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class FalsePositiveReducer:
    """Reduces false positives through whitelisting and context analysis."""
    
    def __init__(self, config_file: str = "whitelist.json"):
        self.config_file = config_file
        self.whitelist: Dict[str, WhitelistEntry] = {}
        self.exchange_addresses: Set[str] = set()
        self.merchant_addresses: Set[str] = set()
        self.legitimate_patterns: Dict[str, Dict] = {}
        self.load_config()
        self.load_known_exchanges()
    
    def load_config(self):
        """Load whitelist configuration."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                for entry_data in config.get('whitelist', []):
                    entry = WhitelistEntry(
                        identifier=entry_data['identifier'],
                        entry_type=entry_data['entry_type'],
                        reason=entry_data['reason'],
                        confidence=entry_data['confidence'],
                        added_at=datetime.fromisoformat(entry_data['added_at']),
                        expires_at=datetime.fromisoformat(entry_data['expires_at']) if entry_data.get('expires_at') else None,
                        metadata=entry_data.get('metadata', {})
                    )
                    self.whitelist[entry.identifier] = entry
                
                logger.info(f"Loaded {len(self.whitelist)} whitelist entries")
                
            except Exception as e:
                logger.error(f"Error loading whitelist: {e}")
        else:
            self.create_default_config()
    
    def create_default_config(self):
        """Create default whitelist configuration."""
        default_config = {
            "whitelist": [
                {
                    "identifier": "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s",
                    "entry_type": "exchange",
                    "reason": "Known Binance exchange address",
                    "confidence": 0.95,
                    "added_at": datetime.utcnow().isoformat(),
                    "expires_at": None,
                    "metadata": {"exchange": "Binance", "verified": True}
                },
                {
                    "identifier": "1FeexV6bAHb8ybZjqQMjJrcCrHGW9sb6uF",
                    "entry_type": "exchange", 
                    "reason": "Known Binance exchange address",
                    "confidence": 0.95,
                    "added_at": datetime.utcnow().isoformat(),
                    "expires_at": None,
                    "metadata": {"exchange": "Binance", "verified": True}
                }
            ]
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2, default=str)
        logger.info("Created default whitelist configuration")
    
    def load_known_exchanges(self):
        """Load known exchange addresses from external sources."""
        try:
            # Load from CoinGecko or similar service
            # For now, use a static list of known exchanges
            known_exchanges = [
                "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s",  # Binance
                "1FeexV6bAHb8ybZjqQMjJrcCrHGW9sb6uF",  # Binance
                "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",  # Coinbase
                "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",  # Kraken
            ]
            
            for addr in known_exchanges:
                self.exchange_addresses.add(addr)
                if addr not in self.whitelist:
                    entry = WhitelistEntry(
                        identifier=addr,
                        entry_type="exchange",
                        reason="Known cryptocurrency exchange",
                        confidence=0.9,
                        added_at=datetime.utcnow()
                    )
                    self.whitelist[addr] = entry
            
            logger.info(f"Loaded {len(known_exchanges)} known exchange addresses")
            
        except Exception as e:
            logger.error(f"Error loading known exchanges: {e}")
    
    def is_whitelisted(self, address: str) -> Tuple[bool, Optional[WhitelistEntry]]:
        """Check if an address is whitelisted."""
        if address in self.whitelist:
            entry = self.whitelist[address]
            
            # Check if entry has expired
            if entry.expires_at and datetime.utcnow() > entry.expires_at:
                logger.info(f"Whitelist entry expired for {address}")
                return False, None
            
            return True, entry
        
        return False, None
    
    def analyze_context(self, address: str, transaction_history: List, 
                       risk_score: float, suspicious_patterns: List) -> Dict[str, any]:
        """Analyze context to determine if risk score should be reduced."""
        context_analysis = {
            "original_risk_score": risk_score,
            "adjusted_risk_score": risk_score,
            "reduction_factors": [],
            "confidence": 1.0
        }
        
        try:
            # Factor 1: Exchange interaction patterns
            if self._has_exchange_interaction_patterns(address, transaction_history):
                reduction = 0.3
                context_analysis["adjusted_risk_score"] = max(0.0, risk_score - reduction)
                context_analysis["reduction_factors"].append({
                    "factor": "exchange_interaction",
                    "reduction": reduction,
                    "reason": "Address shows legitimate exchange interaction patterns"
                })
            
            # Factor 2: Merchant payment patterns
            if self._has_merchant_payment_patterns(address, transaction_history):
                reduction = 0.2
                context_analysis["adjusted_risk_score"] = max(0.0, context_analysis["adjusted_risk_score"] - reduction)
                context_analysis["reduction_factors"].append({
                    "factor": "merchant_payment",
                    "reduction": reduction,
                    "reason": "Address shows legitimate merchant payment patterns"
                })
            
            # Factor 3: Regular user patterns
            if self._has_regular_user_patterns(address, transaction_history):
                reduction = 0.15
                context_analysis["adjusted_risk_score"] = max(0.0, context_analysis["adjusted_risk_score"] - reduction)
                context_analysis["reduction_factors"].append({
                    "factor": "regular_user",
                    "reduction": reduction,
                    "reason": "Address shows regular user transaction patterns"
                })
            
            # Factor 4: Time-based analysis
            if self._has_legitimate_timing_patterns(address, transaction_history):
                reduction = 0.1
                context_analysis["adjusted_risk_score"] = max(0.0, context_analysis["adjusted_risk_score"] - reduction)
                context_analysis["reduction_factors"].append({
                    "factor": "legitimate_timing",
                    "reduction": reduction,
                    "reason": "Address shows legitimate timing patterns"
                })
            
            # Factor 5: Amount analysis
            if self._has_legitimate_amount_patterns(address, transaction_history):
                reduction = 0.1
                context_analysis["adjusted_risk_score"] = max(0.0, context_analysis["adjusted_risk_score"] - reduction)
                context_analysis["reduction_factors"].append({
                    "factor": "legitimate_amounts",
                    "reduction": reduction,
                    "reason": "Address shows legitimate amount patterns"
                })
            
            # Calculate confidence based on number of reduction factors
            num_factors = len(context_analysis["reduction_factors"])
            context_analysis["confidence"] = min(1.0, 0.5 + (num_factors * 0.1))
            
        except Exception as e:
            logger.error(f"Error in context analysis for {address}: {e}")
        
        return context_analysis
    
    def _has_exchange_interaction_patterns(self, address: str, transaction_history: List) -> bool:
        """Check if address shows exchange interaction patterns."""
        try:
            # Look for transactions with known exchange addresses
            exchange_interactions = 0
            total_transactions = len(transaction_history)
            
            if total_transactions == 0:
                return False
            
            for tx in transaction_history:
                if (hasattr(tx, 'to_address') and tx.to_address in self.exchange_addresses) or \
                   (hasattr(tx, 'from_address') and tx.from_address in self.exchange_addresses):
                    exchange_interactions += 1
            
            # If more than 20% of transactions are with exchanges, likely legitimate
            return (exchange_interactions / total_transactions) > 0.2
            
        except Exception as e:
            logger.error(f"Error checking exchange interaction patterns: {e}")
            return False
    
    def _has_merchant_payment_patterns(self, address: str, transaction_history: List) -> bool:
        """Check if address shows merchant payment patterns."""
        try:
            # Look for regular payment amounts (not round numbers)
            non_round_payments = 0
            total_transactions = len(transaction_history)
            
            if total_transactions == 0:
                return False
            
            for tx in transaction_history:
                if hasattr(tx, 'value') and tx.value > 0:
                    # Check if amount is not suspiciously round
                    if not self._is_suspiciously_round(tx.value):
                        non_round_payments += 1
            
            # If more than 60% of payments are non-round, likely legitimate merchant
            return (non_round_payments / total_transactions) > 0.6
            
        except Exception as e:
            logger.error(f"Error checking merchant payment patterns: {e}")
            return False
    
    def _has_regular_user_patterns(self, address: str, transaction_history: List) -> bool:
        """Check if address shows regular user patterns."""
        try:
            total_transactions = len(transaction_history)
            
            if total_transactions < 5:
                return False
            
            # Check for regular transaction intervals
            timestamps = []
            for tx in transaction_history:
                if hasattr(tx, 'timestamp'):
                    timestamps.append(tx.timestamp)
            
            if len(timestamps) < 3:
                return False
            
            timestamps.sort()
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                intervals.append(interval)
            
            # Check for regular intervals (not too frequent, not too sparse)
            avg_interval = sum(intervals) / len(intervals)
            regular_intervals = sum(1 for interval in intervals if 3600 <= interval <= 86400 * 7)  # 1 hour to 1 week
            
            return (regular_intervals / len(intervals)) > 0.5
            
        except Exception as e:
            logger.error(f"Error checking regular user patterns: {e}")
            return False
    
    def _has_legitimate_timing_patterns(self, address: str, transaction_history: List) -> bool:
        """Check if address shows legitimate timing patterns."""
        try:
            # Check for transactions during business hours (UTC)
            business_hour_transactions = 0
            total_transactions = len(transaction_history)
            
            if total_transactions == 0:
                return False
            
            for tx in transaction_history:
                if hasattr(tx, 'timestamp'):
                    hour = tx.timestamp.hour
                    # Business hours: 6 AM to 10 PM UTC
                    if 6 <= hour <= 22:
                        business_hour_transactions += 1
            
            # If more than 40% of transactions are during business hours, likely legitimate
            return (business_hour_transactions / total_transactions) > 0.4
            
        except Exception as e:
            logger.error(f"Error checking legitimate timing patterns: {e}")
            return False
    
    def _has_legitimate_amount_patterns(self, address: str, transaction_history: List) -> bool:
        """Check if address shows legitimate amount patterns."""
        try:
            amounts = []
            for tx in transaction_history:
                if hasattr(tx, 'value') and tx.value > 0:
                    amounts.append(tx.value)
            
            if len(amounts) < 3:
                return False
            
            # Check for reasonable amount distribution (not all very small or very large)
            small_amounts = sum(1 for amount in amounts if amount < 0.01)  # Less than 0.01 BTC
            large_amounts = sum(1 for amount in amounts if amount > 10)    # More than 10 BTC
            
            total_amounts = len(amounts)
            
            # If not dominated by very small or very large amounts, likely legitimate
            return (small_amounts / total_amounts) < 0.8 and (large_amounts / total_amounts) < 0.3
            
        except Exception as e:
            logger.error(f"Error checking legitimate amount patterns: {e}")
            return False
    
    def _is_suspiciously_round(self, amount: float) -> bool:
        """Check if an amount is suspiciously round."""
        # Check for round numbers (1.0, 10.0, 100.0, etc.)
        if amount == 0:
            return False
        
        # Check if amount is a round number
        if amount == int(amount):
            return True
        
        # Check for round decimal places (0.1, 0.01, etc.)
        decimal_places = len(str(amount).split('.')[-1]) if '.' in str(amount) else 0
        if decimal_places <= 2 and amount * (10 ** decimal_places) == int(amount * (10 ** decimal_places)):
            return True
        
        return False
    
    def add_to_whitelist(self, identifier: str, entry_type: str, reason: str, 
                        confidence: float = 0.8, expires_days: Optional[int] = None):
        """Add an entry to the whitelist."""
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        entry = WhitelistEntry(
            identifier=identifier,
            entry_type=entry_type,
            reason=reason,
            confidence=confidence,
            added_at=datetime.utcnow(),
            expires_at=expires_at
        )
        
        self.whitelist[identifier] = entry
        self.save_config()
        logger.info(f"Added {identifier} to whitelist: {reason}")
    
    def save_config(self):
        """Save whitelist configuration."""
        try:
            config = {
                "whitelist": []
            }
            
            for entry in self.whitelist.values():
                config["whitelist"].append({
                    "identifier": entry.identifier,
                    "entry_type": entry.entry_type,
                    "reason": entry.reason,
                    "confidence": entry.confidence,
                    "added_at": entry.added_at.isoformat(),
                    "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                    "metadata": entry.metadata
                })
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving whitelist configuration: {e}")
    
    def get_whitelist_status(self) -> Dict:
        """Get status of the whitelist."""
        total_entries = len(self.whitelist)
        active_entries = sum(1 for entry in self.whitelist.values() 
                           if not entry.expires_at or datetime.utcnow() < entry.expires_at)
        
        return {
            "total_entries": total_entries,
            "active_entries": active_entries,
            "exchange_addresses": len(self.exchange_addresses),
            "merchant_addresses": len(self.merchant_addresses)
        }

# Global instance
false_positive_reducer = FalsePositiveReducer()
