#!/usr/bin/env python3
"""
Enhanced Risk Scoring System
Comprehensive risk assessment for cryptocurrency addresses and transactions

This module provides:
- Multi-factor risk scoring
- Threat intelligence integration
- Real-time risk monitoring
- Risk trend analysis
- Compliance reporting
- Machine learning-based risk prediction
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import requests
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskCategory(Enum):
    """Risk categories for comprehensive assessment"""
    TRANSACTION_VOLUME = "transaction_volume"
    TRANSACTION_FREQUENCY = "transaction_frequency"
    NETWORK_CENTRALITY = "network_centrality"
    TEMPORAL_PATTERNS = "temporal_patterns"
    THREAT_INTELLIGENCE = "threat_intelligence"
    BEHAVIORAL_ANOMALIES = "behavioral_anomalies"
    COMPLIANCE_VIOLATIONS = "compliance_violations"
    CROSS_BLOCKCHAIN = "cross_blockchain"

class RiskLevel(Enum):
    """Risk levels with numerical scores"""
    CRITICAL = (5, 0.8, 1.0)
    HIGH = (4, 0.6, 0.8)
    MEDIUM = (3, 0.4, 0.6)
    LOW = (2, 0.2, 0.4)
    CLEAN = (1, 0.0, 0.2)

@dataclass
class RiskFactor:
    """Individual risk factor assessment"""
    category: RiskCategory
    score: float
    weight: float
    confidence: float
    evidence: Dict[str, any]
    timestamp: datetime

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment for an address"""
    address: str
    overall_risk_score: float
    risk_level: RiskLevel
    risk_factors: List[RiskFactor]
    threat_intelligence: Dict[str, any]
    behavioral_profile: Dict[str, any]
    compliance_status: Dict[str, any]
    risk_trend: List[Tuple[datetime, float]]
    recommendations: List[str]
    assessment_timestamp: datetime
    next_assessment: datetime

class ThreatIntelligenceProvider:
    """Threat intelligence data provider"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.cache = {}
        self.cache_ttl = timedelta(hours=1)
        
    def get_address_intelligence(self, address: str) -> Dict[str, any]:
        """Get threat intelligence for an address"""
        # Check cache first
        if address in self.cache:
            cached_data, timestamp = self.cache[address]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_data
        
        intelligence = {
            'bitcoinwhoswho': self._query_bitcoinwhoswho(address),
            'chainalysis': self._query_chainalysis(address),
            'blockchain_info': self._query_blockchain_info(address),
            'custom_sources': self._query_custom_sources(address)
        }
        
        # Cache the result
        self.cache[address] = (intelligence, datetime.now())
        
        return intelligence
    
    def _query_bitcoinwhoswho(self, address: str) -> Dict[str, any]:
        """Query BitcoinWhosWho for address intelligence"""
        # Simplified implementation - in production, this would make actual API calls
        return {
            'score': 0.0,
            'scam_reports': 0,
            'confidence': 0.0,
            'tags': [],
            'last_updated': datetime.now().isoformat()
        }
    
    def _query_chainalysis(self, address: str) -> Dict[str, any]:
        """Query Chainalysis API for address intelligence"""
        # Simplified implementation
        return {
            'risk_level': 'UNKNOWN',
            'category': 'UNKNOWN',
            'confidence': 0.0,
            'last_updated': datetime.now().isoformat()
        }
    
    def _query_blockchain_info(self, address: str) -> Dict[str, any]:
        """Query Blockchain.info for address information"""
        # Simplified implementation
        return {
            'balance': 0.0,
            'transaction_count': 0,
            'first_seen': None,
            'last_seen': None
        }
    
    def _query_custom_sources(self, address: str) -> Dict[str, any]:
        """Query custom threat intelligence sources"""
        # This could include internal databases, law enforcement feeds, etc.
        return {
            'internal_risk_score': 0.0,
            'law_enforcement_flags': [],
            'compliance_violations': [],
            'last_updated': datetime.now().isoformat()
        }

class EnhancedRiskScorer:
    """
    Enhanced risk scoring system with comprehensive threat intelligence integration
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the enhanced risk scorer"""
        self.config = config or {}
        self.threat_intel = ThreatIntelligenceProvider(config.get('threat_intelligence', {}))
        
        # Risk factor weights (sum to 1.0)
        self.risk_weights = {
            RiskCategory.TRANSACTION_VOLUME: 0.15,
            RiskCategory.TRANSACTION_FREQUENCY: 0.10,
            RiskCategory.NETWORK_CENTRALITY: 0.15,
            RiskCategory.TEMPORAL_PATTERNS: 0.10,
            RiskCategory.THREAT_INTELLIGENCE: 0.25,
            RiskCategory.BEHAVIORAL_ANOMALIES: 0.15,
            RiskCategory.COMPLIANCE_VIOLATIONS: 0.05,
            RiskCategory.CROSS_BLOCKCHAIN: 0.05
        }
        
        # Machine learning models
        self.ml_models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
        logger.info("Enhanced Risk Scorer initialized")

    def calculate_comprehensive_risk_score(self, address: str, graph: nx.DiGraph, 
                                         transactions: List[Dict], 
                                         laundering_patterns: List[Dict] = None) -> RiskAssessment:
        """
        Calculate comprehensive risk score for an address
        """
        logger.info(f"Calculating comprehensive risk score for {address}")
        
        # Get threat intelligence
        threat_intelligence = self.threat_intel.get_address_intelligence(address)
        
        # Calculate individual risk factors
        risk_factors = []
        
        # Transaction volume risk
        volume_factor = self._calculate_volume_risk(address, transactions)
        risk_factors.append(volume_factor)
        
        # Transaction frequency risk
        frequency_factor = self._calculate_frequency_risk(address, transactions)
        risk_factors.append(frequency_factor)
        
        # Network centrality risk
        centrality_factor = self._calculate_centrality_risk(address, graph)
        risk_factors.append(centrality_factor)
        
        # Temporal patterns risk
        temporal_factor = self._calculate_temporal_risk(address, transactions)
        risk_factors.append(temporal_factor)
        
        # Threat intelligence risk
        threat_factor = self._calculate_threat_intelligence_risk(address, threat_intelligence)
        risk_factors.append(threat_factor)
        
        # Behavioral anomalies risk
        behavioral_factor = self._calculate_behavioral_risk(address, transactions, laundering_patterns)
        risk_factors.append(behavioral_factor)
        
        # Compliance violations risk
        compliance_factor = self._calculate_compliance_risk(address, transactions)
        risk_factors.append(compliance_factor)
        
        # Cross-blockchain risk
        cross_blockchain_factor = self._calculate_cross_blockchain_risk(address, transactions)
        risk_factors.append(cross_blockchain_factor)
        
        # Calculate overall risk score
        overall_score = self._calculate_overall_score(risk_factors)
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_score)
        
        # Generate behavioral profile
        behavioral_profile = self._generate_behavioral_profile(address, transactions)
        
        # Generate compliance status
        compliance_status = self._generate_compliance_status(address, transactions)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_factors, threat_intelligence)
        
        # Create risk trend (simplified)
        risk_trend = self._generate_risk_trend(address, overall_score)
        
        assessment = RiskAssessment(
            address=address,
            overall_risk_score=overall_score,
            risk_level=risk_level,
            risk_factors=risk_factors,
            threat_intelligence=threat_intelligence,
            behavioral_profile=behavioral_profile,
            compliance_status=compliance_status,
            risk_trend=risk_trend,
            recommendations=recommendations,
            assessment_timestamp=datetime.now(),
            next_assessment=datetime.now() + timedelta(hours=24)
        )
        
        logger.info(f"Risk assessment completed for {address}: {risk_level.name} ({overall_score:.3f})")
        return assessment

    def _calculate_volume_risk(self, address: str, transactions: List[Dict]) -> RiskFactor:
        """Calculate transaction volume risk"""
        address_transactions = [tx for tx in transactions 
                               if tx.get('from_address') == address or tx.get('to_address') == address]
        
        if not address_transactions:
            return RiskFactor(
                category=RiskCategory.TRANSACTION_VOLUME,
                score=0.0,
                weight=self.risk_weights[RiskCategory.TRANSACTION_VOLUME],
                confidence=1.0,
                evidence={'transaction_count': 0, 'total_volume': 0.0},
                timestamp=datetime.now()
            )
        
        total_volume = sum(tx.get('value', 0) for tx in address_transactions)
        avg_volume = total_volume / len(address_transactions)
        
        # Risk increases with volume (logarithmic scale)
        volume_score = min(np.log10(max(total_volume, 1)) / 10.0, 1.0)
        
        return RiskFactor(
            category=RiskCategory.TRANSACTION_VOLUME,
            score=volume_score,
            weight=self.risk_weights[RiskCategory.TRANSACTION_VOLUME],
            confidence=0.9,
            evidence={
                'transaction_count': len(address_transactions),
                'total_volume': total_volume,
                'average_volume': avg_volume,
                'volume_percentile': 0.8  # Placeholder
            },
            timestamp=datetime.now()
        )

    def _calculate_frequency_risk(self, address: str, transactions: List[Dict]) -> RiskFactor:
        """Calculate transaction frequency risk"""
        address_transactions = [tx for tx in transactions 
                               if tx.get('from_address') == address or tx.get('to_address') == address]
        
        if len(address_transactions) < 2:
            return RiskFactor(
                category=RiskCategory.TRANSACTION_FREQUENCY,
                score=0.0,
                weight=self.risk_weights[RiskCategory.TRANSACTION_FREQUENCY],
                confidence=1.0,
                evidence={'transaction_count': len(address_transactions)},
                timestamp=datetime.now()
            )
        
        # Calculate transaction intervals
        timestamps = [datetime.fromisoformat(tx.get('timestamp', datetime.now().isoformat())) 
                     for tx in address_transactions]
        timestamps.sort()
        
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600  # hours
                    for i in range(len(timestamps)-1)]
        
        avg_interval = np.mean(intervals) if intervals else 0
        interval_variance = np.var(intervals) if len(intervals) > 1 else 0
        
        # High frequency and irregular patterns are suspicious
        frequency_score = min(1.0 / max(avg_interval, 1), 1.0) if avg_interval > 0 else 0
        irregularity_score = min(interval_variance / 100, 1.0)  # Normalize variance
        
        combined_score = (frequency_score + irregularity_score) / 2
        
        return RiskFactor(
            category=RiskCategory.TRANSACTION_FREQUENCY,
            score=combined_score,
            weight=self.risk_weights[RiskCategory.TRANSACTION_FREQUENCY],
            confidence=0.8,
            evidence={
                'transaction_count': len(address_transactions),
                'average_interval_hours': avg_interval,
                'interval_variance': interval_variance,
                'frequency_score': frequency_score,
                'irregularity_score': irregularity_score
            },
            timestamp=datetime.now()
        )

    def _calculate_centrality_risk(self, address: str, graph: nx.DiGraph) -> RiskFactor:
        """Calculate network centrality risk"""
        if not graph.has_node(address):
            return RiskFactor(
                category=RiskCategory.NETWORK_CENTRALITY,
                score=0.0,
                weight=self.risk_weights[RiskCategory.NETWORK_CENTRALITY],
                confidence=1.0,
                evidence={'centrality_measures': {}},
                timestamp=datetime.now()
            )
        
        try:
            # Calculate various centrality measures
            degree_centrality = nx.degree_centrality(graph).get(address, 0.0)
            betweenness_centrality = nx.betweenness_centrality(graph).get(address, 0.0)
            closeness_centrality = nx.closeness_centrality(graph).get(address, 0.0)
            eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000).get(address, 0.0)
            
            # Combine centrality measures
            centrality_score = (degree_centrality + betweenness_centrality + 
                              closeness_centrality + eigenvector_centrality) / 4
            
        except Exception as e:
            logger.warning(f"Centrality calculation failed for {address}: {e}")
            centrality_score = 0.0
            degree_centrality = betweenness_centrality = closeness_centrality = eigenvector_centrality = 0.0
        
        return RiskFactor(
            category=RiskCategory.NETWORK_CENTRALITY,
            score=centrality_score,
            weight=self.risk_weights[RiskCategory.NETWORK_CENTRALITY],
            confidence=0.7,
            evidence={
                'degree_centrality': degree_centrality,
                'betweenness_centrality': betweenness_centrality,
                'closeness_centrality': closeness_centrality,
                'eigenvector_centrality': eigenvector_centrality,
                'combined_score': centrality_score
            },
            timestamp=datetime.now()
        )

    def _calculate_temporal_risk(self, address: str, transactions: List[Dict]) -> RiskFactor:
        """Calculate temporal pattern risk"""
        address_transactions = [tx for tx in transactions 
                               if tx.get('from_address') == address or tx.get('to_address') == address]
        
        if len(address_transactions) < 3:
            return RiskFactor(
                category=RiskCategory.TEMPORAL_PATTERNS,
                score=0.0,
                weight=self.risk_weights[RiskCategory.TEMPORAL_PATTERNS],
                confidence=1.0,
                evidence={'transaction_count': len(address_transactions)},
                timestamp=datetime.now()
            )
        
        # Analyze temporal patterns
        timestamps = [datetime.fromisoformat(tx.get('timestamp', datetime.now().isoformat())) 
                     for tx in address_transactions]
        
        # Check for unusual timing patterns
        hours = [ts.hour for ts in timestamps]
        days = [ts.weekday() for ts in timestamps]
        
        # Unusual hours (late night/early morning) are more suspicious
        unusual_hours = sum(1 for h in hours if h < 6 or h > 22)
        unusual_hour_ratio = unusual_hours / len(hours)
        
        # Weekend activity might be suspicious for certain types
        weekend_activity = sum(1 for d in days if d >= 5)  # Saturday=5, Sunday=6
        weekend_ratio = weekend_activity / len(days)
        
        # Rapid succession of transactions
        rapid_transactions = 0
        for i in range(len(timestamps)-1):
            if (timestamps[i+1] - timestamps[i]).total_seconds() < 300:  # 5 minutes
                rapid_transactions += 1
        
        rapid_ratio = rapid_transactions / max(len(timestamps)-1, 1)
        
        # Combine temporal risk factors
        temporal_score = (unusual_hour_ratio + weekend_ratio + rapid_ratio) / 3
        
        return RiskFactor(
            category=RiskCategory.TEMPORAL_PATTERNS,
            score=temporal_score,
            weight=self.risk_weights[RiskCategory.TEMPORAL_PATTERNS],
            confidence=0.8,
            evidence={
                'transaction_count': len(address_transactions),
                'unusual_hour_ratio': unusual_hour_ratio,
                'weekend_ratio': weekend_ratio,
                'rapid_transaction_ratio': rapid_ratio,
                'temporal_score': temporal_score
            },
            timestamp=datetime.now()
        )

    def _calculate_threat_intelligence_risk(self, address: str, threat_intelligence: Dict) -> RiskFactor:
        """Calculate threat intelligence risk"""
        risk_score = 0.0
        confidence = 0.0
        evidence = {}
        
        # BitcoinWhosWho risk
        bww_data = threat_intelligence.get('bitcoinwhoswho', {})
        bww_score = bww_data.get('score', 0.0)
        bww_confidence = bww_data.get('confidence', 0.0)
        evidence['bitcoinwhoswho'] = bww_data
        
        # Chainalysis risk
        chainalysis_data = threat_intelligence.get('chainalysis', {})
        chainalysis_risk_level = chainalysis_data.get('risk_level', 'UNKNOWN')
        chainalysis_confidence = chainalysis_data.get('confidence', 0.0)
        
        # Map Chainalysis risk levels to scores
        chainalysis_score_map = {
            'CRITICAL': 1.0,
            'HIGH': 0.8,
            'MEDIUM': 0.6,
            'LOW': 0.3,
            'UNKNOWN': 0.0
        }
        chainalysis_score = chainalysis_score_map.get(chainalysis_risk_level, 0.0)
        evidence['chainalysis'] = chainalysis_data
        
        # Custom sources risk
        custom_data = threat_intelligence.get('custom_sources', {})
        custom_score = custom_data.get('internal_risk_score', 0.0)
        evidence['custom_sources'] = custom_data
        
        # Combine threat intelligence scores
        scores = [bww_score, chainalysis_score, custom_score]
        confidences = [bww_confidence, chainalysis_confidence, 0.8]  # Custom sources assumed high confidence
        
        # Weighted average based on confidence
        if sum(confidences) > 0:
            risk_score = sum(s * c for s, c in zip(scores, confidences)) / sum(confidences)
            confidence = np.mean(confidences)
        
        return RiskFactor(
            category=RiskCategory.THREAT_INTELLIGENCE,
            score=risk_score,
            weight=self.risk_weights[RiskCategory.THREAT_INTELLIGENCE],
            confidence=confidence,
            evidence=evidence,
            timestamp=datetime.now()
        )

    def _calculate_behavioral_risk(self, address: str, transactions: List[Dict], 
                                 laundering_patterns: List[Dict] = None) -> RiskFactor:
        """Calculate behavioral anomaly risk"""
        address_transactions = [tx for tx in transactions 
                               if tx.get('from_address') == address or tx.get('to_address') == address]
        
        behavioral_score = 0.0
        evidence = {'anomalies': []}
        
        # Check for laundering patterns involving this address
        if laundering_patterns:
            address_patterns = [p for p in laundering_patterns if address in p.get('addresses_involved', [])]
            if address_patterns:
                pattern_risk = sum(p.get('risk_score', 0.0) for p in address_patterns) / len(address_patterns)
                behavioral_score += pattern_risk * 0.5
                evidence['laundering_patterns'] = len(address_patterns)
        
        # Check for unusual transaction amounts
        amounts = [tx.get('value', 0) for tx in address_transactions]
        if amounts:
            # Round amounts are suspicious
            round_amounts = sum(1 for amount in amounts if amount % 100000000 == 0)  # Round BTC amounts
            round_ratio = round_amounts / len(amounts)
            behavioral_score += round_ratio * 0.3
            
            # Very small or very large amounts
            avg_amount = np.mean(amounts)
            if avg_amount < 1000 or avg_amount > 100000000000:  # Very small or very large
                behavioral_score += 0.2
            
            evidence['round_amount_ratio'] = round_ratio
            evidence['average_amount'] = avg_amount
        
        # Check for unusual transaction patterns
        if len(address_transactions) > 10:
            # Many small transactions (smurfing)
            small_transactions = sum(1 for tx in address_transactions if tx.get('value', 0) < 1000000)
            smurfing_ratio = small_transactions / len(address_transactions)
            behavioral_score += smurfing_ratio * 0.4
            evidence['smurfing_ratio'] = smurfing_ratio
        
        return RiskFactor(
            category=RiskCategory.BEHAVIORAL_ANOMALIES,
            score=min(behavioral_score, 1.0),
            weight=self.risk_weights[RiskCategory.BEHAVIORAL_ANOMALIES],
            confidence=0.7,
            evidence=evidence,
            timestamp=datetime.now()
        )

    def _calculate_compliance_risk(self, address: str, transactions: List[Dict]) -> RiskFactor:
        """Calculate compliance violation risk"""
        # Simplified compliance checks
        address_transactions = [tx for tx in transactions 
                               if tx.get('from_address') == address or tx.get('to_address') == address]
        
        compliance_score = 0.0
        violations = []
        
        # Check for transactions above reporting thresholds
        large_transactions = [tx for tx in address_transactions if tx.get('value', 0) > 10000000000]  # 100 BTC
        if large_transactions:
            compliance_score += 0.3
            violations.append(f"Large transactions: {len(large_transactions)}")
        
        # Check for rapid transactions (potential structuring)
        if len(address_transactions) > 5:
            timestamps = [datetime.fromisoformat(tx.get('timestamp', datetime.now().isoformat())) 
                         for tx in address_transactions]
            timestamps.sort()
            
            rapid_count = 0
            for i in range(len(timestamps)-1):
                if (timestamps[i+1] - timestamps[i]).total_seconds() < 60:  # 1 minute
                    rapid_count += 1
            
            if rapid_count > 3:
                compliance_score += 0.4
                violations.append(f"Rapid transactions: {rapid_count}")
        
        return RiskFactor(
            category=RiskCategory.COMPLIANCE_VIOLATIONS,
            score=min(compliance_score, 1.0),
            weight=self.risk_weights[RiskCategory.COMPLIANCE_VIOLATIONS],
            confidence=0.8,
            evidence={'violations': violations, 'violation_count': len(violations)},
            timestamp=datetime.now()
        )

    def _calculate_cross_blockchain_risk(self, address: str, transactions: List[Dict]) -> RiskFactor:
        """Calculate cross-blockchain risk"""
        # Simplified implementation - in production, this would analyze multi-blockchain data
        return RiskFactor(
            category=RiskCategory.CROSS_BLOCKCHAIN,
            score=0.0,
            weight=self.risk_weights[RiskCategory.CROSS_BLOCKCHAIN],
            confidence=0.5,
            evidence={'cross_blockchain_transactions': 0},
            timestamp=datetime.now()
        )

    def _calculate_overall_score(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate overall risk score from individual factors"""
        weighted_score = 0.0
        total_weight = 0.0
        
        for factor in risk_factors:
            weighted_score += factor.score * factor.weight * factor.confidence
            total_weight += factor.weight * factor.confidence
        
        if total_weight > 0:
            overall_score = weighted_score / total_weight
        else:
            overall_score = 0.0
        
        return min(overall_score, 1.0)

    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level from score"""
        if score >= RiskLevel.CRITICAL.value[1]:
            return RiskLevel.CRITICAL
        elif score >= RiskLevel.HIGH.value[1]:
            return RiskLevel.HIGH
        elif score >= RiskLevel.MEDIUM.value[1]:
            return RiskLevel.MEDIUM
        elif score >= RiskLevel.LOW.value[1]:
            return RiskLevel.LOW
        else:
            return RiskLevel.CLEAN

    def _generate_behavioral_profile(self, address: str, transactions: List[Dict]) -> Dict[str, any]:
        """Generate behavioral profile for address"""
        address_transactions = [tx for tx in transactions 
                               if tx.get('from_address') == address or tx.get('to_address') == address]
        
        if not address_transactions:
            return {'profile_type': 'inactive', 'confidence': 1.0}
        
        # Analyze transaction patterns
        amounts = [tx.get('value', 0) for tx in address_transactions]
        timestamps = [datetime.fromisoformat(tx.get('timestamp', datetime.now().isoformat())) 
                     for tx in address_transactions]
        
        profile = {
            'transaction_count': len(address_transactions),
            'total_volume': sum(amounts),
            'average_transaction_size': np.mean(amounts) if amounts else 0,
            'transaction_frequency': len(address_transactions) / max((max(timestamps) - min(timestamps)).days, 1),
            'activity_hours': list(set(ts.hour for ts in timestamps)),
            'activity_days': list(set(ts.weekday() for ts in timestamps)),
            'profile_type': 'unknown',
            'confidence': 0.7
        }
        
        # Determine profile type
        if profile['transaction_frequency'] > 10:  # More than 10 transactions per day
            profile['profile_type'] = 'high_frequency'
        elif profile['average_transaction_size'] > 1000000000:  # Large transactions
            profile['profile_type'] = 'whale'
        elif len(profile['activity_hours']) > 16:  # Active throughout the day
            profile['profile_type'] = 'bot_or_service'
        else:
            profile['profile_type'] = 'regular_user'
        
        return profile

    def _generate_compliance_status(self, address: str, transactions: List[Dict]) -> Dict[str, any]:
        """Generate compliance status for address"""
        address_transactions = [tx for tx in transactions 
                               if tx.get('from_address') == address or tx.get('to_address') == address]
        
        status = {
            'sar_threshold_exceeded': False,
            'ctr_threshold_exceeded': False,
            'structuring_detected': False,
            'compliance_score': 1.0,
            'violations': []
        }
        
        if not address_transactions:
            return status
        
        # Check for SAR threshold (simplified)
        total_volume = sum(tx.get('value', 0) for tx in address_transactions)
        if total_volume > 100000000000:  # 1000 BTC
            status['sar_threshold_exceeded'] = True
            status['compliance_score'] -= 0.3
            status['violations'].append('SAR threshold exceeded')
        
        # Check for CTR threshold (simplified)
        large_transactions = [tx for tx in address_transactions if tx.get('value', 0) > 10000000000]  # 100 BTC
        if len(large_transactions) > 0:
            status['ctr_threshold_exceeded'] = True
            status['compliance_score'] -= 0.2
            status['violations'].append('CTR threshold exceeded')
        
        # Check for structuring
        if len(address_transactions) > 10:
            amounts = [tx.get('value', 0) for tx in address_transactions]
            small_transactions = sum(1 for amount in amounts if amount < 1000000)  # 0.01 BTC
            if small_transactions / len(amounts) > 0.8:
                status['structuring_detected'] = True
                status['compliance_score'] -= 0.4
                status['violations'].append('Potential structuring detected')
        
        status['compliance_score'] = max(status['compliance_score'], 0.0)
        
        return status

    def _generate_recommendations(self, risk_factors: List[RiskFactor], 
                                threat_intelligence: Dict) -> List[str]:
        """Generate actionable recommendations based on risk assessment"""
        recommendations = []
        
        # High threat intelligence risk
        threat_factor = next((f for f in risk_factors if f.category == RiskCategory.THREAT_INTELLIGENCE), None)
        if threat_factor and threat_factor.score > 0.7:
            recommendations.append("Immediate investigation required - high threat intelligence risk")
        
        # High behavioral risk
        behavioral_factor = next((f for f in risk_factors if f.category == RiskCategory.BEHAVIORAL_ANOMALIES), None)
        if behavioral_factor and behavioral_factor.score > 0.6:
            recommendations.append("Monitor for money laundering patterns - behavioral anomalies detected")
        
        # Compliance violations
        compliance_factor = next((f for f in risk_factors if f.category == RiskCategory.COMPLIANCE_VIOLATIONS), None)
        if compliance_factor and compliance_factor.score > 0.5:
            recommendations.append("Review compliance requirements - violations detected")
        
        # High volume risk
        volume_factor = next((f for f in risk_factors if f.category == RiskCategory.TRANSACTION_VOLUME), None)
        if volume_factor and volume_factor.score > 0.8:
            recommendations.append("Consider enhanced due diligence - high transaction volume")
        
        # Network centrality
        centrality_factor = next((f for f in risk_factors if f.category == RiskCategory.NETWORK_CENTRALITY), None)
        if centrality_factor and centrality_factor.score > 0.7:
            recommendations.append("Investigate network connections - high centrality suggests hub activity")
        
        if not recommendations:
            recommendations.append("Continue monitoring - no immediate concerns identified")
        
        return recommendations

    def _generate_risk_trend(self, address: str, current_score: float) -> List[Tuple[datetime, float]]:
        """Generate risk trend data (simplified implementation)"""
        # In production, this would retrieve historical risk scores
        trend = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(30):
            timestamp = base_time + timedelta(days=i)
            # Simulate trend with some variation
            score = current_score + np.random.normal(0, 0.05)
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            trend.append((timestamp, score))
        
        return trend

    def train_ml_models(self, training_data: List[Dict]) -> bool:
        """Train machine learning models for risk prediction"""
        logger.info("Training machine learning models for risk prediction")
        
        try:
            # Prepare training data
            df = pd.DataFrame(training_data)
            
            # Feature engineering
            features = self._extract_features(df)
            labels = df['risk_level'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest classifier
            self.ml_models['random_forest'] = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            self.ml_models['random_forest'].fit(X_train_scaled, y_train)
            
            # Train Isolation Forest for anomaly detection
            self.ml_models['isolation_forest'] = IsolationForest(
                contamination=0.1, random_state=42
            )
            self.ml_models['isolation_forest'].fit(X_train_scaled)
            
            # Evaluate models
            rf_score = self.ml_models['random_forest'].score(X_test_scaled, y_test)
            logger.info(f"Random Forest accuracy: {rf_score:.3f}")
            
            self.is_trained = True
            logger.info("Machine learning models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train ML models: {e}")
            return False

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for machine learning"""
        # Simplified feature extraction
        features = []
        
        for _, row in df.iterrows():
            feature_vector = [
                row.get('transaction_count', 0),
                row.get('total_volume', 0.0),
                row.get('average_transaction_size', 0.0),
                row.get('transaction_frequency', 0.0),
                row.get('degree_centrality', 0.0),
                row.get('betweenness_centrality', 0.0),
                row.get('threat_intelligence_score', 0.0),
                row.get('behavioral_anomaly_score', 0.0)
            ]
            features.append(feature_vector)
        
        return np.array(features)

    def predict_risk(self, address_data: Dict) -> Dict[str, any]:
        """Predict risk using trained ML models"""
        if not self.is_trained:
            return {'error': 'Models not trained'}
        
        try:
            # Extract features
            features = self._extract_features(pd.DataFrame([address_data]))
            features_scaled = self.scaler.transform(features)
            
            # Predict risk level
            risk_prediction = self.ml_models['random_forest'].predict(features_scaled)[0]
            risk_probability = self.ml_models['random_forest'].predict_proba(features_scaled)[0]
            
            # Detect anomalies
            anomaly_score = self.ml_models['isolation_forest'].decision_function(features_scaled)[0]
            is_anomaly = self.ml_models['isolation_forest'].predict(features_scaled)[0] == -1
            
            return {
                'predicted_risk_level': risk_prediction,
                'risk_probabilities': dict(zip(
                    self.ml_models['random_forest'].classes_, 
                    risk_probability
                )),
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly,
                'confidence': max(risk_probability)
            }
            
        except Exception as e:
            logger.error(f"Risk prediction failed: {e}")
            return {'error': str(e)}

    def export_risk_assessment(self, assessment: RiskAssessment, format: str = 'json') -> str:
        """Export risk assessment to file"""
        timestamp = assessment.assessment_timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"risk_assessment_{assessment.address}_{timestamp}.{format}"
        
        if format == 'json':
            # Convert to serializable format
            data = {
                'address': assessment.address,
                'overall_risk_score': assessment.overall_risk_score,
                'risk_level': assessment.risk_level.name,
                'risk_factors': [
                    {
                        'category': factor.category.name,
                        'score': factor.score,
                        'weight': factor.weight,
                        'confidence': factor.confidence,
                        'evidence': factor.evidence,
                        'timestamp': factor.timestamp.isoformat()
                    }
                    for factor in assessment.risk_factors
                ],
                'threat_intelligence': assessment.threat_intelligence,
                'behavioral_profile': assessment.behavioral_profile,
                'compliance_status': assessment.compliance_status,
                'risk_trend': [
                    {'timestamp': timestamp.isoformat(), 'score': score}
                    for timestamp, score in assessment.risk_trend
                ],
                'recommendations': assessment.recommendations,
                'assessment_timestamp': assessment.assessment_timestamp.isoformat(),
                'next_assessment': assessment.next_assessment.isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Risk assessment exported to {filename}")
        return filename

# Example usage
if __name__ == "__main__":
    # Initialize risk scorer
    scorer = EnhancedRiskScorer()
    
    # Sample data
    sample_transactions = [
        {
            'tx_hash': 'tx1',
            'from_address': 'addr1',
            'to_address': 'addr2',
            'value': 100000000,
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    # Create sample graph
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_edge('addr1', 'addr2')
    
    # Calculate risk assessment
    assessment = scorer.calculate_comprehensive_risk_score('addr1', graph, sample_transactions)
    
    print(f"Risk Assessment for {assessment.address}:")
    print(f"Risk Level: {assessment.risk_level.name}")
    print(f"Risk Score: {assessment.overall_risk_score:.3f}")
    print(f"Recommendations: {assessment.recommendations}")
    
    # Export assessment
    scorer.export_risk_assessment(assessment)
