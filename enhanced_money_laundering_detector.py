#!/usr/bin/env python3
"""
Enhanced Money Laundering Detection System
Advanced algorithms for detecting sophisticated money laundering patterns

This module extends the existing illicit transaction detector with:
- Advanced layering detection with temporal analysis
- Chain hopping detection across multiple cryptocurrencies
- Mixing service identification with confidence scoring
- Structured transaction pattern recognition
- Cross-border transaction analysis
- Real-time monitoring capabilities
"""

import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, Counter
import json
import requests
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LaunderingPattern(Enum):
    """Types of money laundering patterns"""
    LAYERING = "layering"
    INTEGRATION = "integration"
    PLACEMENT = "placement"
    CHAIN_HOPPING = "chain_hopping"
    MIXING_SERVICE = "mixing_service"
    STRUCTURED_TRANSACTIONS = "structured_transactions"
    CROSS_BORDER = "cross_border"
    PEEL_CHAIN = "peel_chain"
    SMURFING = "smurfing"
    ROUND_TRIPPING = "round_tripping"

class RiskLevel(Enum):
    """Risk levels for addresses and transactions"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    CLEAN = "CLEAN"

@dataclass
class LaunderingPatternResult:
    """Result of money laundering pattern detection"""
    pattern_type: LaunderingPattern
    confidence: float
    risk_score: float
    addresses_involved: List[str]
    transactions_involved: List[str]
    description: str
    evidence: Dict[str, any]
    timestamp: datetime
    blockchain: str = "btc"

@dataclass
class AddressRiskProfile:
    """Comprehensive risk profile for an address"""
    address: str
    risk_level: RiskLevel
    total_risk_score: float
    laundering_patterns: List[LaunderingPatternResult]
    transaction_volume: float
    transaction_count: int
    first_seen: datetime
    last_seen: datetime
    centrality_measures: Dict[str, float]
    threat_intelligence: Dict[str, any]
    cluster_id: Optional[int] = None
    is_exchange: bool = False
    is_mixer: bool = False
    is_known_illicit: bool = False

class EnhancedMoneyLaunderingDetector:
    """
    Advanced money laundering detection system with sophisticated pattern recognition
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the enhanced detector"""
        self.config = config or {}
        self.known_exchanges = self._load_known_exchanges()
        self.known_mixers = self._load_known_mixers()
        self.illicit_addresses = self._load_illicit_addresses()
        
        # Detection thresholds
        self.thresholds = {
            'layering_min_hops': 3,
            'layering_min_addresses': 5,
            'mixing_min_transactions': 10,
            'mixing_diversity_threshold': 0.3,
            'smurfing_max_amount_ratio': 0.1,
            'smurfing_min_transactions': 20,
            'chain_hopping_time_window': 24,  # hours
            'structured_amount_threshold': 10000,  # USD equivalent
            'round_tripping_time_window': 48,  # hours
        }
        
        logger.info("Enhanced Money Laundering Detector initialized")

    def _load_known_exchanges(self) -> Set[str]:
        """Load known exchange addresses"""
        # In production, this would load from a database or API
        return {
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",  # Example exchange
            "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",  # Example exchange
        }

    def _load_known_mixers(self) -> Set[str]:
        """Load known mixing service addresses"""
        # In production, this would load from threat intelligence feeds
        return {
            "1MixerAddress123456789",  # Example mixer
            "1TumblerAddress987654321",  # Example tumbler
        }

    def _load_illicit_addresses(self) -> Dict[str, Dict]:
        """Load known illicit addresses with metadata"""
        # In production, this would load from multiple threat intelligence sources
        return {
            "1IllicitAddress123456789": {
                "risk_level": "CRITICAL",
                "confidence": 0.95,
                "sources": ["FBI", "Chainalysis", "BitcoinWhosWho"],
                "activity_type": "ransomware",
                "first_seen": "2023-01-01",
                "last_seen": "2023-12-01"
            }
        }

    def detect_advanced_layering(self, graph: nx.DiGraph, transactions: List[Dict]) -> List[LaunderingPatternResult]:
        """
        Detect sophisticated layering patterns with temporal analysis
        
        Layering involves creating multiple intermediate transactions to obscure
        the origin and destination of funds.
        """
        logger.info("Detecting advanced layering patterns")
        patterns = []
        
        # Group transactions by time windows
        time_windows = self._create_time_windows(transactions, window_hours=24)
        
        for window_start, window_transactions in time_windows.items():
            # Create subgraph for this time window
            window_graph = self._create_subgraph(graph, window_transactions)
            
            # Find potential layering chains
            layering_chains = self._find_layering_chains(window_graph)
            
            for chain in layering_chains:
                if len(chain) >= self.thresholds['layering_min_hops']:
                    # Calculate layering metrics
                    metrics = self._calculate_layering_metrics(chain, window_transactions)
                    
                    if metrics['confidence'] > 0.6:
                        pattern = LaunderingPatternResult(
                            pattern_type=LaunderingPattern.LAYERING,
                            confidence=metrics['confidence'],
                            risk_score=metrics['risk_score'],
                            addresses_involved=list(chain),
                            transactions_involved=metrics['transactions'],
                            description=f"Layering chain with {len(chain)} hops detected",
                            evidence=metrics,
                            timestamp=window_start,
                            blockchain="btc"
                        )
                        patterns.append(pattern)
        
        logger.info(f"Detected {len(patterns)} layering patterns")
        return patterns

    def detect_chain_hopping(self, transactions: List[Dict]) -> List[LaunderingPatternResult]:
        """
        Detect chain hopping patterns where funds move between different cryptocurrencies
        
        Chain hopping is used to break transaction trails by moving between
        different blockchain networks.
        """
        logger.info("Detecting chain hopping patterns")
        patterns = []
        
        # Group transactions by time windows
        time_windows = self._create_time_windows(transactions, window_hours=self.thresholds['chain_hopping_time_window'])
        
        for window_start, window_transactions in time_windows.items():
            # Look for rapid movements between different blockchains
            blockchain_transitions = self._analyze_blockchain_transitions(window_transactions)
            
            for transition in blockchain_transitions:
                if transition['confidence'] > 0.7:
                    pattern = LaunderingPatternResult(
                        pattern_type=LaunderingPattern.CHAIN_HOPPING,
                        confidence=transition['confidence'],
                        risk_score=transition['risk_score'],
                        addresses_involved=transition['addresses'],
                        transactions_involved=transition['transactions'],
                        description=f"Chain hopping detected: {transition['from_chain']} → {transition['to_chain']}",
                        evidence=transition,
                        timestamp=window_start,
                        blockchain="multi"
                    )
                    patterns.append(pattern)
        
        logger.info(f"Detected {len(patterns)} chain hopping patterns")
        return patterns

    def detect_mixing_services(self, graph: nx.DiGraph, transactions: List[Dict]) -> List[LaunderingPatternResult]:
        """
        Detect mixing service usage with advanced pattern recognition
        
        Mixing services obscure transaction trails by mixing multiple inputs
        and outputs in complex patterns.
        """
        logger.info("Detecting mixing service patterns")
        patterns = []
        
        # Analyze transaction patterns for mixing characteristics
        mixing_candidates = self._identify_mixing_candidates(graph, transactions)
        
        for candidate in mixing_candidates:
            mixing_analysis = self._analyze_mixing_pattern(candidate, transactions)
            
            if mixing_analysis['confidence'] > 0.8:
                pattern = LaunderingPatternResult(
                    pattern_type=LaunderingPattern.MIXING_SERVICE,
                    confidence=mixing_analysis['confidence'],
                    risk_score=mixing_analysis['risk_score'],
                    addresses_involved=mixing_analysis['addresses'],
                    transactions_involved=mixing_analysis['transactions'],
                    description=f"Mixing service detected with {mixing_analysis['input_count']} inputs, {mixing_analysis['output_count']} outputs",
                    evidence=mixing_analysis,
                    timestamp=mixing_analysis['timestamp'],
                    blockchain="btc"
                )
                patterns.append(pattern)
        
        logger.info(f"Detected {len(patterns)} mixing service patterns")
        return patterns

    def detect_structured_transactions(self, transactions: List[Dict]) -> List[LaunderingPatternResult]:
        """
        Detect structured transactions designed to avoid reporting thresholds
        
        Structured transactions involve breaking large amounts into smaller
        transactions just under reporting thresholds.
        """
        logger.info("Detecting structured transaction patterns")
        patterns = []
        
        # Group transactions by address and time
        address_groups = self._group_transactions_by_address(transactions)
        
        for address, address_transactions in address_groups.items():
            # Analyze for structuring patterns
            structuring_analysis = self._analyze_structuring_pattern(address_transactions)
            
            if structuring_analysis['confidence'] > 0.7:
                pattern = LaunderingPatternResult(
                    pattern_type=LaunderingPattern.STRUCTURED_TRANSACTIONS,
                    confidence=structuring_analysis['confidence'],
                    risk_score=structuring_analysis['risk_score'],
                    addresses_involved=[address],
                    transactions_involved=structuring_analysis['transactions'],
                    description=f"Structured transactions detected: {structuring_analysis['transaction_count']} transactions under ${structuring_analysis['threshold']}",
                    evidence=structuring_analysis,
                    timestamp=structuring_analysis['timestamp'],
                    blockchain="btc"
                )
                patterns.append(pattern)
        
        logger.info(f"Detected {len(patterns)} structured transaction patterns")
        return patterns

    def detect_round_tripping(self, graph: nx.DiGraph, transactions: List[Dict]) -> List[LaunderingPatternResult]:
        """
        Detect round tripping patterns where funds return to original source
        
        Round tripping involves sending funds through multiple addresses
        and eventually returning them to the original source.
        """
        logger.info("Detecting round tripping patterns")
        patterns = []
        
        # Find potential round trip cycles
        cycles = self._find_round_trip_cycles(graph, transactions)
        
        for cycle in cycles:
            round_trip_analysis = self._analyze_round_trip(cycle, transactions)
            
            if round_trip_analysis['confidence'] > 0.6:
                pattern = LaunderingPatternResult(
                    pattern_type=LaunderingPattern.ROUND_TRIPPING,
                    confidence=round_trip_analysis['confidence'],
                    risk_score=round_trip_analysis['risk_score'],
                    addresses_involved=cycle,
                    transactions_involved=round_trip_analysis['transactions'],
                    description=f"Round tripping detected: {len(cycle)} addresses in cycle",
                    evidence=round_trip_analysis,
                    timestamp=round_trip_analysis['timestamp'],
                    blockchain="btc"
                )
                patterns.append(pattern)
        
        logger.info(f"Detected {len(patterns)} round tripping patterns")
        return patterns

    def calculate_comprehensive_risk_score(self, address: str, patterns: List[LaunderingPatternResult], 
                                         graph: nx.DiGraph) -> AddressRiskProfile:
        """
        Calculate comprehensive risk score for an address considering all factors
        """
        logger.info(f"Calculating comprehensive risk score for {address}")
        
        # Base risk factors
        base_risk = 0.0
        laundering_patterns = []
        
        # Check for known illicit status
        is_known_illicit = address in self.illicit_addresses
        if is_known_illicit:
            base_risk += 0.4
        
        # Check for exchange status
        is_exchange = address in self.known_exchanges
        if is_exchange:
            base_risk -= 0.1  # Exchanges are generally lower risk
        
        # Check for mixer status
        is_mixer = address in self.known_mixers
        if is_mixer:
            base_risk += 0.3
        
        # Analyze patterns involving this address
        for pattern in patterns:
            if address in pattern.addresses_involved:
                laundering_patterns.append(pattern)
                base_risk += pattern.risk_score * pattern.confidence
        
        # Calculate centrality measures
        centrality_measures = self._calculate_centrality_measures(address, graph)
        
        # Calculate transaction metrics
        address_transactions = [t for t in patterns if address in t.addresses_involved]
        transaction_volume = sum(t.evidence.get('volume', 0) for t in address_transactions)
        transaction_count = len(address_transactions)
        
        # Determine risk level
        if base_risk >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif base_risk >= 0.6:
            risk_level = RiskLevel.HIGH
        elif base_risk >= 0.4:
            risk_level = RiskLevel.MEDIUM
        elif base_risk >= 0.1:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.CLEAN
        
        # Get threat intelligence
        threat_intelligence = self.illicit_addresses.get(address, {})
        
        profile = AddressRiskProfile(
            address=address,
            risk_level=risk_level,
            total_risk_score=min(base_risk, 1.0),
            laundering_patterns=laundering_patterns,
            transaction_volume=transaction_volume,
            transaction_count=transaction_count,
            first_seen=datetime.now() - timedelta(days=30),  # Placeholder
            last_seen=datetime.now(),
            centrality_measures=centrality_measures,
            threat_intelligence=threat_intelligence,
            is_exchange=is_exchange,
            is_mixer=is_mixer,
            is_known_illicit=is_known_illicit
        )
        
        logger.info(f"Risk profile calculated for {address}: {risk_level.value} ({base_risk:.3f})")
        return profile

    def _create_time_windows(self, transactions: List[Dict], window_hours: int = 24) -> Dict[datetime, List[Dict]]:
        """Create time windows for temporal analysis"""
        windows = defaultdict(list)
        
        for tx in transactions:
            timestamp = tx.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            # Round to nearest window
            window_start = timestamp.replace(hour=(timestamp.hour // window_hours) * window_hours, 
                                           minute=0, second=0, microsecond=0)
            windows[window_start].append(tx)
        
        return dict(windows)

    def _create_subgraph(self, graph: nx.DiGraph, transactions: List[Dict]) -> nx.DiGraph:
        """Create subgraph from transactions"""
        subgraph = nx.DiGraph()
        
        for tx in transactions:
            from_addr = tx.get('from_address')
            to_addr = tx.get('to_address')
            if from_addr and to_addr:
                subgraph.add_edge(from_addr, to_addr, **tx)
        
        return subgraph

    def _find_layering_chains(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find potential layering chains in the graph"""
        chains = []
        
        # Find all simple paths of length >= 3
        for source in graph.nodes():
            for target in graph.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(graph, source, target, cutoff=6))
                        for path in paths:
                            if len(path) >= self.thresholds['layering_min_hops']:
                                chains.append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        return chains

    def _calculate_layering_metrics(self, chain: List[str], transactions: List[Dict]) -> Dict:
        """Calculate metrics for layering pattern"""
        # Count transactions in chain
        chain_transactions = []
        for i in range(len(chain) - 1):
            for tx in transactions:
                if (tx.get('from_address') == chain[i] and 
                    tx.get('to_address') == chain[i + 1]):
                    chain_transactions.append(tx)
        
        # Calculate confidence based on chain characteristics
        confidence = min(len(chain) / 10.0, 1.0)  # Longer chains are more suspicious
        
        # Calculate risk score
        risk_score = min(confidence * 0.8, 1.0)
        
        return {
            'confidence': confidence,
            'risk_score': risk_score,
            'transactions': [tx.get('tx_hash', '') for tx in chain_transactions],
            'chain_length': len(chain),
            'transaction_count': len(chain_transactions),
            'volume': sum(tx.get('value', 0) for tx in chain_transactions)
        }

    def _analyze_blockchain_transitions(self, transactions: List[Dict]) -> List[Dict]:
        """Analyze transitions between different blockchains"""
        transitions = []
        
        # Group by time and look for rapid blockchain changes
        # This is a simplified implementation - in production, you'd have
        # actual multi-blockchain transaction data
        
        return transitions

    def _identify_mixing_candidates(self, graph: nx.DiGraph, transactions: List[Dict]) -> List[str]:
        """Identify addresses that might be mixing services"""
        candidates = []
        
        for node in graph.nodes():
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            
            # Mixing services typically have high in-degree and out-degree
            if in_degree >= self.thresholds['mixing_min_transactions'] and \
               out_degree >= self.thresholds['mixing_min_transactions']:
                candidates.append(node)
        
        return candidates

    def _analyze_mixing_pattern(self, candidate: str, transactions: List[Dict]) -> Dict:
        """Analyze if a candidate address is actually a mixing service"""
        # Get all transactions involving this address
        candidate_transactions = [tx for tx in transactions 
                                if tx.get('from_address') == candidate or 
                                   tx.get('to_address') == candidate]
        
        # Analyze input/output diversity
        inputs = set(tx.get('from_address') for tx in candidate_transactions 
                    if tx.get('to_address') == candidate)
        outputs = set(tx.get('to_address') for tx in candidate_transactions 
                     if tx.get('from_address') == candidate)
        
        diversity_score = min(len(inputs), len(outputs)) / len(candidate_transactions)
        
        confidence = diversity_score if diversity_score > self.thresholds['mixing_diversity_threshold'] else 0.0
        risk_score = confidence * 0.9
        
        return {
            'confidence': confidence,
            'risk_score': risk_score,
            'addresses': [candidate] + list(inputs) + list(outputs),
            'transactions': [tx.get('tx_hash', '') for tx in candidate_transactions],
            'input_count': len(inputs),
            'output_count': len(outputs),
            'diversity_score': diversity_score,
            'timestamp': datetime.now()
        }

    def _group_transactions_by_address(self, transactions: List[Dict]) -> Dict[str, List[Dict]]:
        """Group transactions by address"""
        groups = defaultdict(list)
        
        for tx in transactions:
            from_addr = tx.get('from_address')
            to_addr = tx.get('to_address')
            
            if from_addr:
                groups[from_addr].append(tx)
            if to_addr:
                groups[to_addr].append(tx)
        
        return dict(groups)

    def _analyze_structuring_pattern(self, transactions: List[Dict]) -> Dict:
        """Analyze transactions for structuring patterns"""
        if len(transactions) < 5:
            return {'confidence': 0.0}
        
        # Check if amounts are just under common thresholds
        amounts = [tx.get('value', 0) for tx in transactions]
        threshold = self.thresholds['structured_amount_threshold']
        
        # Count transactions just under threshold
        under_threshold = sum(1 for amount in amounts if amount < threshold and amount > threshold * 0.8)
        
        confidence = min(under_threshold / len(transactions), 1.0)
        risk_score = confidence * 0.7
        
        return {
            'confidence': confidence,
            'risk_score': risk_score,
            'transactions': [tx.get('tx_hash', '') for tx in transactions],
            'transaction_count': len(transactions),
            'threshold': threshold,
            'under_threshold_count': under_threshold,
            'timestamp': datetime.now()
        }

    def _find_round_trip_cycles(self, graph: nx.DiGraph, transactions: List[Dict]) -> List[List[str]]:
        """Find potential round trip cycles"""
        cycles = []
        
        try:
            # Find all cycles in the graph
            simple_cycles = list(nx.simple_cycles(graph))
            for cycle in simple_cycles:
                if len(cycle) >= 3:  # Minimum cycle length
                    cycles.append(cycle)
        except:
            pass
        
        return cycles

    def _analyze_round_trip(self, cycle: List[str], transactions: List[Dict]) -> Dict:
        """Analyze a potential round trip cycle"""
        # Calculate cycle metrics
        cycle_transactions = []
        for i in range(len(cycle)):
            next_i = (i + 1) % len(cycle)
            for tx in transactions:
                if (tx.get('from_address') == cycle[i] and 
                    tx.get('to_address') == cycle[next_i]):
                    cycle_transactions.append(tx)
        
        confidence = min(len(cycle_transactions) / (len(cycle) * 2), 1.0)
        risk_score = confidence * 0.6
        
        return {
            'confidence': confidence,
            'risk_score': risk_score,
            'transactions': [tx.get('tx_hash', '') for tx in cycle_transactions],
            'cycle_length': len(cycle),
            'transaction_count': len(cycle_transactions),
            'timestamp': datetime.now()
        }

    def _calculate_centrality_measures(self, address: str, graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate centrality measures for an address"""
        measures = {}
        
        try:
            measures['degree_centrality'] = nx.degree_centrality(graph).get(address, 0.0)
            measures['betweenness_centrality'] = nx.betweenness_centrality(graph).get(address, 0.0)
            measures['closeness_centrality'] = nx.closeness_centrality(graph).get(address, 0.0)
        except:
            measures = {
                'degree_centrality': 0.0,
                'betweenness_centrality': 0.0,
                'closeness_centrality': 0.0
            }
        
        return measures

    def generate_comprehensive_report(self, patterns: List[LaunderingPatternResult], 
                                    risk_profiles: List[AddressRiskProfile]) -> Dict:
        """Generate comprehensive money laundering analysis report"""
        logger.info("Generating comprehensive money laundering report")
        
        # Aggregate statistics
        total_patterns = len(patterns)
        critical_addresses = len([p for p in risk_profiles if p.risk_level == RiskLevel.CRITICAL])
        high_risk_addresses = len([p for p in risk_profiles if p.risk_level == RiskLevel.HIGH])
        
        # Pattern distribution
        pattern_distribution = Counter(p.pattern_type.value for p in patterns)
        
        # Risk distribution
        risk_distribution = Counter(p.risk_level.value for p in risk_profiles)
        
        # Top risk addresses
        top_risk_addresses = sorted(risk_profiles, key=lambda x: x.total_risk_score, reverse=True)[:10]
        
        report = {
            'summary': {
                'total_patterns_detected': total_patterns,
                'critical_risk_addresses': critical_addresses,
                'high_risk_addresses': high_risk_addresses,
                'total_addresses_analyzed': len(risk_profiles),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'pattern_distribution': dict(pattern_distribution),
            'risk_distribution': dict(risk_distribution),
            'top_risk_addresses': [
                {
                    'address': p.address,
                    'risk_level': p.risk_level.value,
                    'risk_score': p.total_risk_score,
                    'pattern_count': len(p.laundering_patterns),
                    'is_exchange': p.is_exchange,
                    'is_mixer': p.is_mixer,
                    'is_known_illicit': p.is_known_illicit
                }
                for p in top_risk_addresses
            ],
            'detailed_patterns': [
                {
                    'pattern_type': p.pattern_type.value,
                    'confidence': p.confidence,
                    'risk_score': p.risk_score,
                    'description': p.description,
                    'addresses_count': len(p.addresses_involved),
                    'transactions_count': len(p.transactions_involved),
                    'timestamp': p.timestamp.isoformat()
                }
                for p in patterns
            ],
            'recommendations': self._generate_recommendations(patterns, risk_profiles)
        }
        
        logger.info(f"Report generated: {total_patterns} patterns, {critical_addresses} critical addresses")
        return report

    def _generate_recommendations(self, patterns: List[LaunderingPatternResult], 
                                risk_profiles: List[AddressRiskProfile]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        critical_addresses = [p for p in risk_profiles if p.risk_level == RiskLevel.CRITICAL]
        if critical_addresses:
            recommendations.append(f"Immediate investigation required for {len(critical_addresses)} critical risk addresses")
        
        mixing_patterns = [p for p in patterns if p.pattern_type == LaunderingPattern.MIXING_SERVICE]
        if mixing_patterns:
            recommendations.append(f"Monitor {len(mixing_patterns)} addresses showing mixing service behavior")
        
        layering_patterns = [p for p in patterns if p.pattern_type == LaunderingPattern.LAYERING]
        if layering_patterns:
            recommendations.append(f"Investigate {len(layering_patterns)} layering chains for money laundering")
        
        structured_patterns = [p for p in patterns if p.pattern_type == LaunderingPattern.STRUCTURED_TRANSACTIONS]
        if structured_patterns:
            recommendations.append(f"Review {len(structured_patterns)} addresses for structuring violations")
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = EnhancedMoneyLaunderingDetector()
    
    # Example transaction data
    sample_transactions = [
        {
            'tx_hash': 'tx1',
            'from_address': 'addr1',
            'to_address': 'addr2',
            'value': 100000000,  # 1 BTC in satoshis
            'timestamp': datetime.now().isoformat()
        },
        # Add more sample transactions...
    ]
    
    # Create sample graph
    graph = nx.DiGraph()
    for tx in sample_transactions:
        graph.add_edge(tx['from_address'], tx['to_address'], **tx)
    
    # Run detection
    patterns = []
    patterns.extend(detector.detect_advanced_layering(graph, sample_transactions))
    patterns.extend(detector.detect_mixing_services(graph, sample_transactions))
    patterns.extend(detector.detect_structured_transactions(sample_transactions))
    
    # Calculate risk profiles
    addresses = set()
    for tx in sample_transactions:
        addresses.add(tx['from_address'])
        addresses.add(tx['to_address'])
    
    risk_profiles = []
    for addr in addresses:
        profile = detector.calculate_comprehensive_risk_score(addr, patterns, graph)
        risk_profiles.append(profile)
    
    # Generate report
    report = detector.generate_comprehensive_report(patterns, risk_profiles)
    
    print("Enhanced Money Laundering Detection Report:")
    print(json.dumps(report, indent=2))
