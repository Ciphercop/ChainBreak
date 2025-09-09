#!/usr/bin/env python3
"""
Test script for the Illicit Transaction Detector
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from illicit_transaction_detector import (
    IllicitTransactionDetector,
    Transaction,
    GraphVisualizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_transactions():
    """Create test transactions with various suspicious patterns."""
    transactions = []
    base_time = datetime.now() - timedelta(days=7)
    
    # Test addresses
    addresses = [
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",  # Genesis block
        "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",  # Satoshi's address
        "1FeexV6bAHb8ybZjqQMjJrcCrHGW9sb6uF",  # Exchange address
        "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s",  # Exchange address
        "1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ",  # Exchange address
    ]
    
    # Create peel chain pattern
    logger.info("Creating peel chain pattern...")
    peel_amounts = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0]
    for i, amount in enumerate(peel_amounts):
        transactions.append(Transaction(
            tx_hash=f"peel_tx_{i:03d}",
            from_address=addresses[0],
            to_address=addresses[1],
            value=amount,
            timestamp=base_time + timedelta(minutes=i * 10),
            block_height=800000 + i,
            fee=0.001,
            confirmations=6
        ))
    
    # Create mixing pattern
    logger.info("Creating mixing pattern...")
    mixing_inputs = addresses[1:3]
    mixing_outputs = addresses[3:5]
    for i in range(20):
        tx_time = base_time + timedelta(minutes=i % 5)  # Within 5 minutes
        transactions.append(Transaction(
            tx_hash=f"mix_tx_{i:03d}",
            from_address=mixing_inputs[i % len(mixing_inputs)],
            to_address=mixing_outputs[i % len(mixing_outputs)],
            value=0.5 + (i % 3) * 0.1,
            timestamp=tx_time,
            block_height=800010 + i,
            fee=0.001,
            confirmations=6
        ))
    
    # Create smurfing pattern
    logger.info("Creating smurfing pattern...")
    large_amount = 100.0
    small_amount = 0.01
    num_small_txs = 50
    for i in range(num_small_txs):
        transactions.append(Transaction(
            tx_hash=f"smurf_tx_{i:03d}",
            from_address=addresses[0],
            to_address=f"smurf_addr_{i:03d}",
            value=small_amount,
            timestamp=base_time + timedelta(minutes=i),
            block_height=800030 + i,
            fee=0.001,
            confirmations=6
        ))
    
    # Create rapid transfers
    logger.info("Creating rapid transfer pattern...")
    for i in range(10):
        transactions.append(Transaction(
            tx_hash=f"rapid_tx_{i:03d}",
            from_address=addresses[1],
            to_address=f"rapid_addr_{i:03d}",
            value=1.0,
            timestamp=base_time + timedelta(seconds=i * 30),  # 30 seconds apart
            block_height=800080 + i,
            fee=0.001,
            confirmations=6
        ))
    
    # Create round amounts
    logger.info("Creating round amount pattern...")
    round_amounts = [1.0, 10.0, 100.0, 1000.0, 0.1, 0.01]
    for i, amount in enumerate(round_amounts):
        transactions.append(Transaction(
            tx_hash=f"round_tx_{i:03d}",
            from_address=addresses[2],
            to_address=f"round_addr_{i:03d}",
            value=amount,
            timestamp=base_time + timedelta(hours=i),
            block_height=800090 + i,
            fee=0.001,
            confirmations=6
        ))
    
    logger.info(f"Created {len(transactions)} test transactions")
    return transactions

def test_detector():
    """Test the illicit transaction detector."""
    logger.info("Starting illicit transaction detector test...")
    
    # Initialize detector without Chainalysis API key for testing
    # (The API key provided appears to be expired/invalid)
    detector = IllicitTransactionDetector(chainalysis_api_key=None)
    
    # Create test transactions
    transactions = create_test_transactions()
    
    # Perform analysis
    logger.info("Performing illicit transaction analysis...")
    analysis = detector.analyze_transactions(transactions)
    
    # Print results
    logger.info("Analysis Results:")
    logger.info(f"  Total addresses: {analysis.total_addresses}")
    logger.info(f"  Total transactions: {analysis.total_transactions}")
    logger.info(f"  High-risk addresses: {len(analysis.high_risk_addresses)}")
    logger.info(f"  Suspicious patterns detected: {len(analysis.suspicious_patterns)}")
    
    # Print risk distribution
    logger.info("Risk Distribution:")
    for risk_level, count in analysis.risk_distribution.items():
        logger.info(f"  {risk_level}: {count}")
    
    # Print detected patterns
    logger.info("Detected Patterns:")
    for i, pattern in enumerate(analysis.suspicious_patterns):
        logger.info(f"  {i+1}. {pattern.pattern_type.value}: {pattern.description}")
        logger.info(f"     Confidence: {pattern.confidence:.2f}, Risk Score: {pattern.risk_score:.2f}")
        logger.info(f"     Addresses: {len(pattern.addresses)}, Transactions: {len(pattern.transactions)}")
    
    # Print high-risk addresses
    if analysis.high_risk_addresses:
        logger.info("High-Risk Addresses:")
        for address in analysis.high_risk_addresses[:5]:  # Show top 5
            node = analysis.addresses[address]
            logger.info(f"  {address}: {node.risk_level.value} (score: {node.risk_score:.3f})")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    graph = detector._build_transaction_graph(transactions)
    viz_files = detector.create_visualizations(analysis, graph, "test_visualizations")
    
    logger.info("Visualization files created:")
    for viz_type, file_path in viz_files.items():
        if viz_type != 'graph_data':
            logger.info(f"  {viz_type}: {file_path}")
    
    logger.info("Test completed successfully!")
    return analysis

def test_visualizer():
    """Test the graph visualizer independently."""
    logger.info("Testing graph visualizer...")
    
    visualizer = GraphVisualizer()
    
    # Test color palette
    logger.info("Color palette:")
    for risk_level, color in visualizer.color_palette.items():
        logger.info(f"  {risk_level}: {color}")
    
    logger.info("Pattern colors:")
    for pattern, color in visualizer.pattern_colors.items():
        logger.info(f"  {pattern}: {color}")
    
    logger.info("Graph visualizer test completed!")

if __name__ == "__main__":
    try:
        # Test visualizer
        test_visualizer()
        
        # Test detector
        analysis = test_detector()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)
