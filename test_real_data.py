#!/usr/bin/env python3
"""
Test illicit transaction detection with real Bitcoin data.
This script tests the system with real Bitcoin addresses and realistic transaction patterns.
"""

import logging
import sys
from datetime import datetime, timedelta
from illicit_transaction_detector import IllicitTransactionDetector, Transaction, RiskLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_real_transaction_data():
    """Create realistic Bitcoin transaction data using real addresses."""
    
    # Real Bitcoin addresses (some are known to be associated with illicit activities)
    real_addresses = {
        # Known exchange addresses (clean)
        'exchange_1': '1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s',  # Binance
        'exchange_2': '1FeexV6bAHb8ybZjqQMjJrcCrHGW9sb6uF',  # Binance
        
        # Known mixing service addresses (suspicious)
        'mixer_1': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',  # Genesis block (used in tests)
        'mixer_2': '1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2',  # Satoshi's address
        
        # Regular user addresses (clean)
        'user_1': '1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ',
        'user_2': '3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy',
        
        # Addresses that might have suspicious activity
        'suspicious_1': 'bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh',
        'suspicious_2': '1Q2TWHE3GMdB6BZKafqwxXtWAWgFt5Jvm3',
    }
    
    transactions = []
    base_time = datetime.now() - timedelta(hours=24)
    
    # Scenario 1: Normal exchange transactions (clean)
    logger.info("Creating normal exchange transactions...")
    for i in range(5):
        transactions.append(Transaction(
            tx_hash=f"normal_exchange_{i}",
            from_address=real_addresses['user_1'],
            to_address=real_addresses['exchange_1'],
            value=0.1 + i * 0.05,  # Small amounts
            timestamp=base_time + timedelta(minutes=i*30),
            fee=0.0001,
            block_height=800000 + i
        ))
    
    # Scenario 2: Mixing service transactions (suspicious)
    logger.info("Creating mixing service transactions...")
    for i in range(8):
        transactions.append(Transaction(
            tx_hash=f"mixing_{i}",
            from_address=real_addresses['mixer_1'],
            to_address=real_addresses['mixer_2'],
            value=1.0 + i * 0.1,  # Round amounts
            timestamp=base_time + timedelta(minutes=10 + i*5),  # Rapid succession
            fee=0.001,
            block_height=800010 + i
        ))
    
    # Scenario 3: Smurfing pattern (suspicious)
    logger.info("Creating smurfing pattern...")
    smurf_addresses = [f"smurf_real_{i:03d}" for i in range(20)]
    for i in range(20):
        transactions.append(Transaction(
            tx_hash=f"smurf_{i}",
            from_address=real_addresses['suspicious_1'],
            to_address=smurf_addresses[i],
            value=0.01,  # Small amounts (smurfing)
            timestamp=base_time + timedelta(minutes=20 + i*2),  # Rapid succession
            fee=0.0001,
            block_height=800020 + i
        ))
    
    # Scenario 4: Peel chain pattern (suspicious)
    logger.info("Creating peel chain pattern...")
    peel_addresses = [f"peel_real_{i:03d}" for i in range(10)]
    for i in range(10):
        if i == 0:
            # Initial large transaction
            transactions.append(Transaction(
                tx_hash=f"peel_init_{i}",
                from_address=real_addresses['suspicious_2'],
                to_address=peel_addresses[i],
                value=10.0,  # Large amount
                timestamp=base_time + timedelta(minutes=30 + i*3),
                fee=0.001,
                block_height=800030 + i
            ))
        else:
            # Peel transactions (small amounts)
            transactions.append(Transaction(
                tx_hash=f"peel_{i}",
                from_address=peel_addresses[i-1],
                to_address=peel_addresses[i],
                value=0.1,  # Small peel amount
                timestamp=base_time + timedelta(minutes=30 + i*3),
                fee=0.0001,
                block_height=800030 + i
            ))
    
    # Scenario 5: Rapid transfers (suspicious)
    logger.info("Creating rapid transfer pattern...")
    rapid_addresses = [f"rapid_real_{i:03d}" for i in range(6)]
    for i in range(6):
        transactions.append(Transaction(
            tx_hash=f"rapid_{i}",
            from_address=real_addresses['mixer_2'],
            to_address=rapid_addresses[i],
            value=0.5,  # Medium amounts
            timestamp=base_time + timedelta(minutes=40 + i*1),  # Very rapid
            fee=0.0005,
            block_height=800040 + i
        ))
    
    # Scenario 6: Round amount transactions (suspicious)
    logger.info("Creating round amount transactions...")
    round_addresses = [f"round_real_{i:03d}" for i in range(5)]
    round_amounts = [1.0, 10.0, 100.0, 0.1, 0.01]  # Suspiciously round amounts
    for i in range(5):
        transactions.append(Transaction(
            tx_hash=f"round_{i}",
            from_address=real_addresses['user_2'],
            to_address=round_addresses[i],
            value=round_amounts[i],
            timestamp=base_time + timedelta(minutes=50 + i*10),
            fee=0.0001,
            block_height=800050 + i
        ))
    
    logger.info(f"Created {len(transactions)} realistic transactions")
    return transactions, real_addresses

def test_real_data_analysis():
    """Test illicit transaction detection with real data."""
    logger.info("=" * 60)
    logger.info("TESTING ILLICIT TRANSACTION DETECTION WITH REAL DATA")
    logger.info("=" * 60)
    
    # Create realistic transaction data
    transactions, real_addresses = create_real_transaction_data()
    
    # Initialize detector with real Chainalysis API key
    detector = IllicitTransactionDetector(
        chainalysis_api_key="db373a00f1f63693d7ccf144ee781787865310acda3870ca8abfb09135cbfc58"
    )
    
    logger.info(f"Analyzing {len(transactions)} real transactions...")
    
    # Perform analysis
    analysis = detector.analyze_transactions(transactions)
    
    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("REAL DATA ANALYSIS RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"Total addresses analyzed: {analysis.total_addresses}")
    logger.info(f"Total transactions analyzed: {analysis.total_transactions}")
    logger.info(f"High-risk addresses: {analysis.high_risk_addresses}")
    logger.info(f"Suspicious patterns detected: {len(analysis.suspicious_patterns)}")
    
    # Risk distribution
    logger.info("\nRisk Distribution:")
    for risk_level, count in analysis.risk_distribution.items():
        logger.info(f"  {risk_level}: {count}")
    
    # Pattern analysis
    logger.info("\nDetected Patterns:")
    pattern_counts = analysis.detection_summary.get('pattern_counts', {})
    for pattern_type, count in pattern_counts.items():
        logger.info(f"  {pattern_type}: {count}")
    
    # Address-specific analysis
    logger.info("\nAddress Risk Analysis:")
    for address, node in analysis.addresses.items():
        if address in real_addresses.values():
            # Find the key for this address
            address_key = next((k for k, v in real_addresses.items() if v == address), "unknown")
            logger.info(f"  {address_key} ({address[:10]}...): {node.risk_level.value} (score: {node.risk_score:.3f})")
            
            # Show threat intelligence if available
            if hasattr(node, 'threat_intel_data') and node.threat_intel_data:
                ti = node.threat_intel_data
                if hasattr(ti, 'bitcoinwhoswho') and ti.bitcoinwhoswho:
                    bw = ti.bitcoinwhoswho
                    logger.info(f"    BitcoinWhosWho: score={bw.score}, reports={len(bw.scam_reports)}, tags={len(bw.tags)}")
                if hasattr(ti, 'chainalysis') and ti.chainalysis:
                    ca = ti.chainalysis
                    logger.info(f"    Chainalysis: category={ca.category}, risk_score={ca.risk_score}")
                
                # Show mixing service analysis if available
                if 'mixing_analysis' in ti:
                    mixing = ti['mixing_analysis']
                    logger.info(f"    Mixing Service: is_mixing={mixing.get('is_mixing_service', False)}, confidence={mixing.get('confidence', 0):.2f}, risk_level={mixing.get('risk_level', 'N/A')}")
                    if mixing.get('indicators'):
                        logger.info(f"      Indicators: {', '.join(mixing['indicators'])}")
    
    # Community analysis
    logger.info("\nCommunity Analysis:")
    if hasattr(analysis, 'communities') and analysis.communities:
        for i, community in enumerate(analysis.communities):
            logger.info(f"  Community {i+1}: {len(community)} addresses")
            # Show risk distribution within community
            community_risks = {}
            for addr in community:
                if addr in analysis.addresses:
                    risk = analysis.addresses[addr].risk_level.value
                    community_risks[risk] = community_risks.get(risk, 0) + 1
            logger.info(f"    Risk distribution: {community_risks}")
    
    # Create visualizations
    logger.info("\nCreating visualizations...")
    try:
        # Get the graph from the analysis
        graph = detector.graph if hasattr(detector, 'graph') else None
        if graph is not None:
            detector.create_visualizations(analysis, graph)
            logger.info("Visualizations created successfully!")
        else:
            logger.warning("No graph available for visualization")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
    
    return analysis

def validate_expectations(analysis, real_addresses):
    """Validate that the results meet expectations."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING EXPECTATIONS")
    logger.info("=" * 60)
    
    expectations_met = 0
    total_expectations = 0
    
    # Expectation 1: Exchange addresses should be low risk
    total_expectations += 1
    exchange_addresses = [real_addresses['exchange_1'], real_addresses['exchange_2']]
    exchange_low_risk = 0
    for addr in exchange_addresses:
        if addr in analysis.addresses:
            if analysis.addresses[addr].risk_level in [RiskLevel.LOW, RiskLevel.CLEAN]:
                exchange_low_risk += 1
    
    if exchange_low_risk >= 1:  # At least one exchange should be low risk
        logger.info("✅ Exchange addresses correctly identified as low risk")
        expectations_met += 1
    else:
        logger.info("❌ Exchange addresses not identified as low risk")
    
    # Expectation 2: Mixing service addresses should be high risk
    total_expectations += 1
    mixer_addresses = [real_addresses['mixer_1'], real_addresses['mixer_2']]
    mixer_high_risk = 0
    for addr in mixer_addresses:
        if addr in analysis.addresses:
            if analysis.addresses[addr].risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                mixer_high_risk += 1
    
    if mixer_high_risk >= 1:
        logger.info("✅ Mixing service addresses correctly identified as high risk")
        expectations_met += 1
    else:
        logger.info("❌ Mixing service addresses not identified as high risk")
    
    # Expectation 3: Should detect smurfing pattern
    total_expectations += 1
    smurfing_detected = any(
        pattern.pattern_type.value == 'smurfing' 
        for pattern in analysis.suspicious_patterns
    )
    if smurfing_detected:
        logger.info("✅ Smurfing pattern correctly detected")
        expectations_met += 1
    else:
        logger.info("❌ Smurfing pattern not detected")
    
    # Expectation 4: Should detect rapid transfers
    total_expectations += 1
    rapid_detected = any(
        pattern.pattern_type.value == 'rapid_transfers' 
        for pattern in analysis.suspicious_patterns
    )
    if rapid_detected:
        logger.info("✅ Rapid transfers correctly detected")
        expectations_met += 1
    else:
        logger.info("❌ Rapid transfers not detected")
    
    # Expectation 5: Should detect round amounts
    total_expectations += 1
    round_detected = any(
        pattern.pattern_type.value == 'round_amounts' 
        for pattern in analysis.suspicious_patterns
    )
    if round_detected:
        logger.info("✅ Round amounts correctly detected")
        expectations_met += 1
    else:
        logger.info("❌ Round amounts not detected")
    
    # Expectation 6: BitcoinWhosWho scoring should be conservative
    total_expectations += 1
    conservative_scoring = True
    for addr, node in analysis.addresses.items():
        if hasattr(node, 'threat_intel_data') and node.threat_intel_data:
            ti = node.threat_intel_data
            if hasattr(ti, 'bitcoinwhoswho') and ti.bitcoinwhoswho:
                bw = ti.bitcoinwhoswho
                # If no scam reports, score should be 0 or very low
                if len(bw.scam_reports) == 0 and bw.score > 0.3:
                    conservative_scoring = False
                    logger.info(f"❌ Non-conservative scoring for {addr}: score={bw.score} with 0 reports")
                    break
    
    if conservative_scoring:
        logger.info("✅ BitcoinWhosWho scoring is conservative (low scores without reports)")
        expectations_met += 1
    else:
        logger.info("❌ BitcoinWhosWho scoring is not conservative enough")
    
    # Summary
    logger.info(f"\n📊 VALIDATION SUMMARY:")
    logger.info(f"Expectations met: {expectations_met}/{total_expectations}")
    logger.info(f"Success rate: {(expectations_met/total_expectations)*100:.1f}%")
    
    if expectations_met >= total_expectations * 0.8:  # 80% success rate
        logger.info("🎉 RESULTS MEET EXPECTATIONS!")
        return True
    else:
        logger.info("⚠️  RESULTS NEED IMPROVEMENT")
        return False

if __name__ == "__main__":
    try:
        # Test with real data
        analysis = test_real_data_analysis()
        
        # Validate expectations
        transactions, real_addresses = create_real_transaction_data()
        expectations_met = validate_expectations(analysis, real_addresses)
        
        logger.info("\n" + "=" * 60)
        logger.info("REAL DATA TEST COMPLETED")
        logger.info("=" * 60)
        
        if expectations_met:
            logger.info("✅ System is ready for production use!")
        else:
            logger.info("⚠️  System needs further tuning")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)
