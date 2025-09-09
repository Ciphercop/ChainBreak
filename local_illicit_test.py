#!/usr/bin/env python3
"""
Local illicit analysis test - NO external connections!
"""

from illicit_transaction_detector import IllicitTransactionDetector, Transaction, RiskLevel
from datetime import datetime
import random

def create_local_test_data():
    """Create test data for local analysis only."""
    transactions = []
    
    # Create addresses with known patterns
    suspicious_addr = "1SuspiciousAddr"  # Will be flagged
    clean_addrs = ["1Clean1", "1Clean2", "1Clean3"]
    
    # Create suspicious pattern: many small transactions (smurfing)
    for i in range(20):
        tx = Transaction(
            tx_hash=f"suspicious_tx_{i:03d}",
            from_address=suspicious_addr,
            to_address=random.choice(clean_addrs),
            value=random.uniform(0.001, 0.01),  # Small amounts
            timestamp=datetime.now(),
            block_height=800000 + i
        )
        transactions.append(tx)
    
    # Create normal transactions
    for i in range(10):
        tx = Transaction(
            tx_hash=f"normal_tx_{i:03d}",
            from_address=random.choice(clean_addrs),
            to_address=random.choice(clean_addrs),
            value=random.uniform(0.1, 5.0),  # Normal amounts
            timestamp=datetime.now(),
            block_height=800100 + i
        )
        transactions.append(tx)
    
    return transactions

def run_local_illicit_analysis():
    """Run illicit analysis with NO external connections."""
    print("🚀 LOCAL ILLICIT ANALYSIS (NO EXTERNAL CONNECTIONS)")
    print("=" * 55)
    
    # Create test data
    transactions = create_local_test_data()
    print(f"✅ Created {len(transactions)} test transactions")
    
    # Initialize detector
    detector = IllicitTransactionDetector()
    
    # Build graph manually to avoid external calls
    print("🕸️  Building transaction graph...")
    graph = detector._build_transaction_graph(transactions)
    print(f"📊 Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Extract addresses manually
    print("👥 Extracting address nodes...")
    addresses = detector._extract_address_nodes(graph, transactions)
    print(f"📋 Extracted {len(addresses)} addresses")
    
    # Detect suspicious patterns manually
    print("🔍 Detecting suspicious patterns...")
    suspicious_patterns = detector._detect_suspicious_patterns(graph, addresses, transactions)
    print(f"🚨 Found {len(suspicious_patterns)} suspicious patterns")
    
    # Calculate risk scores manually (no external APIs)
    print("📊 Calculating risk scores...")
    risk_scores = detector._calculate_risk_scores(addresses, suspicious_patterns, graph)
    
    # Show results
    print("\n🎯 ADDRESS ANALYSIS RESULTS:")
    print("-" * 40)
    
    illicit_count = 0
    suspicious_count = 0
    
    for addr, node in addresses.items():
        risk_score = risk_scores.get(addr, 0.0)
        print(f"  {addr[:20]}...")
        print(f"    Risk Level: {node.risk_level.value}")
        print(f"    Risk Score: {risk_score:.3f}")
        print(f"    Patterns: {[p.value for p in node.suspicious_patterns]}")
        print(f"    TX Count: {node.transaction_count}")
        print()
        
        if node.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            illicit_count += 1
        elif node.risk_level == RiskLevel.MEDIUM:
            suspicious_count += 1
    
    print("📊 SUMMARY:")
    print(f"  🔴 High Risk: {illicit_count}")
    print(f"  🟠 Medium Risk: {suspicious_count}")
    print(f"  ✅ Low/Clean: {len(addresses) - illicit_count - suspicious_count}")
    
    # Test the illicit analysis function directly
    print("\n🔍 TESTING ILLICIT ANALYSIS FUNCTION...")
    print("-" * 40)
    
    for node in graph.nodes():
        address_node = addresses.get(node)
        if address_node:
            # Run illicit analysis
            node_analysis = detector.visualizer._analyze_node_illicit_patterns(node, graph, None)
            illicit_score = node_analysis['illicit_score']
            
            if illicit_score > 0.7:
                print(f"🚨 ILLICIT: {node[:20]}... (Score: {illicit_score:.3f})")
            elif illicit_score > 0.4:
                print(f"⚠️  SUSPICIOUS: {node[:20]}... (Score: {illicit_score:.3f})")
            else:
                print(f"✅ CLEAN: {node[:20]}... (Score: {illicit_score:.3f})")
    
    print("\n🎉 LOCAL ANALYSIS COMPLETE!")
    print("✅ No external connections made!")
    return True

if __name__ == "__main__":
    run_local_illicit_analysis()
