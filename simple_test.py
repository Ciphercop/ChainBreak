#!/usr/bin/env python3
"""
Simple test for illicit analysis button without external connections.
"""

from illicit_transaction_detector import IllicitTransactionDetector, Transaction
from datetime import datetime
import random

def create_simple_test_data():
    """Create simple test data without external API calls."""
    transactions = []
    
    # Create some addresses
    addresses = [
        "1Suspicious1",  # Will be flagged as illicit
        "1Suspicious2",   # Will be flagged as suspicious  
        "1Clean1",
        "1Clean2",
        "1Clean3",
    ]
    
    # Create transactions with suspicious patterns
    for i in range(30):
        if i < 15:  # Suspicious transactions (small amounts, rapid transfers)
            from_addr = "1Suspicious1"
            to_addr = random.choice(["1Clean1", "1Clean2", "1Clean3"])
            value = random.uniform(0.001, 0.01)  # Small amounts
        elif i < 25:  # More suspicious
            from_addr = "1Suspicious2" 
            to_addr = random.choice(["1Clean1", "1Clean2"])
            value = random.uniform(0.01, 0.05)
        else:  # Normal transactions
            from_addr = random.choice(["1Clean1", "1Clean2"])
            to_addr = random.choice(["1Clean1", "1Clean2", "1Clean3"])
            value = random.uniform(0.1, 5.0)
            
        tx = Transaction(
            tx_hash=f"test_tx_{i:03d}",
            from_address=from_addr,
            to_address=to_addr,
            value=value,
            timestamp=datetime.now(),
            block_height=800000 + i
        )
        transactions.append(tx)
    
    return transactions

def test_illicit_detection():
    """Test the illicit detection functionality."""
    print("🚀 SIMPLE ILLICIT ANALYSIS TEST")
    print("=" * 40)
    
    # Create test data
    transactions = create_simple_test_data()
    print(f"✅ Created {len(transactions)} test transactions")
    
    # Initialize detector
    detector = IllicitTransactionDetector()
    
    # Run analysis (this will skip external API calls)
    print("🔍 Running illicit analysis...")
    analysis = detector.analyze_transactions(transactions)
    
    # Build graph
    graph = detector._build_transaction_graph(transactions)
    print(f"📊 Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Check addresses
    print("\n🎯 ADDRESS ANALYSIS:")
    for addr, node in analysis.addresses.items():
        print(f"  {addr}: Risk={node.risk_level.value}, Score={node.risk_score:.3f}")
    
    # Run illicit detection on nodes
    print("\n🔍 ILLICIT DETECTION RESULTS:")
    illicit_count = 0
    suspicious_count = 0
    
    for node in graph.nodes():
        address_node = analysis.addresses.get(node)
        if address_node:
            # Run illicit analysis
            node_analysis = detector.visualizer._analyze_node_illicit_patterns(node, graph, analysis)
            illicit_score = node_analysis['illicit_score']
            
            if illicit_score > 0.7:
                illicit_count += 1
                print(f"  🚨 ILLICIT: {node} (Score: {illicit_score:.3f})")
            elif illicit_score > 0.4:
                suspicious_count += 1
                print(f"  ⚠️  SUSPICIOUS: {node} (Score: {illicit_score:.3f})")
            else:
                print(f"  ✅ CLEAN: {node} (Score: {illicit_score:.3f})")
    
    print(f"\n📊 SUMMARY:")
    print(f"  🔴 Illicit addresses: {illicit_count}")
    print(f"  🟠 Suspicious addresses: {suspicious_count}")
    print(f"  ✅ Clean addresses: {len(analysis.addresses) - illicit_count - suspicious_count}")
    
    print("\n🎉 TEST COMPLETE!")
    return True

if __name__ == "__main__":
    test_illicit_detection()
